"""
train_base.py — Entrena la base (oruga + flippers) con PPO. Script general
del pipeline: red, buffer, PPO update y bucle de entrenamiento.

Toda la logica de "mundo" (navegador global, observacion, reward, reset/step
sobre el bridge ROS2) vive en BaseRosEnv (base_env.py) — este script solo la
usa, igual que el ppo_conv_train.py original usaba AesirMuJoCoEnv.

Pipeline (visto desde aqui):

    obs = env.reset()
    loop:
        action, logp, val = policy.act(obs)
        obs, reward, done, info = env.step(action)   # <- ROS2 + bridge + nav, ver base_env.py
        buffer.store(...)
        si listo: ppo_update(...)

REQUIERE, en OTRA terminal, el bridge corriendo:
    cd rl_ws && MUJOCO_GL=glfw python3 base_training/mujoco_sim_rosbridge.py

Uso:
    cd rl_ws && python3 base_training/train_base.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# Raiz del proyecto (aesir_rl) al path para resolver `rl_ws.*` corriendo directo.
# (base_training/ -> rl_ws/ -> aesir_rl/)
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_ws.base_training.base_ros_env import BaseRosEnv
import rl_ws.base_training.config as C
from rl_ws.base_training.ppo import MLPActorCritic, ppo_update

try:
    import wandb
    _HAS_WANDB = callable(getattr(wandb, "init", None))
    if not _HAS_WANDB:
        print("[train_base] WARNING: imported wandb module has no init(), disabling wandb logging.")
except ImportError:
    _HAS_WANDB = False

CHECKPOINT_DIR = C.CHECKPOINT_DIR   # absoluto (no depende del CWD)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────── Buffer (obs vectorial) ───────────────────────
@dataclass
class RolloutBuffer:
    capacity: int
    obs_dim:  int
    act_dim:  int

    obs:     np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    logps:   np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values:  np.ndarray = field(init=False)
    dones:   np.ndarray = field(init=False)
    idx:     int = 0

    def __post_init__(self):
        self.obs     = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.logps   = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values  = np.zeros(self.capacity, dtype=np.float32)
        self.dones   = np.zeros(self.capacity, dtype=np.float32)

    def store(self, obs, action, logp, reward, value, done):
        i = self.idx
        self.obs[i]     = obs
        self.actions[i] = action
        self.logps[i]   = logp
        self.rewards[i] = reward
        self.values[i]  = value
        self.dones[i]   = float(done)
        self.idx = (self.idx + 1) % self.capacity

    def compute_gae(self, last_val: float, gamma: float, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv = last_val if t == self.capacity - 1 else self.values[t + 1]
            nd = 1.0 - self.dones[t]
            d  = self.rewards[t] + gamma * nv * nd - self.values[t]
            gae = d + gamma * lam * nd * gae
            adv[t] = gae
        ret = adv + self.values
        return (adv - adv.mean()) / (adv.std() + 1e-8), ret


# ──────────────────────────── Helpers ──────────────────────────────────────
def save_checkpoint(path, policy, opt, it, avg):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": opt.state_dict(), "avg_ep_r": avg}, path)


# ──────────────────────────── Training loop ────────────────────────────────
def train(num_iterations: int = C.ROS_ITERS,
          steps_per_iter: int = C.ROS_STEPS_PER_ITER,
          ppo_epochs:     int = C.PPO_EPOCHS,
          batch_size:     int = C.ROS_BATCH_SIZE,
          gamma:          float = C.GAMMA,
          gae_lambda:     float = C.GAE_LAMBDA,
          clip_param:     float = C.CLIP,
          vf_coef:        float = C.VF_COEF,
          ent_coef:       float = C.ENT_COEF,
          lr:             float = C.LR,
          save_every:     int = C.ROS_SAVE_EVERY,
          device_str:     str = C.DEVICE,
          use_wandb:      bool = True,
          wandb_project:  str = C.WANDB_PROJECT_ROS,
          resume_from:    Optional[str] = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    env = BaseRosEnv()
    print(f"obs_dim = {env.obs_dim}  act_len = {env.act_len}")

    policy = MLPActorCritic(env.obs_dim, env.act_len).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    start_iter, best_avg = 0, -1e9
    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0)
        best_avg   = ckpt.get("avg_ep_r", -1e9)
        print(f"Resumiendo desde iter {start_iter} (best={best_avg:.2f})")

    buf = RolloutBuffer(steps_per_iter, env.obs_dim, env.act_len)

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=wandb_project, config=dict(
            steps_per_iter=steps_per_iter, ppo_epochs=ppo_epochs, batch_size=batch_size,
            gamma=gamma, gae_lambda=gae_lambda, clip_param=clip_param, vf_coef=vf_coef,
            ent_coef=ent_coef, lr=lr, obs_dim=env.obs_dim, act_dim=env.act_len))

    obs = env.reset()
    ep_reward = 0.0
    ep_history: List[float] = []

    try:
        for it in range(start_iter, start_iter + num_iterations):
            t0 = time.time()
            for _ in range(steps_per_iter):
                action, raw, logp, val = policy.act(obs, device)
                nobs, rew, done, _info = env.step(action)
                buf.store(obs, raw, logp, rew, val, done)
                obs = nobs
                ep_reward += rew
                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20: ep_history.pop(0)
                    ep_reward = 0.0
                    obs = env.reset()

            with torch.no_grad():
                _, lv = policy(torch.as_tensor(obs, dtype=torch.float32,
                                               device=device).unsqueeze(0))
            adv, ret = buf.compute_gae(float(lv.item()), gamma, gae_lambda)
            m = ppo_update(policy, opt, buf.obs, buf.actions, buf.logps, adv, ret,
                           ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)

            avg = float(np.mean(ep_history)) if ep_history else float("nan")
            dt  = time.time() - t0
            print(f"[Iter {it:4d}] avg_ep_r={avg:8.2f}  pi={m['pi']:+.4f}  "
                  f"v={m['v']:.4f}  ent={m['ent']:.3f}  ({dt:.1f}s)")

            if use_wandb:
                wandb.log({"iter": it, "avg_ep_r": avg, "policy_loss": m["pi"],
                           "value_loss": m["v"], "entropy": m["ent"], "iter_time_s": dt})

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"base_iter{it+1:05d}.pt"
                save_checkpoint(p, policy, opt, it + 1, avg)
                print(f"  ↳ checkpoint: {p}")
            if ep_history and avg > best_avg:
                best_avg = avg
                save_checkpoint(CHECKPOINT_DIR / "base_best.pt", policy, opt, it + 1, avg)
                print(f"  ↳ NUEVO MEJOR ({avg:.2f})")

    finally:
        env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Entrena la base con PPO sobre el bridge ROS2.")
    # OJO con el default: entrenar DESDE CERO. Antes esto estaba fijo en
    # base_best.pt, o sea que cada corrida reanudaba la anterior sin decirlo — y
    # si esa anterior aprendio algo malo (p.ej. la que corrio con el maze roto,
    # donde is_fallen() mataba el episodio en el paso 1: avg_ep_r=-240 y accion
    # media ~0, "no moverse"), la corrida nueva HEREDA esa politica y el robot
    # se queda quieto aunque la pista ya este arreglada. Reanudar ahora es
    # explicito: --resume <ruta>.
    ap.add_argument("--resume", default="",
                    help="Checkpoint del que reanudar. Vacio (default) = desde cero. "
                         "Ej: --resume ../checkpoints_base/base_best.pt")
    ap.add_argument("--iters", type=int, default=C.ROS_ITERS)
    ap.add_argument("--wandb", action="store_true",
                    help="logea metricas a Weights & Biases (requiere `wandb login`)")
    args = ap.parse_args()

    train(num_iterations=args.iters, use_wandb=args.wandb,
          resume_from=(args.resume or None))
