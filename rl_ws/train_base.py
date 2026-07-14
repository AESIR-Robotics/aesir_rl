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
    cd rl_ws && MUJOCO_GL=glfw python3 mujoco_ros_bridge.py

Uso:
    cd rl_ws && python3 train_base.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Raiz del proyecto (aesir_rl) al path para resolver `rl_ws.*` corriendo directo.
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_ws.base_ros_env import BaseRosEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

CHECKPOINT_DIR = Path(_ROOT) / "checkpoints_base"   # absoluto (no depende del CWD)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────── Red (MLP, obs vectorial) ─────────────────────
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_dim  = act_dim

    def forward(self, obs):
        z       = self.trunk(obs)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())

    def evaluate(self, obs, actions):
        mu, std, value = self(obs)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return logp, value, entropy


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


# ──────────────────────────── PPO ──────────────────────────────────────────
def ppo_update(policy, opt, buf, adv, ret, epochs, batch, clip, vf_c, ent_c, device):
    obs     = torch.as_tensor(buf.obs,     dtype=torch.float32, device=device)
    actions = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_log = torch.as_tensor(buf.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)), batch, False):
            lp, val, ent = policy.evaluate(obs[idx], actions[idx])
            r  = torch.exp(lp - old_log[idx])
            pl = -torch.min(r * adv_t[idx],
                            torch.clamp(r, 1 - clip, 1 + clip) * adv_t[idx]).mean()
            vl = F.smooth_l1_loss(val, ret_t[idx])
            loss = pl + vf_c * vl - ent_c * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            m["pi"] = pl.item(); m["v"] = vl.item(); m["ent"] = ent.item()
    return m


# ──────────────────────────── Helpers ──────────────────────────────────────
def save_checkpoint(path, policy, opt, it, avg):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": opt.state_dict(), "avg_ep_r": avg}, path)


# ──────────────────────────── Training loop ────────────────────────────────
def train(num_iterations: int = 500,
          steps_per_iter: int = 1024,
          ppo_epochs:     int = 10,
          batch_size:     int = 256,
          gamma:          float = 0.99,
          gae_lambda:     float = 0.95,
          clip_param:     float = 0.2,
          vf_coef:        float = 0.5,
          ent_coef:       float = 0.005,
          lr:             float = 3e-4,
          save_every:     int = 25,
          device_str:     str = "auto",
          use_wandb:      bool = True,
          wandb_project:  str = "AIDL-PPO-AESIR-BASE",
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
                action, logp, val = policy.act(obs, device)
                nobs, rew, done, _info = env.step(action)
                buf.store(obs, action, logp, rew, val, done)
                obs = nobs
                ep_reward += rew
                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20: ep_history.pop(0)
                    ep_reward = 0.0
                    obs = env.reset()

            with torch.no_grad():
                _, _, lv = policy(torch.as_tensor(obs, dtype=torch.float32,
                                                  device=device).unsqueeze(0))
            adv, ret = buf.compute_gae(float(lv.item()), gamma, gae_lambda)
            m = ppo_update(policy, opt, buf, adv, ret,
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
    train(
        use_wandb=True,   # logea metricas a Weights & Biases (requiere `wandb login`)
        resume_from="./checkpoints_base/base_best.pt",
    )
