#!/usr/bin/env python3
"""
train_fast.py — PPO vectorizado sobre MuJoCo DIRECTO (sin ROS), para entrenar
la base rapido. Usa VecMujocoEnv (N envs en threads) + la misma logica de tarea
(world_base) que el backend ROS, asi que la politica entrenada aca es la misma
que se despliega por mujoco_ros_bridge.py.

No necesita el bridge ni ROS corriendo. La red va en GPU; la fisica en CPU
(paralelizada por threads).

Uso:
    cd rl_ws
    python3 train_fast.py                       # 6 envs, defaults
    python3 train_fast.py --n-envs 6 --steps 512 --iters 2000
    python3 train_fast.py --wandb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Permite `python3 rl_ws/base_training/train_fast.py` desde cualquier lado:
# agrega la raiz del proyecto (aesir_rl) al path para resolver `rl_ws.*`.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import rl_ws.base_training.base_env as W
from rl_ws.base_training.mujoco_sim_base import VecMujocoEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

CHECKPOINT_DIR = Path(_ROOT) / "checkpoints_base"   # absoluto (no depende del CWD)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Red (misma arquitectura que train_base.py -> checkpoints intercambiables) ─
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, log_std_init: float = -0.5):
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
        z = self.trunk(obs)
        mu = torch.tanh(self.actor_mu(z))
        value = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        return (raw.cpu().numpy(), logp.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    def evaluate(self, obs, actions):
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, value, entropy


# ── PPO update (batch plano) ────────────────────────────────────────────────
def ppo_update(policy, opt, obs, actions, old_logp, adv, ret,
               epochs, batch, clip, vf_c, ent_c, device):
    obs     = torch.as_tensor(obs,     dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)
    n = obs.shape[0]

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(n)), batch, False):
            lp, val, ent = policy.evaluate(obs[idx], actions[idx])
            r  = torch.exp(lp - old_logp[idx])
            pl = -torch.min(r * adv_t[idx],
                            torch.clamp(r, 1 - clip, 1 + clip) * adv_t[idx]).mean()
            vl = F.smooth_l1_loss(val, ret_t[idx])
            loss = pl + vf_c * vl - ent_c * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            m["pi"] = pl.item(); m["v"] = vl.item(); m["ent"] = ent.item()
    return m


def compute_gae(rew, val, done, last_val, gamma, lam):
    """GAE por env (columnas). rew/val/done son (T, N); last_val es (N,)."""
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nextval = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nextval * nonterm - val[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    ret = adv + val
    return adv, ret


def train(n_envs=6, steps_per_env=512, iters=2000, ppo_epochs=10, batch_size=1024,
          gamma=0.99, gae_lambda=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.005,
          lr=3e-4, save_every=50, use_wandb=False, resume_from=None, device_str="auto"):

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available())
                          else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    venv = VecMujocoEnv(n_envs=n_envs)
    obs_dim, act_dim = venv.obs_dim, venv.act_dim

    policy = MLPActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    start_iter, best_avg = 0, -1e9
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        saved, msd = ckpt["policy"], policy.state_dict()
        compat = {k: v for k, v in saved.items() if k in msd and v.shape == msd[k].shape}
        skipped = [k for k in saved if k not in compat]
        msd.update(compat); policy.load_state_dict(msd)
        if skipped:
            print(f"[resume] arquitectura distinta: transferidos {len(compat)}, "
                  f"reinicializados {skipped}; optimizer/contadores de cero.")
        else:
            opt.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt.get("iter", 0); best_avg = ckpt.get("avg_ep_r", -1e9)
            print(f"Resumiendo desde iter {start_iter} (best={best_avg:.2f})")

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project="AIDL-PPO-AESIR-BASE-FAST", config=dict(
            n_envs=n_envs, steps_per_env=steps_per_env, ppo_epochs=ppo_epochs,
            batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda, clip=clip,
            vf_coef=vf_coef, ent_coef=ent_coef, lr=lr, obs_dim=obs_dim, act_dim=act_dim))

    T, N = steps_per_env, n_envs
    b_obs  = np.zeros((T, N, obs_dim), dtype=np.float32)
    b_act  = np.zeros((T, N, act_dim), dtype=np.float32)
    b_logp = np.zeros((T, N), dtype=np.float32)
    b_rew  = np.zeros((T, N), dtype=np.float32)
    b_val  = np.zeros((T, N), dtype=np.float32)
    b_done = np.zeros((T, N), dtype=np.float32)

    obs = venv.reset()
    try:
        for it in range(start_iter, start_iter + iters):
            t0 = time.time()
            for t in range(T):
                act, logp, val = policy.act_batch(obs, device)
                nobs, rew, done, _info = venv.step(act)
                b_obs[t], b_act[t], b_logp[t] = obs, act, logp
                b_rew[t], b_val[t], b_done[t] = rew, val, done
                obs = nobs

            with torch.no_grad():
                _, _, lv = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
            last_val = lv.squeeze(-1).cpu().numpy()

            adv, ret = compute_gae(b_rew, b_val, b_done, last_val, gamma, gae_lambda)
            # aplanar (T*N, ...) y normalizar ventaja
            fobs = b_obs.reshape(T * N, obs_dim)
            fact = b_act.reshape(T * N, act_dim)
            flogp = b_logp.reshape(T * N)
            fadv = adv.reshape(T * N); fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
            fret = ret.reshape(T * N)
            m = ppo_update(policy, opt, fobs, fact, flogp, fadv, fret,
                           ppo_epochs, batch_size, clip, vf_coef, ent_coef, device)

            avg = venv.avg_return()
            dt = time.time() - t0
            sps = T * N / dt
            print(f"[Iter {it:4d}] avg_ret={avg:8.2f}  reached={venv.n_reached:3d}  "
                  f"pi={m['pi']:+.4f}  v={m['v']:.3f}  ent={m['ent']:.3f}  "
                  f"({dt:.1f}s, {sps:.0f} steps/s)")

            if use_wandb:
                wandb.log({"iter": it, "avg_ep_r": avg, "n_reached": venv.n_reached,
                           "policy_loss": m["pi"], "value_loss": m["v"],
                           "entropy": m["ent"], "steps_per_s": sps})

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"fast_iter{it+1:05d}.pt"
                torch.save({"iter": it + 1, "policy": policy.state_dict(),
                            "optimizer": opt.state_dict(), "avg_ep_r": avg}, p)
                print(f"  ↳ checkpoint: {p}")
            if not np.isnan(avg) and avg > best_avg:
                best_avg = avg
                torch.save({"iter": it + 1, "policy": policy.state_dict(),
                            "optimizer": opt.state_dict(), "avg_ep_r": avg},
                           CHECKPOINT_DIR / "fast_best.pt")
    finally:
        venv.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=512, help="pasos por env por rollout")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", default=str(CHECKPOINT_DIR / "fast_best.pt"))
    #ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    train(n_envs=args.n_envs, steps_per_env=args.steps, iters=args.iters,
          batch_size=args.batch, lr=args.lr, use_wandb=args.wandb, resume_from=args.resume)
