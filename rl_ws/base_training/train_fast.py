#!/usr/bin/env python3
"""
train_fast.py — Script principal (main) del entrenamiento rapido de la base.

Solo ORQUESTA: crea el vec-env (VecMujocoEnv, N envs MuJoCo en threads), la red
y el update de PPO (ppo.py), corre el loop rollout -> GAE -> update, y maneja
checkpoints/resume/wandb. Los parametros default salen todos de config.py; los
flags de CLI solo los sobreescriben para experimentar.

Pipeline:  train_fast -> ppo (accion) -> mujoco_sim_base [robot_control ->
           mj_step -> global_navigator -> base_env (obs/reward)] -> train_fast

No necesita el bridge ni ROS corriendo. La red va en GPU; la fisica en CPU
(paralelizada por threads).

Uso:
    cd rl_ws
    python3 base_training/train_fast.py                # defaults de config.py
    python3 base_training/train_fast.py --n-envs 6 --steps 512 --iters 2000
    python3 base_training/train_fast.py --wandb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import rl_ws.base_training.config as C
from rl_ws.base_training.mujoco_sim_base import VecMujocoEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

C.CHECKPOINT_DIR.mkdir(exist_ok=True)

from rl_ws.base_training.ppo import ppo_update, compute_gae   # sin MLPActorCritic
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic


def train(n_envs=C.N_ENVS, steps_per_env=C.STEPS_PER_ENV, iters=C.ITERS,
          ppo_epochs=C.PPO_EPOCHS, batch_size=C.BATCH_SIZE,
          gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, clip=C.CLIP,
          vf_coef=C.VF_COEF, ent_coef=C.ENT_COEF, lr=C.LR,
          save_every=C.SAVE_EVERY, use_wandb=False, resume_from=None,
          device_str=C.DEVICE):

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available())
                          else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    venv = VecMujocoEnv(n_envs=n_envs)
    obs_dim, act_dim = venv.obs_dim, venv.act_dim

    policy = CNNActorCritic(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
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
        wandb.init(project=C.WANDB_PROJECT, config=dict(
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
                action, raw, logp, val = policy.act_batch(obs, device)
                nobs, rew, done, _info = venv.step(action)
                b_obs[t], b_act[t], b_logp[t] = obs, raw, logp
                b_rew[t], b_val[t], b_done[t] = rew, val, done
                obs = nobs

            with torch.no_grad():
                _, lv = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
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
                p = C.CHECKPOINT_DIR / f"fast_iter{it+1:05d}.pt"
                torch.save({"iter": it + 1, "policy": policy.state_dict(),
                            "optimizer": opt.state_dict(), "avg_ep_r": avg}, p)
                print(f"  ↳ checkpoint: {p}")
            if not np.isnan(avg) and avg > best_avg:
                best_avg = avg
                torch.save({"iter": it + 1, "policy": policy.state_dict(),
                            "optimizer": opt.state_dict(), "avg_ep_r": avg},
                           C.CHECKPOINT_DIR / "fast_best.pt")
    finally:
        venv.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS)
    ap.add_argument("--steps", type=int, default=C.STEPS_PER_ENV, help="pasos por env por rollout")
    ap.add_argument("--iters", type=int, default=C.ITERS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", default=str(C.CHECKPOINT_DIR / "fast_iter01000.pt"))
    #ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    train(n_envs=args.n_envs, steps_per_env=args.steps, iters=args.iters,
          batch_size=args.batch, lr=args.lr, use_wandb=args.wandb, resume_from=args.resume)
