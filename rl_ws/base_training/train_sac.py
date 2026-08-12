#!/usr/bin/env python3
"""
train_sac.py — Entrenamiento con SAC. Espejo de train_fast.py: mismo env
(VecMujocoEnv), misma tarea (base_env), mismo config y las MISMAS metricas
(success rate global y por pista), para que los resultados sean comparables
contra PPO sin asteriscos.

Diferencia estructural con PPO: no hay "rollout -> update". Se recolecta
continuamente al buffer y se hacen SAC_UTD updates por paso de entorno. Una
"iteracion" aqui es solo un bloque de STEPS_PER_ENV pasos para poder imprimir
y comparar con la cadencia de train_fast.

Uso:
    cd rl_ws
    python3 base_training/train_sac.py --wandb
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
from rl_ws.base_training.sac import (
    Actor, Critic, ReplayBuffer, sac_update, to_env_action)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

C.CHECKPOINT_DIR.mkdir(exist_ok=True)


def train(n_envs=C.N_ENVS, steps_per_env=C.STEPS_PER_ENV, iters=C.ITERS,
          batch_size=C.SAC_BATCH_SIZE, lr=C.SAC_LR, utd=C.SAC_UTD,
          save_every=C.SAVE_EVERY, use_wandb=False, resume_from=None,
          device_str=C.DEVICE):

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available())
                          else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    venv = VecMujocoEnv(n_envs=n_envs)
    obs_dim = venv.obs_dim
    # El actor vive en [-1,1]; a la convencion del env se pasa via
    # to_env_action -- ver el docstring de sac.py.
    act_dim = C.SAC_ACT_DIM
    state_dim = obs_dim - C.HEATMAP_PIXELS ** 2

    actor = Actor(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
    critic = Critic(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
    critic_targ = Critic(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
    critic_targ.load_state_dict(critic.state_dict())
    for p in critic_targ.parameters():
        p.requires_grad_(False)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)

    start_iter = 0
    if resume_from and Path(resume_from).is_file():
        ck = torch.load(resume_from, map_location=device)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        critic_targ.load_state_dict(ck["critic_targ"])
        with torch.no_grad():
            log_alpha.copy_(ck["log_alpha"].to(device))
        opt_a.load_state_dict(ck["opt_a"]); opt_c.load_state_dict(ck["opt_c"])
        opt_alpha.load_state_dict(ck["opt_alpha"])
        start_iter = ck.get("iter", 0)
        print(f"Resumiendo desde iter {start_iter}")

    buf = ReplayBuffer(C.SAC_BUFFER_SIZE, obs_dim, act_dim, state_dim, device)
    print(f"Replay buffer: {C.SAC_BUFFER_SIZE} transiciones en {device} "
          f"(heatmap uint8, ~{C.SAC_BUFFER_SIZE*(state_dim*4*2 + (obs_dim-state_dim)*2)/1e9:.1f} GB)")

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=C.WANDB_PROJECT_SAC, config=dict(
            algo="SAC", n_envs=n_envs, batch_size=batch_size, lr=lr, utd=utd,
            buffer=C.SAC_BUFFER_SIZE, tau=C.SAC_TAU, gamma=C.SAC_GAMMA,
            target_entropy=C.SAC_TARGET_ENTROPY, reward_scale=C.SAC_REWARD_SCALE,
            obs_dim=obs_dim, act_dim=act_dim))

    T, N = steps_per_env, n_envs
    obs = venv.reset()
    env_steps = start_iter * T * N
    m = {"q": 0.0, "pi": 0.0, "alpha": 1.0, "entropy": 0.0}
    try:
        for it in range(start_iter, start_iter + iters):
            t0 = time.time()
            n_done = n_ok = 0
            for t in range(T):
                if env_steps < C.SAC_START_STEPS:
                    raw = np.random.uniform(-1, 1, (N, act_dim)).astype(np.float32)
                else:
                    with torch.no_grad():
                        a, _ = actor(torch.as_tensor(obs, dtype=torch.float32, device=device),
                                     with_logp=False)
                    raw = a.cpu().numpy()

                nobs, rew, done, infos = venv.step(to_env_action(raw))
                env_steps += N

                # La obs terminal la pisa el auto-reset: para el buffer hay que
                # usar la original (ver VecMujocoEnv.step). Y `done` solo marca
                # terminal REAL -- en truncaciones (limite de pasos, atascado)
                # el bootstrap debe seguir, si no se sesga el critico.
                buf_next = nobs.copy()
                buf_done = done.copy()
                for i, (d, inf) in enumerate(zip(done, infos)):
                    if d:
                        buf_next[i] = inf["terminal_obs"]
                        if inf.get("truncated"):
                            buf_done[i] = 0.0
                        n_done += 1
                        n_ok += int(bool(inf.get("reached")))
                # Escala solo para el critico: las metricas y avg_ret que se
                # reportan siguen en la escala original del env.
                buf.add(obs, raw, rew * C.SAC_REWARD_SCALE, buf_next, buf_done)
                obs = nobs

                if len(buf) >= C.SAC_LEARN_STARTS:
                    for _ in range(max(1, int(round(utd * N)))):
                        m = sac_update(actor, critic, critic_targ, log_alpha,
                                       opt_a, opt_c, opt_alpha, buf, batch_size,
                                       C.SAC_GAMMA, C.SAC_TAU, C.SAC_TARGET_ENTROPY)

            sr_iter = n_ok / max(n_done, 1)
            sr_100 = venv.success_rate()
            sr_track = venv.success_rate_by_track()
            avg = venv.avg_return()
            dt = time.time() - t0
            sps = T * N / dt
            print(f"[Iter {it:4d}] success={sr_100:5.1%} (100ep)  avg_ret={avg:8.2f}  "
                  f"reached={venv.n_reached:4d}  q={m['q']:.3f}  pi={m['pi']:+.2f}  "
                  f"alpha={m['alpha']:.3f}  ent={m['entropy']:+.2f}  ({dt:.1f}s, {sps:.0f} steps/s)")
            print(f"           episodios: {n_done:3d}  meta={n_ok:3d} ({sr_iter:.0%})  "
                  f"buffer={len(buf)}")
            if len(sr_track) > 1:
                print("           por pista: " + "  ".join(
                    f"{t}={r:.0%}" if r == r else f"{t}=--" for t, r in sr_track.items()))

            if use_wandb:
                log = {"iter": it, "avg_ep_r": avg, "n_reached": venv.n_reached,
                       "q_loss": m["q"], "pi_loss": m["pi"], "alpha": m["alpha"],
                       "entropy": m["entropy"], "steps_per_s": sps,
                       "success_rate": sr_100, "success_rate_iter": sr_iter,
                       "buffer": len(buf), "env_steps": env_steps}
                log.update({f"success_rate/{t}": r for t, r in sr_track.items() if r == r})
                wandb.log(log)

            if (it + 1) % save_every == 0:
                p = C.CHECKPOINT_DIR / f"sac_iter{it+1:05d}.pt"
                torch.save({"iter": it + 1, "actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "critic_targ": critic_targ.state_dict(),
                            "log_alpha": log_alpha.detach().cpu(),
                            "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                            "opt_alpha": opt_alpha.state_dict(),
                            "success_rate": sr_100}, p)
                print(f"  ↳ checkpoint: {p}")
    finally:
        venv.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS)
    ap.add_argument("--steps", type=int, default=C.STEPS_PER_ENV)
    ap.add_argument("--iters", type=int, default=C.ITERS)
    ap.add_argument("--batch", type=int, default=C.SAC_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.SAC_LR)
    ap.add_argument("--utd", type=float, default=C.SAC_UTD)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    train(n_envs=args.n_envs, steps_per_env=args.steps, iters=args.iters,
          batch_size=args.batch, lr=args.lr, utd=args.utd,
          use_wandb=args.wandb, resume_from=args.resume)
