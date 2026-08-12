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
import mujoco
import mujoco.viewer

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MUJOCO_GL", "glfw")

import rl_ws.base_training.config as C
from rl_ws.base_training.mujoco_sim_base import VecMujocoEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

C.CHECKPOINT_DIR.mkdir(exist_ok=True)

from rl_ws.base_training.ppo import (ppo_update, compute_gae,  # sin MLPActorCritic
                                     RunningMeanStd)
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic


def train(n_envs=C.N_ENVS, steps_per_env=C.STEPS_PER_ENV, iters=C.ITERS,
          ppo_epochs=C.PPO_EPOCHS, batch_size=C.BATCH_SIZE,
          gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, clip=C.CLIP,
          vf_coef=C.VF_COEF, ent_coef=C.ENT_COEF, lr=C.LR,
          save_every=C.SAVE_EVERY, use_wandb=False, resume_from=None,
          device_str=C.DEVICE, seed=None, ckpt_dir=None, run_name=None,
          show_viewer=False):

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available())
                          else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Semilla: {seed}  (separa corridas; no reproducible bit a bit "
              f"por el paralelismo en threads)")

    # Directorio propio por corrida para que las semillas no se pisen los
    # checkpoints (todas escriben fast_iterNNNNN.pt / fast_best.pt).
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else C.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    venv = VecMujocoEnv(n_envs=n_envs)
    obs_dim, act_dim = venv.obs_dim, venv.act_dim

    viewer = None
    if show_viewer and hasattr(venv, "envs") and venv.envs:
        try:
            viewer = mujoco.viewer.launch_passive(venv.envs[0].model, venv.envs[0].data)
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            print("[viewer] MuJoCo viewer abierto")
        except Exception as exc:
            print(f"[viewer] no se pudo abrir: {exc}")
            viewer = None

    policy = CNNActorCritic(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    # El critico predice en espacio NORMALIZADO (ver RunningMeanStd en ppo.py).
    ret_rms = RunningMeanStd()
    start_iter, best_avg = 0, -1e9
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        saved, msd = ckpt["policy"], policy.state_dict()
        if "ret_rms" in ckpt:
            ret_rms.load_state_dict(ckpt["ret_rms"])
        else:
            # Checkpoint PRE-normalizacion: su cabeza de critico esta entrenada
            # para escupir retornos crudos (|W|~61 medido, 100x el actor). Con
            # objetivo normalizado esos pesos son basura y volverian a ahogar
            # el gradiente de la politica -> se descartan y se reinicializan.
            saved = {k: v for k, v in saved.items() if not k.startswith("critic.")}
            print("[resume] checkpoint sin ret_rms: descarto la cabeza del "
                  "critico (estaba en escala cruda) y la reinicializo.")
        compat = {k: v for k, v in saved.items() if k in msd and v.shape == msd[k].shape}
        skipped = [k for k in msd if k not in compat]
        dropped = [k for k in saved if k not in msd]
        msd.update(compat); policy.load_state_dict(msd)
        if dropped:
            print(f"[resume] el checkpoint trae {len(dropped)} tensores que esta "
                  f"red ya no tiene ({dropped}); se ignoran y el optimizer va de "
                  f"cero.")
        if skipped:
            print(f"[resume] transferidos {len(compat)}, "
                  f"reinicializados {skipped}; optimizer/contadores de cero.")
        elif not dropped:
            opt.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt.get("iter", 0); best_avg = ckpt.get("avg_ep_r", -1e9)
            print(f"Resumiendo desde iter {start_iter} (best={best_avg:.2f})")

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=C.WANDB_PROJECT, name=run_name, config=dict(
            seed=seed, tracks=list(C.ACTIVE_TRACKS),
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
    if viewer is not None and getattr(viewer, "is_running", lambda: False)():
        viewer.sync()
    try:
        for it in range(start_iter, start_iter + iters):
            t0 = time.time()
            n_done = n_ok = 0
            for t in range(T):
                act, raw, logp, val = policy.act_batch(obs, device)
                nobs, rew, done, infos = venv.step(act)
                if viewer is not None and getattr(viewer, "is_running", lambda: False)():
                    viewer.sync()
                b_obs[t], b_act[t], b_logp[t] = obs, act, logp
                b_rew[t], b_val[t], b_done[t] = rew, val, done
                obs = nobs

                for d, inf in zip(done, infos):
                    if d:
                        n_done += 1
                        n_ok += int(bool(inf.get("reached")))

            sr_iter = n_ok / max(n_done, 1)      # tasa cruda de ESTA iteracion
            sr_100 = venv.success_rate()          # movil, ultimos 100 episodios
            sr_track = venv.success_rate_by_track()   # movil, por pista

            with torch.no_grad():
                _, lv = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
            last_val = lv.squeeze(-1).cpu().numpy()

            # El critico predice normalizado; GAE trabaja en la escala real del
            # reward -> desnormalizar los valores ANTES, y volver a normalizar
            # el retorno DESPUES para usarlo como objetivo (ver ppo.py).
            adv, ret = compute_gae(b_rew, ret_rms.denormalize(b_val), b_done,
                                   ret_rms.denormalize(last_val), gamma, gae_lambda)
            ret_rms.update(ret)
            # aplanar (T*N, ...) y normalizar ventaja
            fobs = b_obs.reshape(T * N, obs_dim)
            fact = b_act.reshape(T * N, act_dim)
            flogp = b_logp.reshape(T * N)
            fadv = adv.reshape(T * N); fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
            fret = ret_rms.normalize(ret).reshape(T * N).astype(np.float32)
            m = ppo_update(policy, opt, fobs, fact, flogp, fadv, fret,
                           ppo_epochs, batch_size, clip, vf_coef, ent_coef, device)

            avg = venv.avg_return()
            dt = time.time() - t0
            sps = T * N / dt
            print(f"[Iter {it:4d}] success={sr_100:5.1%} (100ep)  avg_ret={avg:8.2f}  "
                  f"reached={venv.n_reached:4d}  pi={m['pi']:+.4f}  v={m['v']:.3f}  "
                  f"ent={m['ent']:.3f}  ({dt:.1f}s, {sps:.0f} steps/s)")
            print(f"           episodios: {n_done:3d}  meta={n_ok:3d} ({sr_iter:.0%})")
            if len(sr_track) > 1:      # con UNA pista el desglose == el global
                print("           por pista: " + "  ".join(
                    f"{t}={r:.0%}" if r == r else f"{t}=--" for t, r in sr_track.items()))

            if use_wandb:
                log = {"iter": it, "avg_ep_r": avg, "n_reached": venv.n_reached,
                       "policy_loss": m["pi"], "value_loss": m["v"],
                       "entropy": m["ent"], "steps_per_s": sps,
                       "success_rate": sr_100, "success_rate_iter": sr_iter}
                # nan = esa pista aun no termina ningun episodio; no lo mandamos
                # para no ensuciar la grafica de wandb con huecos.
                log.update({f"success_rate/{t}": r for t, r in sr_track.items() if r == r})
                wandb.log(log)

            # ret_rms va SIEMPRE con el checkpoint: sin el, el critico guardado
            # (que predice normalizado) se leeria en la escala equivocada.
            def _ckpt():
                return {"iter": it + 1, "policy": policy.state_dict(),
                        "optimizer": opt.state_dict(), "avg_ep_r": avg,
                        "ret_rms": ret_rms.state_dict()}

            if (it + 1) % save_every == 0:
                p = ckpt_dir / f"fast_iter{it+1:05d}.pt"
                torch.save(_ckpt(), p)
                print(f"  ↳ checkpoint: {p}")
            if not np.isnan(avg) and avg > best_avg:
                best_avg = avg
                torch.save(_ckpt(), ckpt_dir / "fast_best.pt")
    finally:
        if viewer is not None:
            viewer.close()
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
    ap.add_argument("--viewer", action="store_true", help="abrir el visualizador de MuJoCo")
    ap.add_argument("--resume", default="",
                    help='Checkpoint del que reanudar; por default DESDE CERO. '
                         'Antes el default apuntaba a un checkpoint concreto, lo '
                         'que arrancaba en caliente sin avisar -- veneno para un '
                         'barrido de semillas, donde cada corrida tiene que ser '
                         'independiente. Para reanudar, pasalo explicito.')
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla de numpy/torch. Separa corridas; no da "
                         "reproducibilidad bit a bit (envs en threads).")
    ap.add_argument("--ckpt-dir", default=None,
                    help="Directorio de checkpoints propio de esta corrida. ")
    ap.add_argument("--name", default=None, help="Nombre de la corrida en wandb.")
    args = ap.parse_args()
    train(n_envs=args.n_envs, steps_per_env=args.steps, iters=args.iters,
          batch_size=args.batch, lr=args.lr, use_wandb=args.wandb,
          resume_from=args.resume, seed=args.seed, ckpt_dir=args.ckpt_dir,
          run_name=args.name, show_viewer=args.viewer)
