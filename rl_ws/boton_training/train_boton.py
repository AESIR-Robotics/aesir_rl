#!/usr/bin/env python3
"""
train_boton.py — PPO para el brazo contra la torre de botones.

Espejo de base_training/train_fast.py: solo ORQUESTA. Crea el vec-env
(VecBotonEnv, N envs MuJoCo en hilos), la red (net.BetaActorCritic) y usa el
MISMO update de PPO, el MISMO GAE y la MISMA normalizacion de valor que la base
-- se importan tal cual de base_training.ppo, no se reimplementan. Lo unico
propio de esta tarea es la distribucion de la accion (Beta en vez de la hibrida
Normal+Beta de la base) y el tamaño de la observacion.

Pipeline:  train_boton -> net (accion) -> vec_boton [boton_env: fisica MuJoCo,
           obs, reward, castigos] -> train_boton

UNA diferencia deliberada con train_fast.py
===========================================
train_fast trata todos los `done` igual. Aqui NO: agotar el presupuesto de
tiempo es TRUNCAR, no terminar. Si se trata como terminal, el critico aprende
que quedarse sin tiempo "vale 0", y como la observacion no incluye el tiempo
restante el problema deja de ser markoviano: el mismo estado vale una cosa a
mitad de episodio y otra al final.

En la base esto casi no se nota (sus episodios acaban sobre todo por caida o
meta, que SI son terminales de verdad). Aqui el timeout es el fallo dominante
-- es literalmente como se define fallar un boton -- asi que sesgaria todo.
Solucion estandar: en el paso truncado se suma gamma*V(obs_terminal) al reward
y se deja done=1 para que GAE corte la traza.

Uso:
    cd rl_ws
    python3 -m boton_training.train_boton                    # entrenar
    python3 -m boton_training.train_boton --iters 200
    python3 -m boton_training.train_boton --resume ../checkpoints_boton/boton_best.pt

    # entrenar VIENDO la simulacion (abre el viewer sobre el env 0)
    python3 -m boton_training.train_boton --viewer
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RL_WS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_RL_WS)
for _p in (_RL_WS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_ws.base_training.ppo import (RunningMeanStd, compute_gae,  # noqa: E402
                                     ppo_update)
from boton_training import config as C           # noqa: E402
from boton_training.net import BetaActorCritic   # noqa: E402
from boton_training.vec_boton import VecBotonEnv  # noqa: E402
from boton_env import (BOTON_MAZE_REF, EPISODE_TIME_S,  # noqa: E402
                       HOLD_TIME_S)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def abrir_viewer(venv):
    """Viewer de MuJoCo enganchado al env 0 del vec-env.

    Se ve UN env de los N; los demas siguen corriendo sin dibujar. Lo que se ve
    es la politica CON ruido de exploracion (esta entrenando), asi que se mueve
    de forma mas erratica que la politica final -- para verla limpia, usar
    tests/test_boton.py --policy CKPT --render, que evalua con la media.

    Cuesta velocidad: el sync compite con los 8 envs por CPU.
    """
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
    try:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(venv.envs[0].model, venv.envs[0].data)
        v.cam.distance = 2.2
        v.cam.elevation = -20
        v.cam.azimuth = 150
        v.cam.lookat[:] = [0.35, 0.0, 0.45]
        print("[viewer] abierto sobre el env 0 (PYGLFW_LIBRARY_VARIANT=x11)")
        return v
    except Exception as exc:
        print(f"[viewer] no se pudo abrir: {exc}")
        return None


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(iters: int = C.ITERS,
          n_envs: int = C.N_ENVS,
          steps_per_env: int = C.STEPS_PER_ENV,
          targets=None,
          seed: int = 0,
          ckpt_dir: Path = C.CHECKPOINT_DIR,
          resume: str = "",
          device_str: str = C.DEVICE,
          use_wandb: bool = True,
          show_viewer: bool = False) -> None:

    device = pick_device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    venv = VecBotonEnv(n_envs=n_envs, targets=targets, seed=seed)
    viewer = abrir_viewer(venv) if show_viewer else None
    obs_dim, act_dim = venv.obs_dim, venv.act_dim
    T, N = steps_per_env, n_envs

    policy = BetaActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=C.LR)
    ret_rms = RunningMeanStd()

    start_iter, best = 0, -1e9
    if resume and Path(resume).is_file():
        ck = torch.load(resume, map_location=device)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        if "ret_rms" in ck:
            ret_rms.load_state_dict(ck["ret_rms"])
        start_iter = int(ck.get("iter", 0))
        best = float(ck.get("best", -1e9))
        print(f"[resume] desde {resume}  iter={start_iter}  best={best:.2f}")

    print(f"dispositivo={device}  obs={obs_dim}  act={act_dim}  "
          f"envs={N}  rollout={T}x{N}={T*N}  params={sum(p.numel() for p in policy.parameters())}")

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=C.WANDB_PROJECT, config=dict(
            n_envs=N, steps_per_env=T, lr=C.LR, gamma=C.GAMMA,
            gae_lambda=C.GAE_LAMBDA, clip=C.CLIP, ent_coef=C.ENT_COEF,
            vf_coef=C.VF_COEF, hidden=C.HIDDEN, seed=seed,
            targets=venv.targets, episode_time_s=EPISODE_TIME_S))

    b_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    b_act = np.zeros((T, N, act_dim), dtype=np.float32)
    b_logp = np.zeros((T, N), dtype=np.float32)
    b_rew = np.zeros((T, N), dtype=np.float32)
    b_val = np.zeros((T, N), dtype=np.float32)
    b_done = np.zeros((T, N), dtype=np.float32)

    # CURRICULO ADAPTATIVO de distractores: sube cuando la politica va bien y baja
    # cuando se atasca. Arranca en 0 (un solo boton) porque con distractores desde
    # el principio el 60% de los episodios moria por tocar un vecino y la politica
    # no llegaba ni a aprender a presionar.
    dificultad = 0.0
    venv.set_dificultad(dificultad)

    obs = venv.reset()
    try:
        for it in range(start_iter, start_iter + iters):
            t0 = time.time()
            n_done = n_ok = n_trunc = n_ajeno = 0

            for t in range(T):
                act, _, logp, val = policy.act_batch(obs, device)
                nobs, rew, done, infos = venv.step(act)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                b_obs[t], b_act[t], b_logp[t] = obs, act, logp
                b_rew[t], b_val[t], b_done[t] = rew, val, done
                obs = nobs

                # Bootstrapping de las TRUNCADAS (ver docstring del modulo).
                trunc_idx = [i for i, inf in enumerate(infos)
                             if done[i] and inf.get("truncated")]
                if trunc_idx:
                    term = np.stack([infos[i]["terminal_obs"] for i in trunc_idx])
                    with torch.no_grad():
                        _, _, v_term = policy(torch.as_tensor(
                            term, dtype=torch.float32, device=device))
                    # b_rew esta en la escala REAL del reward; el critico predice
                    # normalizado -> desnormalizar antes de sumarlo.
                    v_real = ret_rms.denormalize(v_term.squeeze(-1).cpu().numpy())
                    for k, i in enumerate(trunc_idx):
                        b_rew[t, i] += C.GAMMA * float(v_real[k])
                    n_trunc += len(trunc_idx)

                for d, inf in zip(done, infos):
                    if d:
                        n_done += 1
                        n_ok += int(bool(inf.get("success")))
                        n_ajeno += int(bool(inf.get("fin_boton_ajeno")))

            sr_iter = n_ok / max(n_done, 1)
            sr_100 = venv.success_rate()
            sr_by = venv.success_rate_by_target()

            with torch.no_grad():
                _, _, lv = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
            last_val = lv.squeeze(-1).cpu().numpy()

            adv, ret = compute_gae(b_rew, ret_rms.denormalize(b_val), b_done,
                                   ret_rms.denormalize(last_val), C.GAMMA, C.GAE_LAMBDA)
            ret_rms.update(ret)

            fobs = b_obs.reshape(T * N, obs_dim)
            fact = b_act.reshape(T * N, act_dim)
            flogp = b_logp.reshape(T * N)
            fadv = adv.reshape(T * N)
            fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
            fret = ret_rms.normalize(ret).reshape(T * N).astype(np.float32)
            m = ppo_update(policy, opt, fobs, fact, flogp, fadv, fret,
                           C.PPO_EPOCHS, C.BATCH_SIZE, C.CLIP, C.VF_COEF,
                           C.ENT_COEF, device)

            avg = venv.avg_return()
            dt = time.time() - t0
            print(f"[Iter {it:4d}] exito={sr_100:5.1%} (100ep)  avg_ret={avg:8.2f}  "
                  f"pulsados={venv.n_pressed:4d}  pi={m['pi']:+.4f}  v={m['v']:.3f}  "
                  f"ent={m['ent']:+.3f}  ({dt:.1f}s, {T*N/dt:.0f} pasos/s)")
            print(f"           episodios={n_done:3d}  exito={n_ok:3d} ({sr_iter:.0%})  "
                  f"truncados={n_trunc:3d}  boton_ajeno={n_ajeno:3d} "
                  f"({venv.wrong_button_rate():.0%} de 100ep)  "
                  f"t_medio={venv.avg_press_time_s():.1f}s")
            if sr_100 == sr_100:
                if sr_100 > 0.60 and dificultad < 1.0:
                    dificultad = min(1.0, dificultad + 0.10)
                    venv.set_dificultad(dificultad)
                elif sr_100 < 0.25 and dificultad > 0.0:
                    dificultad = max(0.0, dificultad - 0.05)
                    venv.set_dificultad(dificultad)

            pr_ = venv.progreso()
            if pr_:
                print(f"           progreso:  toca={pr_['toca']:.0%}  "
                      f"hunde={pr_['hunde']:.0%}  sostiene={pr_['sostiene']:.2f}s "
                      f"(mejor {pr_['sostiene_max']:.2f}s de {HOLD_TIME_S:.0f}s)")
            sbm = venv.success_by_mode()
            print("           por modo:  " + "  ".join(
                f"{k}={v:.0%}" if v == v else f"{k}=--" for k, v in sbm.items())
                + f"   dificultad={dificultad:.2f}")
            mix = venv.reason_mix()
            if mix:
                print("           motivos:   " + "  ".join(
                    f"{k}={v:.0%}" for k, v in mix.items() if v >= 0.01))

            if use_wandb:
                log = {"iter": it, "success_rate": sr_100, "avg_return": avg,
                       "policy_loss": m["pi"], "value_loss": m["v"],
                       "entropy": m["ent"], "n_pressed": venv.n_pressed,
                       "truncated": n_trunc, "wrong_button": n_ajeno,
                       "wrong_button_rate": venv.wrong_button_rate(),
                       "steps_per_s": T * N / dt}
                log.update({f"success/{k}": v for k, v in sbm.items()})
                log["dificultad"] = dificultad
                wandb.log(log, step=(it + 1) * T * N)

            def _save(path, tag):
                torch.save({"iter": it + 1, "policy": policy.state_dict(),
                            "optimizer": opt.state_dict(),
                            "ret_rms": ret_rms.state_dict(),
                            "best": best, "obs_dim": obs_dim, "act_dim": act_dim,
                            "targets": venv.targets}, path)
                print(f"           -> {tag}: {path}")

            if (it + 1) % C.SAVE_EVERY == 0:
                _save(ckpt_dir / f"boton_iter{it+1:05d}.pt", "checkpoint")
            if avg == avg and avg > best:
                best = avg
                _save(ckpt_dir / "boton_best.pt", f"NUEVO MEJOR ({avg:.2f})")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        venv.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=C.ITERS)
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS)
    ap.add_argument("--steps-per-env", type=int, default=C.STEPS_PER_ENV)
    ap.add_argument("--targets", type=int, nargs="*", default=C.TRAIN_TARGETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default=str(C.CHECKPOINT_DIR))
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--device", type=str, default=C.DEVICE)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--viewer", action="store_true",
                    help="abre el viewer de MuJoCo sobre el env 0 mientras entrena")
    a = ap.parse_args()
    train(iters=a.iters, n_envs=a.n_envs, steps_per_env=a.steps_per_env,
          targets=a.targets, seed=a.seed, ckpt_dir=Path(a.ckpt_dir),
          resume=a.resume, device_str=a.device, use_wandb=not a.no_wandb,
          show_viewer=a.viewer)
