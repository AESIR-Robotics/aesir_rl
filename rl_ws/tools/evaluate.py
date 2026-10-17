#!/usr/bin/env python3
"""
evaluate.py -- Evaluacion OFFLINE de un checkpoint, sin gradientes ni entrenar.

Existe porque el paper necesita numeros sobre pistas que no estan en
ACTIVE_TRACKS (E2, zero-shot) y sobre backends distintos (E6, paridad ROS),
y ni train_fast.py ni watch_checkpoint.py sirven: el primero entrena, el
segundo es visual y de un episodio.

El criterio de exito es EXACTAMENTE el del entrenamiento -- `info["reached"]`,
que lo pone `terminated()` -- para que estos numeros sean comparables con las
curvas de los .log sin traducir nada.

Muestreo: por defecto SE MUESTREA de la politica, igual que en el rollout de
entrenamiento, para que el numero sea comparable con el success de los logs.
Con --deterministic se usa la media de cada distribucion (Normal: loc;
Beta: a/(a+b)), que es lo que se desplegaria en el robot.

Uso:
    python3 rl_ws/utils/evaluate.py \
        --ckpt runs/paper_archive/e1p_seed1/fast_iter00300.pt \
        --tracks steps1m steps --episodes 100 --n-envs 12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
# Solo la raiz del repo, y todo se importa por su ruta de PAQUETE:
# ppo_cnn_extractor hace `from . import config`, asi que cargarlo como modulo
# suelto (con base_training/ en el path) revienta con "no known parent package".
sys.path.insert(0, str(_HERE.parent.parent))

from rl_ws.base_training import config as C                          # noqa: E402
from rl_ws.base_training.mujoco_sim_base import VecMujocoEnv         # noqa: E402
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic     # noqa: E402


def load_policy(ckpt_path: Path, device) -> CNNActorCritic:
    """Carga los pesos con la MISMA tolerancia que el --resume de train_fast:
    solo transfiere tensores cuyo nombre Y forma coinciden. Si el checkpoint es
    de otra arquitectura hay que enterarse aqui, no por un numero raro al final,
    asi que lo que no encaja se reporta en voz alta."""
    policy = CNNActorCritic(C.OBS_DIM, C.ACT_DIM, map_pixels=C.HEATMAP_PIXELS).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    saved, msd = ckpt["policy"], policy.state_dict()
    compat = {k: v for k, v in saved.items() if k in msd and v.shape == msd[k].shape}
    missing = [k for k in msd if k not in compat]
    if missing:
        raise SystemExit(
            f"El checkpoint no encaja con la arquitectura actual.\n"
            f"  faltan {len(missing)} tensores: {missing[:6]}{'...' if len(missing) > 6 else ''}\n"
            f"  revisa HIDDEN / HEATMAP_PIXELS / OBS_DIM en config.py")
    msd.update(compat)
    policy.load_state_dict(msd)
    policy.eval()
    print(f"  checkpoint: {ckpt_path}  (iter {ckpt.get('iter', '?')})")
    return policy


@torch.no_grad()
def act(policy, obs_np, device, deterministic: bool):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    d_vw, d_flip, _ = policy._dists(obs)
    if deterministic:
        a_vw, a_flip = d_vw.mean, d_flip.mean
    else:
        a_vw, a_flip = d_vw.sample(), d_flip.sample()
    a = torch.cat([a_vw.clamp(-1.0, 1.0), a_flip.clamp(0.0, 1.0)], dim=-1)
    return a.cpu().numpy()


class RosBackend:
    """Adapta BaseRosEnv (un env, sin auto-reset) a la interfaz que consume
    rollout(): N envs, obs apilada, reset automatico al terminar, e `info` con
    las claves 'track' y 'reason' que el backend directo si trae.

    Ademas CRONOMETRA cada step. Esa es la medicion de E6 y hay que leerla con
    cuidado: BaseRosEnv.step() duerme dt=1/CONTROL_HZ y DESPUES hace el trabajo
    (feedback, guia, reward, heatmap, build_obs), asi que el tiempo de pared por
    paso es `dt + trabajo`. Un lazo que cumple 20 Hz mediria 50 ms; lo que
    exceda de ahi es el presupuesto que el bridge y la observacion se comen."""

    def __init__(self, track_name: str):
        from rl_ws.base_training.base_ros_env import BaseRosEnv
        self.env = BaseRosEnv()
        self.track = track_name
        self.tracks = [track_name]
        self.n = 1
        self.step_ms: list[float] = []

    def reset(self):
        return np.asarray(self.env.reset(), dtype=np.float32)[None]

    def step(self, actions):
        import time as _t
        t0 = _t.perf_counter()
        obs, rew, done, info = self.env.step(actions[0])
        self.step_ms.append((_t.perf_counter() - t0) * 1e3)
        info = dict(info)
        info["track"] = self.track
        if done and not info.get("reason"):
            info["reason"] = "META alcanzada" if info.get("reached") else "fin de episodio"
        if done:
            obs = self.env.reset()
        return (np.asarray(obs, dtype=np.float32)[None],
                np.array([rew], dtype=np.float32),
                np.array([float(done)]), [info])

    def close(self):
        self.env.close()


def rollout(env, policy, device, episodes_per_track: int, deterministic: bool,
            max_steps: int):
    """Rueda hasta que CADA pista acumule `episodes_per_track` episodios.

    Cuenta por pista y no en total porque los envs se reparten round-robin: con
    un tope global, una pista con episodios cortos aportaria el triple de
    muestras que otra y la media saldria ponderada por la duracion del episodio,
    que es justo lo que no se quiere medir."""
    eps = defaultdict(list)
    tracks = set(env.tracks)
    obs = env.reset()
    steps = 0
    while True:
        obs, _, dones, infos = env.step(act(policy, obs, device, deterministic))
        steps += 1
        # El fin de episodio se detecta por `done`, no por la presencia de
        # info["reason"]: el backend ROS no devuelve esa clave (la imprime), y
        # keyear por ella se saltaria TODOS sus episodios en silencio.
        for inf, d in zip(infos, dones):
            if not d:
                continue
            t = inf.get("track", "?")
            if len(eps[t]) < episodes_per_track:
                eps[t].append({"reached": bool(inf.get("reached")),
                               "wp": int(inf.get("wp", 0)),
                               "reason": inf.get("reason", "")})
        if all(len(eps[t]) >= episodes_per_track for t in tracks):
            break
        if steps > max_steps:
            print(f"  AVISO: tope de {max_steps} pasos alcanzado; "
                  f"pistas incompletas: "
                  f"{ {t: len(eps[t]) for t in tracks if len(eps[t]) < episodes_per_track} }")
            break
    return dict(eps)


def report(eps: dict, label: str):
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    print(f"{'pista':<12}{'n':>5}{'exito':>9}{'wp medio':>11}   modo de fallo dominante")
    out = {}
    for t in sorted(eps):
        E = eps[t]
        if not E:
            continue
        sr = float(np.mean([e["reached"] for e in E]))
        wp = float(np.mean([e["wp"] for e in E]))
        fails = [e["reason"] for e in E if not e["reached"]]
        top = max(set(fails), key=fails.count) if fails else "-"
        print(f"{t:<12}{len(E):>5}{sr:>8.1%}{wp:>11.1f}   {top[:34]}")
        out[t] = {"n": len(E), "success": sr, "wp_mean": wp,
                  "fail_modes": {r: fails.count(r) for r in set(fails)},
                  # Por episodio, no solo la media: en una pista de mision FIJA
                  # (steps2, pallets) el histograma de wp dice DONDE muere la
                  # politica a lo largo de la ruta, que es la unica pregunta
                  # interesante cuando el exito es 0%.
                  "wp_hist": [e["wp"] for e in E]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tracks", nargs="+", required=True,
                    help="pistas a evaluar; pueden NO estar en ACTIVE_TRACKS")
    ap.add_argument("--episodes", type=int, default=100,
                    help="episodios POR PISTA (default 100)")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--deterministic", action="store_true",
                    help="media de la distribucion en vez de muestrear")
    ap.add_argument("--max-steps", type=int, default=60000,
                    help="tope de pasos de rollout, por si una pista no termina")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("sim", "ros"), default="sim",
                    help="'sim' = MuJoCo directo y vectorizado; 'ros' = "
                         "BaseRosEnv sobre el bridge (E6, un env, tiempo real)")
    ap.add_argument("--out", type=Path, default=None, help="volcar JSON")
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}   muestreo: "
          f"{'determinista (media)' if a.deterministic else 'estocastico'}")

    policy = load_policy(a.ckpt, device)

    # UNA pista a la vez, con TODOS los envs. Si se meten varias en un mismo
    # VecMujocoEnv, el rollout no puede parar hasta que la mas lenta llegue a su
    # cuota, y una pista donde nunca se gana (pallets: todos los episodios
    # corren hasta el tope de pasos) bloquea a las demas durante todo ese
    # tiempo con el resto de envs ya terminados y girando en vacio.
    eps, timing = {}, None
    if a.backend == "ros":
        if len(a.tracks) != 1:
            raise SystemExit("el backend ROS simula UNA pista, la de "
                             "ACTIVE_TRACKS[0]; pasa solo esa en --tracks")
        env = RosBackend(a.tracks[0])
        eps.update(rollout(env, policy, device, a.episodes, a.deterministic,
                           a.max_steps))
        ms = np.array(env.step_ms)
        budget = 1000.0 / C.CONTROL_HZ
        # El TRABAJO es lo que interesa, no el tiempo total: step() duerme el
        # presupuesto entero y luego trabaja, asi que "pasos por encima del
        # presupuesto" es siempre 100% y no informa de nada. Lo que dice si el
        # sistema cabe en el robot es el EXCESO sobre la espera.
        work = ms - budget
        timing = {"n_steps": int(ms.size), "budget_ms": budget,
                  "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
                  "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max()),
                  "work_p50_ms": float(np.percentile(work, 50)),
                  "work_p95_ms": float(np.percentile(work, 95)),
                  "work_max_ms": float(work.max()),
                  "headroom_frac": float(1.0 - np.percentile(work, 95) / budget),
                  "effective_hz": float(1000.0 / ms.mean())}
        env.close()
    else:
        for t in a.tracks:
            env = VecMujocoEnv(n_envs=a.n_envs, tracks=[t], verbose=False)
            eps.update(rollout(env, policy, device, a.episodes, a.deterministic,
                               a.max_steps))
            del env
    res = report(eps, f"{a.ckpt.parent.name} — {a.episodes} episodios por pista")

    if timing:
        print(f"\n{'='*62}\nPRESUPUESTO DEL LAZO  (objetivo {timing['budget_ms']:.1f} ms "
              f"= {C.CONTROL_HZ:.0f} Hz)\n{'='*62}")
        print(f"  tiempo por paso:  media {timing['mean_ms']:.1f} ms   "
              f"p50 {timing['p50_ms']:.1f}   p95 {timing['p95_ms']:.1f}   "
              f"max {timing['max_ms']:.1f}")
        print(f"  TRABAJO (exceso sobre la espera de {timing['budget_ms']:.0f} ms): "
              f"p50 {timing['work_p50_ms']:.1f} ms   p95 {timing['work_p95_ms']:.1f}   "
              f"max {timing['work_max_ms']:.1f}")
        print(f"  holgura sobre el presupuesto (p95): {timing['headroom_frac']:.1%}")
        print(f"  frecuencia efectiva: {timing['effective_hz']:.1f} Hz")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"ckpt": str(a.ckpt), "deterministic": a.deterministic,
             "episodes": a.episodes, "seed": a.seed, "backend": a.backend,
             "loop_timing": timing, "tracks": res}, indent=2))
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
