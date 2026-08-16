#!/usr/bin/env python3
"""
watch_checkpoint.py — Visor standalone de un checkpoint entrenado, UN solo env
(sin VecMujocoEnv/threads) para evitar el segfault del viewer combinado con el
ThreadPoolExecutor de train_fast.py en esta maquina (Wayland/XWayland+amdgpu).

Uso:
    cd rl_ws
    PYGLFW_LIBRARY_VARIANT=x11 python3 base_training/watch_checkpoint.py \
        --ckpt ../checkpoints_base/fast_best.pt
"""
from __future__ import annotations
import argparse, os, sys, time

os.environ.setdefault("MUJOCO_GL", "glfw")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import mujoco
import mujoco.viewer

import rl_ws.base_training.config as C
from rl_ws.base_training.mujoco_sim_base import BaseMujocoEnv
from rl_ws.base_training.map_context import MapContext
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.CHECKPOINT_DIR / "fast_best.pt"))
    ap.add_argument("--track", default=C.ACTIVE_TRACKS[0])
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    track = dict(C.TRACK_DEFS[args.track])
    track["name"] = args.track
    map_ctx = None
    if getattr(C, "USE_HEATMAP", False):
        map_ctx = MapContext(bt_path=track["bt"], resolution=C.OCTOMAP_RESOLUTION,
                             radius_m=C.HEATMAP_RADIUS_M, patch_pixels=C.HEATMAP_PIXELS)
    # BaseMujocoEnv puro, SIN VecMujocoEnv/ThreadPoolExecutor: combinar el
    # visor GLFW con el pool de threads de VecMujocoEnv segfaultea en esta
    # maquina (probado con train_fast.py --viewer, incluso con n_envs=1) --
    # ver conversacion. Un solo env en el hilo principal SI es estable.
    env = BaseMujocoEnv(waypoints=None, map_ctx=map_ctx, track=track)
    obs = env.reset()

    device = torch.device("cpu")
    policy = CNNActorCritic(C.OBS_DIM, C.ACT_DIM, map_pixels=C.HEATMAP_PIXELS).to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        print(f"[watch] checkpoint cargado: {args.ckpt} (iter={ckpt.get('iter', '?')})")
    else:
        print(f"[watch] sin checkpoint valido ({args.ckpt}); politica sin entrenar")
    policy.eval()

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            while not done and viewer.is_running():
                act, _, _, _ = policy.act(obs, device)
                obs, rew, done, info = env.step(act)
                ep_ret += rew
                viewer.sync()
                time.sleep(env._dt * env.decim)
            print(f"[watch] episodio {ep}: ret={ep_ret:.1f} reason={info.get('reason')}")
            if not viewer.is_running():
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
