#!/usr/bin/env python3
"""Evaluate the base + arm mission in a single MuJoCo world (MissionEnv), no ROS.

Failures are reported per stage (fail_stage):
  base_to_tower  the base policy does not reach the pre-dock point
  dock           the scripted controller does not reach the press pose
  arm            the arm does not press the button
  base_to_goal   the base does not reach the final goal after pressing

Examples (from the repo root):
  # full mission with pretrained policies
  MUJOCO_GL=egl .venv/bin/python3 rl_ws/tests/test_fsm.py \\
      --base-checkpoint checkpoints_base/remote_20261010/fast_best.pt \\
      --arm-checkpoint checkpoints_boton/boton_best.pt --heatmap legacy --episodes 5

  # FSM checkpoint from train_fsm.py, random tower
  ... --fsm-checkpoint checkpoints_fsm/<run>/fsm_best.pt --randomize-tower

  # isolate the arm: the chassis is placed exactly in front of the tower
  ... --dock teleport
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL_WS = HERE.parent
ROOT = RL_WS.parent
for path in (ROOT, RL_WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fsm_training.mission_env import MissionEnv
from fsm_training.policies import load_arm, load_base


class Policy:
    """Deterministic (mean) or sampled actions from a base or arm agent."""

    def __init__(self, agent):
        self.agent = agent
        self.agent.net.eval()

    def __call__(self, obs: np.ndarray, stochastic: bool = False) -> np.ndarray:
        if stochastic:          # sample, as in the train_fsm rollout
            return self.agent.act_batch(np.asarray(obs, np.float32)[None])[0][0]
        return self.agent.mean_action(obs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--arm-checkpoint", default=None)
    ap.add_argument("--fsm-checkpoint", default=None,
                    help="train_fsm.py checkpoint (holds base AND arm); "
                         "--base-checkpoint/--arm-checkpoint override it")
    ap.add_argument("--heatmap", choices=("current", "legacy"), default=None,
                    help="base heatmap encoding (legacy = server 50788fee). "
                         "Default: the one stored in the FSM checkpoint, else 'current'")
    ap.add_argument("--randomize-tower", action="store_true",
                    help="sample the tower and approach heading every episode")
    ap.add_argument("--track", default="flat")
    ap.add_argument("--tower", type=float, nargs=2, default=(2.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--approach-yaw-deg", type=float, default=0.0)
    ap.add_argument("--button", type=int, default=0)
    ap.add_argument("--dock", choices=("scripted", "teleport"), default="scripted")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--backoff", type=float, default=None, help="back-off after pressing (m)")
    ap.add_argument("--tower-plan-half", type=float, default=None,
                    help="tower half-box for the planner (m)")
    ap.add_argument("--stochastic", choices=("none", "arm", "base", "both"), default="none",
                    help="agents that SAMPLE their distribution instead of using the mean")
    args = ap.parse_args()

    base_ck = args.base_checkpoint or args.fsm_checkpoint
    arm_ck = args.arm_checkpoint or args.fsm_checkpoint
    if not base_ck or not arm_ck:
        ap.error("--fsm-checkpoint or --base-checkpoint + --arm-checkpoint is required")

    if args.heatmap is None:
        args.heatmap = "current"
        if base_ck == args.fsm_checkpoint:
            ck = torch.load(base_ck, map_location="cpu", weights_only=False)
            args.heatmap = ck.get("heatmap_encoding", "current")
            del ck
    print(f"[env] heatmap={args.heatmap}")

    np.random.seed(args.seed)          # platform spawn/goal use np.random
    torch.manual_seed(args.seed)
    st_base = args.stochastic in ("base", "both")
    st_arm = args.stochastic in ("arm", "both")
    device = torch.device(args.device)
    base_pi = Policy(load_base(base_ck, device)[0])
    arm_pi = Policy(load_arm(arm_ck, device)[0])
    if base_pi.agent.legacy and args.heatmap == "current":
        print("[warning] legacy base with 'current' heatmap is out of distribution "
              "(measured on flat: 0/6 vs 5/6 with --heatmap legacy)")
    env = MissionEnv(track=args.track, tower_xy=tuple(args.tower),
                     approach_yaw=np.radians(args.approach_yaw_deg),
                     target_button=args.button, dock_mode=args.dock,
                     max_steps=args.max_steps, seed=args.seed, verbose=not args.quiet,
                     randomize_tower=args.randomize_tower, heatmap_encoding=args.heatmap,
                     **({"backoff_dist": args.backoff} if args.backoff is not None else {}),
                     **({"tower_plan_half": args.tower_plan_half}
                        if args.tower_plan_half is not None else {}))
    outcomes = Counter()
    milestones = Counter()
    try:
        for ep in range(args.episodes):
            agent, obs = env.reset()
            ret = {"base": 0.0, "arm": 0.0}
            n = {"base": 0, "arm": 0}
            done = False
            while not done:
                action = base_pi(obs, st_base) if agent == "base" else arm_pi(obs, st_arm)
                agent, obs, r, done, info = env.step(action)
                ret[info["agent"]] += r
                n[info["agent"]] += 1
            outcomes[info["fail_stage"] or "SUCCESS"] += 1
            for k in ("reached_tower", "docked", "pressed"):
                milestones[k] += int(bool(info.get(k)))
            extra = ""
            if "dock_err_long" in info:
                extra += (f" dock_err=({info['dock_err_long']*100:+.1f}cm,"
                          f"{info['dock_err_lat']*100:+.1f}cm,{info['dock_err_yaw_deg']:+.1f}deg)")
            if "arm_reason" in info:
                extra += f" arm={info.get('arm_reason')}"
            if info["fail_stage"] in ("base_to_tower", "base_to_goal"):
                fx = info["final_xy"]
                extra += (f" base_reason={info.get('base_reason')} final_xy={np.round(fx, 2)}"
                          f" goal={np.round(info['final_goal'], 2)}"
                          f" tower={np.round(info['tower_xy'], 2)}"
                          f" d_tower={np.linalg.norm(fx - info['tower_xy']):.2f}m")
            print(f"[ep {ep + 1}] success={info['success']} fail_stage={info['fail_stage']} "
                  f"steps base={n['base']} arm={n['arm']} return base={ret['base']:.1f} "
                  f"arm={ret['arm']:.1f} min_dist_predock={info['min_dist_predock']:.2f}m"
                  f"{extra}", flush=True)
    finally:
        env.close()
    n = max(args.episodes, 1)
    print(f"\nsummary ({args.episodes} episodes): {dict(outcomes)}")
    print(f"stages: reached_tower={milestones['reached_tower']}/{n}  docked={milestones['docked']}/{n}  "
          f"pressed={milestones['pressed']}/{n}  final_goal={outcomes['SUCCESS']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
