from __future__ import annotations
import os
import sys
import math
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo_pallets_train import (
    AesirPalletsEnv,
    SPAWN_XYZ, GOAL_XY,
    EPISODE_MAX_STEPS, MAX_WHEEL_VEL, LIDAR_SPIN_VEL,
)
from global_navigator import GlobalNavigator, plan_route, quat_to_yaw

_HERE = os.path.dirname(os.path.abspath(__file__))
NAV_JSON = os.path.join(_HERE, "obstacles.json")

SPIN_THRESHOLD = 0.50      
FINISH_DIST = 0.60          

class TestEnv(AesirPalletsEnv):

    def _apply_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)
        v_norm = float(a[0])
        w_norm = float(a[1])

        vl = float(np.clip((v_norm - w_norm) * MAX_WHEEL_VEL, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
        vr = float(np.clip((v_norm + w_norm) * MAX_WHEEL_VEL, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))

        for i in self.ids_drive_l: self.data.ctrl[i] = vl
        for i in self.ids_drive_r: self.data.ctrl[i] = vr

        for fid in self.ids_flippers:
            self.data.ctrl[fid] = 0.0
        for wids in self.ids_flip_wh.values():
            for wid in wids:
                self.data.ctrl[wid] = 0.0

        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]
        self.data.ctrl[self.id_fing_l] = 0.0
        self.data.ctrl[self.id_fing_r] = 0.0

        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

    def reset(self) -> dict:
        import mujoco as _mj
        _mj.mj_resetData(self.model, self.data)
        self.data.qpos[0] = SPAWN_XYZ[0]
        self.data.qpos[1] = SPAWN_XYZ[1]
        self.data.qpos[2] = 0.25
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._set_arm_rest()
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL
        for _ in range(10):
            _mj.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
        self._step = 0
        self._stuck = 0
        self._fell = False
        base_pos = self.data.xpos[self.base_id]
        self._last_xy = base_pos[:2].copy()
        self._nav.reset(self._last_xy)
        return self._obs()

    def _terminated(self, completed: bool) -> bool:
        if completed: return True
        if self._step >= self.max_steps: return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.20: return True
        if self._fell: return True
        return False

def compute_action(guidance_obs: np.ndarray, k_speed: float = 0.60, k_turn: float = 1.00) -> np.ndarray:
    _, sin_t, cos_t = guidance_obs
    w_norm = float(np.clip(k_turn * sin_t, -1.0, 1.0))
    v_norm = float(np.clip(k_speed * cos_t, 0.0, 1.0)) if cos_t >= SPIN_THRESHOLD else 0.0

    action = np.zeros(14, dtype=np.float32)
    action[0] = v_norm
    action[1] = w_norm
    return action

def run(env: TestEnv, nav: GlobalNavigator, waypoints: list,
        max_steps: int, debug: bool, k_speed: float, k_turn: float) -> dict:

    obs = env.reset()
    robot_xy = env.data.xpos[env.base_id, :2].copy()
    nav.reset(robot_xy)

    t0 = time.time()
    total_reward = 0.0
    wp_log = []
    last_wp = -1
    n_wps = len(waypoints)

    print(f"\n  Reset position: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})")
    print(f"  Waypoints total: {n_wps} | Target goal: {waypoints[-1]}")
    print(f"  Gains: k_speed={k_speed}, k_turn={k_turn} | Spin threshold: cos > {SPIN_THRESHOLD}")

    if debug:
        print(f"\n  {'Step':>5}  {'X':>7} {'Y':>7}  {'Yaw':>6}  {'dn':>5}  {'sin':>5}  {'cos':>5}  {'v':>5}  {'w':>5}  WP  Mode")
        print(f"  {'─'*85}")
    else:
        print(f"  {'WP':^4}  {'Coordinates':^18}  {'Step':>5}  {'Time':>5}  Robot Position")
        print(f"  {'─'*56}")

    termination = "max_steps"

    for step in range(max_steps):
        robot_xy = env.data.xpos[env.base_id, :2].copy()
        robot_yaw = quat_to_yaw(env.data.xquat[env.base_id])

        guidance = nav.step(robot_xy, robot_yaw)
        guidance_obs = guidance["obs"]
        action = compute_action(guidance_obs, k_speed, k_turn)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        current_wp = guidance["wp"]
        if current_wp > last_wp:
            for wi in range(last_wp + 1, min(current_wp + 1, n_wps)):
                elapsed = time.time() - t0
                wp_xy = waypoints[wi]
                wp_log.append((wi, step, elapsed))
                if not debug:
                    print(f"  WP{wi:<3} ({wp_xy[0]:+.1f},{wp_xy[1]:+.1f})  {step:>5}  {elapsed:>4.1f}s  ({robot_xy[0]:.2f},{robot_xy[1]:.2f})")
            last_wp = current_wp

        if debug and step % 50 == 0:
            dn, st, ct = guidance_obs
            mode = "MOVE" if ct >= SPIN_THRESHOLD else "SPIN"
            print(f"  {step:>5}  {robot_xy[0]:>7.2f} {robot_xy[1]:>7.2f}  {math.degrees(robot_yaw):>5.1f}  {dn:.3f} {st:>+.3f} {ct:>+.3f}  {action[0]:.3f} {action[1]:>+.3f}  WP{guidance['wp']}  [{mode}]")

        final_wp = waypoints[-1]
        dist_final = np.linalg.norm(robot_xy - np.array(final_wp))
        if dist_final < FINISH_DIST:
            termination = "COMPLETED"
            elapsed = time.time() - t0
            print(f"\n  Goal waypoint reached at step={step} | Execution time={elapsed:.1f}s")
            break

        if done:
            zmat = env.data.xmat[env.base_id].reshape(3, 3)
            termination = "flipped" if float(zmat[2, 2]) < 0.20 else "timeout"
            break

    return {
        "n_wps": len(wp_log),
        "total_reward": total_reward,
        "steps": step + 1,
        "termination": termination,
        "elapsed_s": time.time() - t0,
    }

def print_result(r: dict, n_total: int):
    n = r["n_wps"]
    bar = "+" * n + "-" * (n_total - n)
    print(f"\n  [{bar}] {n}/{n_total} waypoints tracked")
    print(f"  Accumulated Reward : {r['total_reward']:.1f}")
    print(f"  Total Steps        : {r['steps']}")
    print(f"  Execution Time     : {r['elapsed_s']:.1f}s")
    print(f"  Termination Status : {r['termination']}")
    if r['termination'] == "COMPLETED":
        print("\n  TRAINING INFRASTRUCTURE READY FOR PPO RUN.")
    elif n == 0:
        print("\n  Tracking failed. Run with --debug flag to analyze guidance features.")

def main():
    parser = argparse.ArgumentParser(description="A* Path Following Verification Environment")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-steps", type=int, default=EPISODE_MAX_STEPS * 15)
    parser.add_argument("--json", default=NAV_JSON)
    parser.add_argument("--speed", type=float, default=0.60)
    parser.add_argument("--turn", type=float, default=1.00)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found. Run map_extractor.py first.")
        return

    print(f"\n{'═'*60}")
    print("  A* PATH FOLLOWING TEST ENVIRONMENT")
    print(f"{'═'*60}")
    print(f"  Mission Track: {SPAWN_XYZ[:2]} ──> {GOAL_XY}")

    print("\n  Generating global path via A*...")
    t0 = time.perf_counter()
    
    # Corrected call matching the new clean signature of global_navigator.py
    waypoints = plan_route(
        args.json,
        start_xy = (SPAWN_XYZ[0], SPAWN_XYZ[1]),
        goal_xy = GOAL_XY
    )
    
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Planned {len(waypoints)} precise curvature waypoints in {ms:.0f}ms")
    for i, (x, y) in enumerate(waypoints):
        mark = " <- GOAL" if i == len(waypoints) - 1 else ""
        print(f"    WP{i:2d}: ({x:+.2f}, {y:+.2f}){mark}")

    nav = GlobalNavigator(args.json, waypoints=waypoints)

    env = TestEnv(render=not args.no_render, max_steps=args.max_steps)
    try:
        result = run(env, nav, waypoints, args.max_steps, args.debug, args.speed, args.turn)
        print_result(result, len(waypoints))
    finally:
        env.close()

if __name__ == "__main__":
    main()