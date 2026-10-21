"""
mission_env.py — Base + arm + FSM on a single MjModel / MjData.

Both policies act on the same robot in one world
(models/aesir_complete_flat_tower.xml), taking turns according to the mission
phase:

  NAVIGATE_TO_TOWER  base policy -> pre-dock point, planned with the tower as
                     an obstacle
  DOCK               scripted controller -> exact press pose (tower at
                     TOWER_REL_XY from the chassis, as in the arm's training),
                     flippers stowed, robot stopped
  PRESS_BUTTON       arm policy, chassis welded (BotonArmEnv)
  RETRACT            scripted: release the weld, arm to rest, back off
  NAVIGATE_TO_END    base policy -> final goal
  DONE

Docking is scripted because boton_best.pt was trained with the tower 0.55 m in
front of the chassis (±6 cm lateral jitter, ±20° tower yaw), while the base
policy stops within FINISH_DIST=0.2 m of its goal with an uncontrolled heading.
Neither policy covers those last centimetres.

Interface (one agent acts per step; scripted phases run inside step()):

    agent, obs = env.reset()                    # agent: "base" | "arm"
    agent, obs, reward, done, info = env.step(action_of_active_agent)

`reward` belongs to the agent that acted (info["agent"]); each sub-env keeps
its own reward. info["segment_end"] marks the end of that agent's segment (its
"done" for GAE); info["fail_stage"] tells where the episode failed.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import mujoco
import numpy as np

import rl_ws.base_training.config as C
import rl_ws.base_training.base_env as W
from rl_ws.base_training.map_context import MapContext
from rl_ws.base_training.mujoco_sim_base import BaseMujocoEnv
from rl_ws.global_navigator import build_platform_zone, plan_track_route
from boton_env import BotonArmEnv, FLIPPER_STOW_RAD

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
WORLDS = {
    "flat": os.path.normpath(os.path.join(_MODELS, "aesir_complete_flat_tower.xml")),
}

# Tower pose relative to the chassis used to train boton_best.pt (measured in
# BotonArmEnv.reset(): tower at x=0.55 in torre_botones.xml, chassis settled at
# the origin with yaw ~0).
TOWER_REL_XY = np.array([0.55, 0.0])
TOWER_HALF = 0.25          # tower half-box (with buttons) for the planner
PREDOCK_BACK = 0.60        # m behind the press pose where the base route ends
SPAWN_MIN_DIST = 2.0       # minimum spawn/final-goal distance to the tower (m)

# Docking. Commands measured in this world: turning in place has a track
# friction deadband (w < ~1.2 rad/s does not turn; 1.5 -> ~6 deg/s,
# 2.5 -> ~17 deg/s) and w >= 3 rad/s in place makes the simulation unstable
# (QACC NaN). Achieved linear speed is about half the command.
# Longitudinal target: 2 cm closer than the nominal pose. Measured with
# dock_mode="teleport" over a sweep of offsets: at the nominal pose boton_best.pt
# is erratic (1 cm changes the outcome from 117 steps to a 700-step timeout);
# between +1.5 and +2.0 cm it presses almost always in ~80 steps. Beyond +2.0 cm
# the chassis stops advancing (contact with the tower base), so the approach
# also stops when it stalls.
DOCK_LON_TARGET = 0.02
DOCK_TOL_LONG = 0.015      # m along the approach axis, around DOCK_LON_TARGET
DOCK_TOL_LAT = 0.01        # m lateral (at 1.5 cm boton_best.pt already fails sometimes)
DOCK_TOL_YAW = np.radians(1.5)
DOCK_MAX_STEPS = 1200      # 60 s at 20 Hz, across all attempts
DOCK_ATTEMPTS = 3
TURN_W_MIN, TURN_W_MAX = 1.5, 2.5       # rad/s commanded when turning in place
APPROACH_V = 0.10          # commanded m/s (~0.05 m/s achieved)
SETTLE_STEPS = 30          # standing still before welding
RETRACT_STEPS = 40         # arm back to rest
BACKOFF_DIST = 0.5         # m reversed before replanning to the final goal
BACKOFF_V = 0.15           # m/s
# Per-axis range (m) for the tower when randomize_tower=True. With 5 m the
# pre-dock stays within 5 + 0.55 + 0.60 = 6.15 m of the centre, inside the
# platform (PLATFORM_HALF_EXTENT=10) and the spawn/goal area (±8/±9).
TOWER_XY_RANGE = 5.0


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _flipper_u(rad: float) -> float:
    """Flipper angle -> [0,1] action of the current base (see base_env.flipper_targets)."""
    return float((rad - C.FLIPPER_MIN_RAD) / (C.FLIPPER_MAX_RAD - C.FLIPPER_MIN_RAD))


class MissionEnv:
    def __init__(self,
                 track: str = "flat",
                 tower_xy: Tuple[float, float] = (2.0, 0.0),
                 approach_yaw: float = 0.0,
                 target_button: int = 0,
                 dock_mode: str = "scripted",
                 max_steps: int = 6000,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 randomize_tower: bool = False,
                 tower_xy_range: float = TOWER_XY_RANGE,
                 heatmap_encoding: str = "current",
                 backoff_dist: float = BACKOFF_DIST,
                 tower_plan_half: float = TOWER_HALF):
        """approach_yaw: robot heading while pressing (the tower is ahead).
        dock_mode: "scripted" (real controller) or "teleport" (places the
        chassis at the exact pose; only to isolate the arm in diagnostics).
        randomize_tower: if True, EVERY reset() samples the tower within
        ±tower_xy_range (m, per axis) and the approach heading in [-pi, pi);
        tower_xy/approach_yaw are then only the initial pose.
        heatmap_encoding: "current" (robot-relative ±HEATMAP_Z_RANGE_M window)
        or "legacy" (global octomap z_min/z_max, the encoding the legacy base
        checkpoints were trained with; see legacy_heatmap.py).
        backoff_dist / tower_plan_half: back-off after pressing and tower
        half-box for the planner. The base does not see the tower (it is not
        in the heatmap octomap): with fast_best, base_to_goal failures are the
        robot stuck against the tower while going around it."""
        if track not in WORLDS:
            raise ValueError(f"track '{track}' has no mission world (available: {sorted(WORLDS)})")
        if dock_mode not in ("scripted", "teleport"):
            raise ValueError("dock_mode must be 'scripted' or 'teleport'")
        if heatmap_encoding not in ("current", "legacy"):
            raise ValueError("heatmap_encoding must be 'current' or 'legacy'")
        self.dock_mode = dock_mode
        self.max_steps = max_steps
        self.randomize_tower = bool(randomize_tower)
        self.tower_xy_range = float(tower_xy_range)
        self.heatmap_encoding = heatmap_encoding
        self.backoff_dist = float(backoff_dist)
        self.tower_plan_half = float(tower_plan_half)
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        td = dict(C.TRACK_DEFS[track])
        td["name"] = track
        td["xml"] = WORLDS[track]
        self.track = td
        surface_z = C.PLATFORM_SURFACE_Z if td["kind"] == "platform" else 0.0

        self.model = mujoco.MjModel.from_xml_path(td["xml"])
        self._tower_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torre_botones")
        self._surface_z = surface_z

        map_ctx = None
        if getattr(C, "USE_HEATMAP", False):
            map_ctx = MapContext(bt_path=td["bt"], resolution=C.OCTOMAP_RESOLUTION,
                                 radius_m=C.HEATMAP_RADIUS_M, patch_pixels=C.HEATMAP_PIXELS,
                                 z_range_m=C.HEATMAP_Z_RANGE_M)
            if heatmap_encoding == "legacy":
                from fsm_training.legacy_heatmap import LegacyHeatmapMapContext
                map_ctx = LegacyHeatmapMapContext.from_map_context(map_ctx)
        zone = build_platform_zone() if td["kind"] == "platform" else None
        self.base = BaseMujocoEnv(None, model=self.model, map_ctx=map_ctx,
                                  platform_zone=zone, track=td, max_steps=10 ** 9)
        # The arm shares the model AND the data: the tower it presses is the
        # one the base avoids.
        self.arm = BotonArmEnv(model=self.model, data=self.base.data,
                               randomize_torre=False, force_mode="torre",
                               fixed_target=target_button)
        self.data = self.base.data
        self.set_tower_pose(tower_xy, approach_yaw)

        self.agent = "base"
        self.phase = "DONE"

    def set_tower_pose(self, tower_xy, approach_yaw: float) -> None:
        """Move the tower (body of the shared MjModel) and recompute the press
        pose, the pre-dock point and the planner's tower box.

        BotonArmEnv's reference pose (_torre_pos0/_torre_quat0, stored in its
        __init__) must be updated too: with randomize_torre=False,
        _colocar_botones() restores the tower to that pose on every
        reset_from_current_state(), which would move it back right before
        pressing."""
        self.tower_xy = np.asarray(tower_xy, dtype=np.float64).copy()
        self.approach_yaw = float(approach_yaw)
        pos = np.array([self.tower_xy[0], self.tower_xy[1], self._surface_z])
        quat = np.array([np.cos(0.5 * approach_yaw), 0.0, 0.0, np.sin(0.5 * approach_yaw)])
        self.model.body_pos[self._tower_bid] = pos
        self.model.body_quat[self._tower_bid] = quat
        self.arm._torre_pos0 = pos.copy()
        self.arm._torre_quat0 = quat.copy()

        # Press pose: the chassis pose that puts the tower at TOWER_REL_XY.
        u = np.array([np.cos(approach_yaw), np.sin(approach_yaw)])
        n = np.array([-u[1], u[0]])
        self._u, self._n = u, n
        self.press_xy = self.tower_xy - TOWER_REL_XY[0] * u - TOWER_REL_XY[1] * n
        self.predock_xy = self.press_xy - PREDOCK_BACK * u
        t = self.tower_xy
        h = self.tower_plan_half
        self._tower_box = (t[0] - h, t[1] - h, t[0] + h, t[1] + h)
        mujoco.mj_forward(self.model, self.data)

    # ── helpers ────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[mission] {msg}")

    def _park_loose_buttons(self) -> None:
        for k, b in enumerate(self.arm._b_libre):
            self.arm._aparcar(b, self.arm._g_libre[k], False)
        mujoco.mj_forward(self.model, self.data)

    def _retarget_base(self, goal_xy: np.ndarray) -> np.ndarray:
        """Change the base goal (route, goal_xy, reward state) and return the
        policy observation towards it."""
        b = self.base
        fb = b._feedback()
        waypoints, _ = plan_track_route(fb["xy"], tuple(goal_xy), track=self.track,
                                        start_yaw=float(fb["yaw"]),
                                        overlay=[self._tower_box])
        if not waypoints:
            raise RuntimeError(f"no route {fb['xy']} -> {goal_xy}")
        b.goal_xy = np.asarray(goal_xy, dtype=np.float64)
        b.nav.replan(waypoints, obstacles=[])
        b.nav.reset(fb["xy"])
        guidance = b.nav.step(fb["xy"], fb["yaw"])
        b._rs.reset(fb["xy"], float(np.linalg.norm(fb["xy"] - np.asarray(guidance["target"]))),
                    fb["yaw"])
        b._ep_steps = 0
        return W.build_obs(guidance, fb, b._get_heatmap(fb))

    def _sample_final_goal(self) -> np.ndarray:
        r = C.GOAL_XY_RANGE
        for _ in range(200):
            g = self.rng.uniform(-r, r, 2)
            if (np.linalg.norm(g - self.tower_xy) >= SPAWN_MIN_DIST
                    and np.linalg.norm(g - self.press_xy) >= C.GOAL_MIN_DIST_M):
                return g
        raise RuntimeError("could not sample a final goal")

    def _base_cmd(self, v_mps: float, w_radps: float, flip_rad: float) -> dict:
        """One base control step with a scripted command. Returns feedback."""
        a = np.array([v_mps / C.V_MAX_MPS, w_radps / C.W_MAX_RADPS]
                     + [_flipper_u(flip_rad)] * 4, dtype=np.float32)
        self.base.step(a)
        return self.base._feedback()

    def _pose_error(self, fb: dict) -> Tuple[float, float, float]:
        e = fb["xy"] - self.press_xy
        return float(e @ self._u), float(e @ self._n), _wrap(self.approach_yaw - fb["yaw"])

    # ── scripted phases ────────────────────────────────────────────────────
    def _turn_to(self, yaw_target: float, tol: float, budget: list) -> dict:
        """Turn in place until |error| < tol. budget: [remaining steps]."""
        fb = self.base._feedback()
        while budget[0] > 0:
            e = _wrap(yaw_target - fb["yaw"])
            if abs(e) < tol:
                break
            w = np.sign(e) * np.clip(TURN_W_MIN + 2.0 * abs(e), TURN_W_MIN, TURN_W_MAX)
            fb = self._base_cmd(0.0, float(w), FLIPPER_STOW_RAD)
            budget[0] -= 1
        for _ in range(5):                      # let the turn's inertia die out
            fb = self._base_cmd(0.0, 0.0, FLIPPER_STOW_RAD)
        return fb

    def _dock(self) -> Tuple[bool, dict]:
        b = self.base
        b.goal_xy = self.press_xy.copy()
        fb = b._feedback()
        ok = True
        if self.dock_mode == "teleport":
            a = b._base_qadr
            self.data.qpos[a:a + 2] = self.press_xy + DOCK_LON_TARGET * self._u
            y = self.approach_yaw
            self.data.qpos[a + 3:a + 7] = [np.cos(0.5 * y), 0.0, 0.0, np.sin(0.5 * y)]
            self.data.qvel[b._base_vadr:b._base_vadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            b.ctrl.sync_from_data()
        else:
            # Stow the flippers FIRST: stowing them at the end pushes the
            # chassis ~2 cm forward and undoes the docking.
            for _ in range(20):
                fb = self._base_cmd(0.0, 0.0, FLIPPER_STOW_RAD)
            budget = [DOCK_MAX_STEPS]
            ok = False
            for attempt in range(DOCK_ATTEMPTS):
                lon, lat, _ = self._pose_error(fb)
                if lon > -0.10 or abs(lat) > 0.10:
                    # Badly placed to enter the approach line: go back to a point on the axis.
                    entry = self.press_xy - 0.35 * self._u
                    d = entry - fb["xy"]
                    fb = self._turn_to(float(np.arctan2(d[1], d[0])) + (np.pi if lon > -0.10 else 0.0),
                                       np.radians(3.0), budget)
                    sgn = -1.0 if lon > -0.10 else 1.0
                    while budget[0] > 0 and np.linalg.norm(entry - fb["xy"]) > 0.03:
                        fb = self._base_cmd(sgn * APPROACH_V, 0.0, FLIPPER_STOW_RAD)
                        budget[0] -= 1
                        if (entry - fb["xy"]) @ (sgn * np.array([np.cos(fb["yaw"]), np.sin(fb["yaw"])])) < 0:
                            break
                # 1) Face the approach line, correcting the lateral error.
                lon, lat, _ = self._pose_error(fb)
                fb = self._turn_to(self.approach_yaw - np.arctan(lat / max(-lon, 0.1)),
                                   np.radians(2.0), budget)
                # 2) Drive along the line to the pose, steering on the way.
                hist = []
                while budget[0] > 0:
                    lon, lat, _ = self._pose_error(fb)
                    if lon > DOCK_LON_TARGET - 0.003:
                        break
                    hist.append(lon)
                    if len(hist) > 20 and hist[-1] - hist[-21] < 0.002 and lon > -0.02:
                        break                 # stalled against the tower
                    eh = _wrap(self.approach_yaw - np.arctan(3.0 * lat) - fb["yaw"])
                    w = float(np.clip(6.0 * eh, -1.5, 1.5))
                    v = APPROACH_V if lon < DOCK_LON_TARGET - 0.08 else 0.5 * APPROACH_V
                    fb = self._base_cmd(v, w, FLIPPER_STOW_RAD)
                    budget[0] -= 1
                for _ in range(10):
                    fb = self._base_cmd(0.0, 0.0, FLIPPER_STOW_RAD)
                # 3) Final heading trim.
                fb = self._turn_to(self.approach_yaw, DOCK_TOL_YAW * 0.7, budget)
                lon, lat, eyaw = self._pose_error(fb)
                if W.is_fallen(fb):
                    return False, {"dock_reason": "fallen"}
                if (abs(lon - DOCK_LON_TARGET) < DOCK_TOL_LONG and abs(lat) < DOCK_TOL_LAT
                        and abs(eyaw) < DOCK_TOL_YAW):
                    ok = True
                    break
                self._log(f"dock attempt {attempt + 1}: err=({lon:+.3f},{lat:+.3f},"
                          f"{np.degrees(eyaw):+.1f}deg) -> retry")
                if budget[0] <= 0:
                    break
        for _ in range(SETTLE_STEPS):
            fb = self._base_cmd(0.0, 0.0, FLIPPER_STOW_RAD)
        lon, lat, eyaw = self._pose_error(fb)
        return ok, {"dock_err_long": lon, "dock_err_lat": lat,
                    "dock_err_yaw_deg": float(np.degrees(eyaw)),
                    "dock_steps": (DOCK_MAX_STEPS - budget[0]) if self.dock_mode == "scripted" else 0}

    def _retract(self) -> None:
        eid = self.arm._eq_freeze
        if eid >= 0:
            self.data.eq_active[eid] = 0
        for name, act in zip(C.ARM_JOINTS, self.base._a_arm):
            self.data.ctrl[act] = C.ARM_REST_POSE[name]
        self.base.ctrl.sync_from_data()
        for _ in range(RETRACT_STEPS):
            self._base_cmd(0.0, 0.0, FLIPPER_STOW_RAD)
        start = self.base._feedback()["xy"].copy()
        for _ in range(int(self.backoff_dist / BACKOFF_V / 0.05) + 40):
            fb = self._base_cmd(-BACKOFF_V, 0.0, C.FLIPPER_HOME_RAD)
            if np.linalg.norm(fb["xy"] - start) >= self.backoff_dist:
                break
        for _ in range(10):
            self._base_cmd(0.0, 0.0, C.FLIPPER_HOME_RAD)

    # ── API ────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[str, np.ndarray]:
        if self.randomize_tower:
            r = self.tower_xy_range
            self.set_tower_pose(self.rng.uniform(-r, r, 2), float(self.rng.uniform(-np.pi, np.pi)))
        for _ in range(50):
            self.base.reset()
            fb = self.base._feedback()
            if (np.linalg.norm(fb["xy"] - self.tower_xy) >= SPAWN_MIN_DIST
                    and np.linalg.norm(fb["xy"] - self.predock_xy) >= 1.0):
                break
        else:
            raise RuntimeError("could not sample a spawn far from the tower")
        # No procedural virtual obstacle: the tower is this episode's obstacle.
        gid = self.base._obstacle_gid
        if gid >= 0:
            self.model.geom_pos[gid, 2] = -5.0
        self._park_loose_buttons()
        self.final_goal = self._sample_final_goal()
        self.steps = 0
        self.phase = "NAVIGATE_TO_TOWER"
        self.agent = "base"
        self.stats = {"spawn_xy": fb["xy"].copy(), "final_goal": self.final_goal.copy(),
                      "tower_xy": self.tower_xy.copy(), "approach_yaw": self.approach_yaw,
                      "min_dist_predock": np.inf,
                      # Mission milestones (per-stage rates and training bonuses).
                      "reached_tower": False, "docked": False, "pressed": False}
        obs = self._retarget_base(self.predock_xy)
        self._log(f"reset spawn={fb['xy'].round(2)} predock={self.predock_xy.round(2)} "
                  f"final_goal={self.final_goal.round(2)}")
        return self.agent, obs

    def _finish(self, success: bool, fail_stage: Optional[str], info: dict) -> dict:
        self.phase = "DONE"
        info.update(success=success, fail_stage=fail_stage, phase="DONE", **self.stats)
        info["final_xy"] = self.base._feedback()["xy"].copy()
        self._log(f"DONE success={success} fail_stage={fail_stage}")
        return info

    def step(self, action: np.ndarray):
        if self.phase == "DONE":
            raise RuntimeError("episode finished: call reset()")
        acting = self.agent
        self.steps += 1
        info = {"agent": acting, "segment_end": False, "transition": None}

        if acting == "base":
            obs, reward, done_b, binfo = self.base.step(action)
            info["base_reason"] = binfo.get("reason")
            if self.phase == "NAVIGATE_TO_TOWER":
                d = float(np.linalg.norm(self.base._feedback()["xy"] - self.predock_xy))
                self.stats["min_dist_predock"] = min(self.stats["min_dist_predock"], d)
            if binfo.get("reached"):
                info["segment_end"] = True
                if self.phase == "NAVIGATE_TO_TOWER":
                    self._log(f"reached pre-dock after {self.steps} steps -> DOCK")
                    self.stats["reached_tower"] = True
                    ok, dinfo = self._dock()
                    self.stats.update(dinfo)
                    self.stats["docked"] = bool(ok)
                    if not ok:
                        return self.agent, obs, reward, True, self._finish(False, "dock", info)
                    self.phase, self.agent = "PRESS_BUTTON", "arm"
                    info["transition"] = "NAVIGATE_TO_TOWER->DOCK->PRESS_BUTTON"
                    self._log(f"dock {dinfo}")
                    return self.agent, self.arm.reset_from_current_state(), reward, False, info
                return self.agent, obs, reward, True, self._finish(True, None, info)
            if done_b:
                info["segment_end"] = True
                stage = "base_to_tower" if self.phase == "NAVIGATE_TO_TOWER" else "base_to_goal"
                return self.agent, obs, reward, True, self._finish(False, stage, info)
        else:
            obs, reward, done_a, ainfo = self.arm.step(action)
            info["arm_reason"] = ainfo.get("reason")
            if done_a:
                info["segment_end"] = True
                info["arm_success"] = bool(ainfo["success"])
                self.stats.update(arm_success=bool(ainfo["success"]),
                                  arm_reason=ainfo.get("reason"),
                                  arm_frac_max=float(ainfo.get("frac_max", 0.0)))
                if not ainfo["success"]:
                    return self.agent, obs, reward, True, self._finish(False, "arm", info)
                self.stats["pressed"] = True
                self._retract()
                self.phase, self.agent = "NAVIGATE_TO_END", "base"
                info["transition"] = "PRESS_BUTTON->RETRACT->NAVIGATE_TO_END"
                self._log(f"button pressed ({ainfo.get('reason')}) -> NAVIGATE_TO_END")
                try:
                    obs = self._retarget_base(self.final_goal)
                except RuntimeError:
                    return self.agent, obs, reward, True, self._finish(False, "plan_to_goal", info)
                return self.agent, obs, reward, False, info

        if self.steps >= self.max_steps:
            info["segment_end"] = True
            info["timeout_phase"] = self.phase
            return self.agent, obs, reward, True, self._finish(False, "timeout", info)
        info["phase"] = self.phase
        return self.agent, obs, reward, False, info

    def close(self) -> None:
        self.arm.close()
