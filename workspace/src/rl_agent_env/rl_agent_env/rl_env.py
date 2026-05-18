"""Synchronous, Gym-like RL environment backed by an asynchronous ROS 2 bridge.

The :class:`Env` exposes the *same* surface the PPO trainer in
``train_ppo.py`` already uses (``reset``, ``step(action)``, ``close``) but
under the hood it:

    1. Boots rclpy in the main process.
    2. Instantiates :class:`RosCommunicationNode`.
    3. Spins that node in a *daemon* background thread, so all subscriber
       callbacks fire continuously while the trainer is doing PyTorch work.
    4. On every ``step`` it publishes the action, then performs a passive wait
       (``time.sleep`` in a loop) on the simulated ``/clock`` until exactly
       ``TIMESTEP`` of sim-time has elapsed -- replacing the blocking
       ``mj_step`` call of the original MuJoCo-only env.
    5. After the wait, it snapshots the lock-protected buffers from the
       bridge and packs them into a flat numpy observation.

Action conventions
------------------
PPO outputs tanh-squashed values in ``[-1, 1]``. The env interprets a flat
8-vector as a normalized action and applies the per-DOF physical limits below
before publishing. A dict input ``{"arm": [...6], "base": [...2]}`` is taken
to be in **raw physical units** (m/s, rad/s) and is published verbatim
(useful for teleop / scripted tests).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

import rclpy

from rl_agent_env.ros_bridge import RosCommunicationNode


# --------------------------------------------------------------------------- #
#                                Defaults                                     #
# --------------------------------------------------------------------------- #
TIMESTEP            = 0.05   # seconds of sim-time advanced per env.step()
WAIT_POLL_S         = 1e-3   # sleep granularity while waiting for /clock
CLOCK_WAIT_TIMEOUT  = 5.0    # max wall-clock seconds for /clock to start
EPISODE_DURATION_S  = 8.0    # mirror the original MuJoCo env
SCAN_RAYS           = 16

# ---- Physical limits (ported from the old ActionAdapter) ------------------ #
MAX_LINEAR_VEL      = 0.5    # m/s   - diff drive
MAX_ANGULAR_VEL     = 1.0    # rad/s - diff drive yaw rate
MAX_ARM_LINEAR_VEL  = 0.2    # m/s   - MoveIt Servo Cartesian linear
MAX_ARM_ANGULAR_VEL = 0.5    # rad/s - MoveIt Servo Cartesian angular

# ---- Reward / termination constants (ported from aesir_env.py) ----------- #
DOOR_X              = 3.0    # door position along +X (world frame, m)
CROSSED_X           = 4.0    # past this x the robot has "crossed" the door
FALLEN_Z            = 0.05   # base z below this -> robot considered fallen
PROGRESS_GAIN       = 5.0
POST_CROSS_VX_GAIN  = 2.0
YAW_PENALTY_GAIN    = 0.02
ENERGY_PENALTY_GAIN = 1e-5
FALL_PENALTY        = -100.0
SUCCESS_BONUS       = 200.0


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rotation around Z) from a (x,y,z,w) quaternion -- ROS order."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _tilt_body_z(qx: float, qy: float, qz: float, qw: float) -> float:
    """Z component of the robot-body +Z axis expressed in the world frame."""
    return 1.0 - 2.0 * (qx * qx + qy * qy)


class Env:
    """Sync Gym-like wrapper over an async ROS 2 stack.

    The shape of the observation vector is *fixed at construction time* so
    PPO can preallocate its replay buffer. Missing topics are filled with
    zeros (NaN-replaced) to keep the contract stable.
    """

    def __init__(
        self,
        node_name: str = "rl_communication_bridge",
        scan_rays: int = SCAN_RAYS,
        timestep: float = TIMESTEP,
        episode_duration: float = EPISODE_DURATION_S,
        enable_scan: bool = True,
    ) -> None:
        self.timestep         = float(timestep)
        self.scan_rays        = int(scan_rays)
        self.episode_duration = float(episode_duration)
        self.enable_scan      = bool(enable_scan)

        # ---- ROS init ----------------------------------------------------- #
        if not rclpy.ok():
            rclpy.init()
        self._owns_rclpy = True

        self.ros_node = RosCommunicationNode(
            node_name=node_name,
            enable_scan=self.enable_scan,
        )

        # Spin the executor in a background daemon thread.
        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_node,),
            daemon=True,
            name=f"{node_name}_spin",
        )
        self._spin_thread.start()

        # Block until /clock starts ticking.
        self._wait_for_clock(CLOCK_WAIT_TIMEOUT)

        # ---- shape of obs / act ----------------------------------------- #
        # joint pos (6) + joint vel (6) + odom pose (7) + odom twist (6) +
        # scan (scan_rays if enabled else 0)
        self._n_joints = len(self.ros_node.DEFAULT_ARM_JOINTS)
        self.obs_len = (
            2 * self._n_joints
            + 7 + 6
            + (self.scan_rays if self.enable_scan else 0)
        )
        self.act_len = 8  # 6 arm twist + 2 base twist

        # ---- episode bookkeeping ---------------------------------------- #
        self._episode_start_sim_t: Optional[float] = None
        self._prev_door_dist: Optional[float] = None
        self._max_x: float = 0.0
        self._step_count: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------ #
    #                       Internal helpers                             #
    # ------------------------------------------------------------------ #
    def _wait_for_clock(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ros_node.get_sim_time() > 0.0:
                return
            time.sleep(WAIT_POLL_S * 10)
        self.ros_node.get_logger().warning(
            "Did not receive /clock within timeout; continuing with sim_time=0."
        )

    def _wait_sim_dt(self, dt: float) -> None:
        """Passive wait: spin on time.sleep until sim-time has advanced by dt."""
        if dt <= 0.0:
            return
        start  = self.ros_node.get_sim_time()
        target = start + dt
        while self.ros_node.get_sim_time() < target:
            time.sleep(WAIT_POLL_S)

    def _build_observation(self) -> np.ndarray:
        joint = self.ros_node.get_joint_state()
        odom  = self.ros_node.get_odom()
        scan  = self.ros_node.get_scan(self.scan_rays) if self.enable_scan else None

        parts = []
        if joint is None:
            parts.append(np.zeros(2 * self._n_joints, dtype=np.float64))
        else:
            parts.append(np.nan_to_num(joint["position"], nan=0.0))
            parts.append(np.nan_to_num(joint["velocity"], nan=0.0))

        if odom is None:
            parts.append(np.zeros(13, dtype=np.float64))
        else:
            parts.append(odom["position"])
            parts.append(odom["orientation"])
            parts.append(odom["linear_vel"])
            parts.append(odom["angular_vel"])

        if self.enable_scan:
            parts.append(scan if scan is not None
                         else np.zeros(self.scan_rays, dtype=np.float64))

        state = np.concatenate(parts).astype(np.float64)
        assert state.size == self.obs_len, (
            f"obs size mismatch: got {state.size}, expected {self.obs_len}"
        )
        return state

    def _split_and_scale_action(self, action) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(arm_twist_6, base_twist_2)`` in raw physical units.

        Two input modes:
          * dict ``{"arm": [...6], "base": [...2]}`` -> raw m/s, rad/s
            (no scaling applied; assumed user-curated).
          * flat 8-vector -> normalized in ``[-1, 1]`` (e.g. PPO output),
            multiplied by per-DOF physical limits.
        """
        if isinstance(action, dict):
            arm  = np.asarray(action["arm"],  dtype=np.float64).reshape(-1)
            base = np.asarray(action["base"], dtype=np.float64).reshape(-1)
            if arm.size != 6 or base.size != 2:
                raise ValueError(
                    f"dict action shape: arm={arm.size}, base={base.size}"
                )
            return arm, base

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.act_len:
            raise ValueError(
                f"flat action must have {self.act_len} components; got {a.size}"
            )
        a = np.clip(a, -1.0, 1.0)
        arm_scale = np.array([
            MAX_ARM_LINEAR_VEL,  MAX_ARM_LINEAR_VEL,  MAX_ARM_LINEAR_VEL,
            MAX_ARM_ANGULAR_VEL, MAX_ARM_ANGULAR_VEL, MAX_ARM_ANGULAR_VEL,
        ], dtype=np.float64)
        base_scale = np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL], dtype=np.float64)
        return a[:6] * arm_scale, a[6:] * base_scale

    # ------------------------------------------------------------------ #
    #                        Reward + termination                        #
    # ------------------------------------------------------------------ #
    def _compute_reward_and_done(self, odom: Optional[dict]) -> Tuple[float, bool]:
        """Reward shaping ported from the MuJoCo env, adapted to /odom.

        Things missing vs. the MuJoCo version:
          * Door hinge angle (no dedicated topic yet) -> TODO
          * Per-substep accumulation (here it's per env.step instead).
        """
        if odom is None:
            return 0.0, False

        x  = float(odom["position"][0])
        z  = float(odom["position"][2])
        vx = float(odom["linear_vel"][0])
        qx, qy, qz, qw = (float(v) for v in odom["orientation"])
        yaw = _yaw_from_quat(qx, qy, qz, qw)

        reward    = 0.0
        cur_dd    = abs(DOOR_X - x)

        # Progress toward the door (before crossing).
        if self._prev_door_dist is None:
            self._prev_door_dist = cur_dd
        if x < DOOR_X:
            reward += PROGRESS_GAIN * (self._prev_door_dist - cur_dd)
        else:
            reward += POST_CROSS_VX_GAIN * max(vx, 0.0)
        self._prev_door_dist = cur_dd

        if x > self._max_x:
            self._max_x = x

        # Heading: face +X.
        reward -= YAW_PENALTY_GAIN * (1.0 - np.cos(yaw))
        # Light shake penalty.
        twist_sq = float(np.sum(np.square(odom["linear_vel"]))
                         + np.sum(np.square(odom["angular_vel"])))
        reward -= ENERGY_PENALTY_GAIN * twist_sq

        # TODO: add door hinge bonus once a door_state topic is published.

        # ---- terminal conditions -------------------------------------- #
        done = False
        body_z_world_z = _tilt_body_z(qx, qy, qz, qw)
        tilt_ok = body_z_world_z > 0.3

        sim_elapsed = (
            self.ros_node.get_sim_time() - (self._episode_start_sim_t or 0.0)
        )
        if sim_elapsed >= self.episode_duration:
            done = True
        if z < FALLEN_Z or not tilt_ok:
            done    = True
            reward += FALL_PENALTY
        if x > CROSSED_X:
            done    = True
            reward += SUCCESS_BONUS

        return reward, done

    # ------------------------------------------------------------------ #
    #                              Gym API                               #
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Reset episode bookkeeping and return the current observation.

        NOTE: does *not* reset the simulator state -- that requires a service
        call to mujoco_ros2_control (e.g. /reset_simulation). Wire it up here
        once exposed.
        """
        self.ros_node.stop_robot()
        self._episode_start_sim_t = self.ros_node.get_sim_time()
        self._prev_door_dist      = None
        self._max_x               = 0.0
        self._step_count          = 0
        self.done                 = False
        # TODO: call /mujoco_ros2_control/reset_simulation service here.
        return self._build_observation()

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        """Publish the action, wait one ``timestep`` of sim-time, return (s, r, d)."""
        arm, base = self._split_and_scale_action(action)

        self.ros_node.publish_arm_twist(arm)
        self.ros_node.publish_base_cmd(v_lin=base[0], w_ang=base[1])

        self._wait_sim_dt(self.timestep)

        state            = self._build_observation()
        odom             = self.ros_node.get_odom()
        reward, self.done = self._compute_reward_and_done(odom)

        self._step_count += 1
        return state, float(reward), bool(self.done)

    def close(self, *_args, **_kwargs) -> None:
        """Stop the robot, tear down rclpy and join the spin thread.

        Returns ``None`` (the old MuJoCo env returned a video path; the ROS
        env doesn't record video itself -- record from an image topic in a
        separate node if you want one).
        """
        try:
            self.ros_node.stop_robot()
        except Exception:
            pass

        try:
            self.ros_node.destroy_node()
        except Exception:
            pass

        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        return None


# --------------------------------------------------------------------------- #
#                              Standalone entry                               #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Smoke test: bring up the env, send zero actions for a few steps."""
    env = Env()
    try:
        env.reset()
        for i in range(20):
            obs, reward, done = env.step({"arm": [0.0] * 6, "base": [0.0, 0.0]})
            print(f"step {i:3d}  obs_dim={obs.size}  reward={reward:.3f}  done={done}")
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
