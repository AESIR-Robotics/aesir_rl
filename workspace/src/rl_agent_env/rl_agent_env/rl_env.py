"""Ambiente RL síncrono (Gym-like) sobre el backend ROS 2 asíncrono.

Observación (32 valores con scan habilitado):
  joint_pos(6) | joint_vel(6) | pos(3) | quat(4) | lin_vel(3) | ang_vel(3) | scan(7)

Acción (8 valores, normalizados en [-1, 1] por el trainer PPO):
  Δjoint_1..6 [rad] | v_lin [m/s] | ω_ang [rad/s]
  → escalados a velocidades físicas antes de publicar.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy

from rl_agent_env.ros_bridge import RosCommunicationNode


# --------------------------------------------------------------------------- #
#                              Constantes                                     #
# --------------------------------------------------------------------------- #
TIMESTEP            = 0.05    # segundos de tiempo de simulación por env.step()
WAIT_POLL_S         = 1e-3
CLOCK_WAIT_TIMEOUT  = 5.0
EPISODE_DURATION_S  = 8.0
SCAN_RAYS           = 7       # coincide con los 7 rangefinders de aesir_robot.xml

# Límites físicos
MAX_LINEAR_VEL  = 0.5    # m/s
MAX_ANGULAR_VEL = 1.0    # rad/s
MAX_JOINT_DELTA = 0.10   # rad por step (PPO output en [-1,1] × este valor)

# Reward
DOOR_X              = 3.0
CROSSED_X           = 4.0
FALLEN_Z            = 0.05
PROGRESS_GAIN       = 5.0
POST_CROSS_VX_GAIN  = 2.0
YAW_PENALTY_GAIN    = 0.02
ENERGY_PENALTY_GAIN = 1e-5
FALL_PENALTY        = -100.0
SUCCESS_BONUS       = 200.0


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _tilt_body_z(qx: float, qy: float, qz: float, qw: float) -> float:
    return 1.0 - 2.0 * (qx * qx + qy * qy)


class Env:
    """Wrapper Gym síncrono sobre el stack ROS 2 asíncrono.

    El tamaño del vector de observación se fija en construcción para que PPO
    pueda preasignar su replay buffer. Topics faltantes se rellenan con ceros.
    """

    def __init__(
        self,
        node_name: str       = "rl_communication_bridge",
        scan_rays: int       = SCAN_RAYS,
        timestep: float      = TIMESTEP,
        episode_duration: float = EPISODE_DURATION_S,
        enable_scan: bool    = True,
    ) -> None:
        self.timestep         = float(timestep)
        self.scan_rays        = int(scan_rays)
        self.episode_duration = float(episode_duration)
        self.enable_scan      = bool(enable_scan)

        if not rclpy.ok():
            rclpy.init()
        self._owns_rclpy = True

        self.ros_node = RosCommunicationNode(
            node_name=node_name,
            enable_scan=self.enable_scan,
        )

        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_node,),
            daemon=True,
            name=f"{node_name}_spin",
        )
        self._spin_thread.start()

        self._wait_for_clock(CLOCK_WAIT_TIMEOUT)

        # Dimensiones fijas del espacio de obs/acción
        self._n_joints = len(self.ros_node.DEFAULT_ARM_JOINTS)
        # joint_pos(6) + joint_vel(6) + pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + scan(7|0)
        self.obs_len = (
            2 * self._n_joints          # 12
            + 7 + 6                     # 13 (pose + twist)
            + (self.scan_rays if self.enable_scan else 0)
        )
        self.act_len = 8  # 6 deltas de joints + 2 base (v_lin, ω_ang)

        # Estado del episodio
        self._episode_start_sim_t: Optional[float] = None
        self._prev_door_dist: Optional[float]      = None
        self._max_x: float                         = 0.0
        self._step_count: int                      = 0
        self.done: bool                            = False

    # ------------------------------------------------------------------ #
    #                          Helpers internos                          #
    # ------------------------------------------------------------------ #
    def _wait_for_clock(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ros_node.get_sim_time() > 0.0:
                return
            time.sleep(WAIT_POLL_S * 10)
        self.ros_node.get_logger().warning(
            "No se recibió /clock en el timeout; continuando con sim_time=0."
        )

    def _wait_sim_dt(self, dt: float) -> None:
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
            parts.append(
                scan if scan is not None
                else np.full(self.scan_rays, 15.0, dtype=np.float64)
            )

        state = np.concatenate(parts).astype(np.float64)
        assert state.size == self.obs_len, (
            f"obs size mismatch: got {state.size}, expected {self.obs_len}"
        )
        return state

    def _split_and_scale_action(self, action) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (arm_vel_6, base_cmd_2) en unidades físicas.

        Modos de entrada:
          • dict {"arm": [...6], "base": [...2]} → sin escalado (unidades crudas).
          • vector plano 8-D normalizado en [-1,1] → escala a vel/rad.
        """
        if isinstance(action, dict):
            arm  = np.asarray(action["arm"],  dtype=np.float64).reshape(-1)
            base = np.asarray(action["base"], dtype=np.float64).reshape(-1)
            if arm.size != 6 or base.size != 2:
                raise ValueError(f"dict action shape: arm={arm.size}, base={base.size}")
            return arm, base

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.act_len:
            raise ValueError(
                f"flat action debe tener {self.act_len} componentes; got {a.size}"
            )
        a = np.clip(a, -1.0, 1.0)
        # Convierte delta de posición [rad] a velocidad [rad/s] para el controlador
        arm_vel  = a[:6] * MAX_JOINT_DELTA / self.timestep
        base_cmd = a[6:] * np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL], dtype=np.float64)
        return arm_vel, base_cmd

    # ------------------------------------------------------------------ #
    #                       Reward + terminación                         #
    # ------------------------------------------------------------------ #
    def _compute_reward_and_done(self, odom: Optional[dict]) -> Tuple[float, bool]:
        if odom is None:
            return 0.0, False

        x  = float(odom["position"][0])
        z  = float(odom["position"][2])
        vx = float(odom["linear_vel"][0])
        qx, qy, qz, qw = (float(v) for v in odom["orientation"])
        yaw = _yaw_from_quat(qx, qy, qz, qw)

        reward = 0.0
        cur_dd = abs(DOOR_X - x)

        if self._prev_door_dist is None:
            self._prev_door_dist = cur_dd
        if x < DOOR_X:
            reward += PROGRESS_GAIN * (self._prev_door_dist - cur_dd)
        else:
            reward += POST_CROSS_VX_GAIN * max(vx, 0.0)
        self._prev_door_dist = cur_dd

        if x > self._max_x:
            self._max_x = x

        reward -= YAW_PENALTY_GAIN * (1.0 - np.cos(yaw))
        twist_sq = float(
            np.sum(np.square(odom["linear_vel"])) +
            np.sum(np.square(odom["angular_vel"]))
        )
        reward -= ENERGY_PENALTY_GAIN * twist_sq

        done = False
        tilt_ok = _tilt_body_z(qx, qy, qz, qw) > 0.3

        sim_elapsed = (
            self.ros_node.get_sim_time() - (self._episode_start_sim_t or 0.0)
        )
        if sim_elapsed >= self.episode_duration:
            done = True
        if z < FALLEN_Z or not tilt_ok:
            done   = True
            reward += FALL_PENALTY
        if x > CROSSED_X:
            done   = True
            reward += SUCCESS_BONUS

        return reward, done

    # ------------------------------------------------------------------ #
    #                             API Gym                                #
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Reinicia el bookkeeping del episodio y devuelve la observación actual.

        NOTE: no reinicia el estado del simulador. Implementar llamada al
        servicio /mujoco_ros2_control/reset_simulation cuando esté disponible.
        """
        self.ros_node.stop_robot()
        self._episode_start_sim_t = self.ros_node.get_sim_time()
        self._prev_door_dist      = None
        self._max_x               = 0.0
        self._step_count          = 0
        self.done                 = False
        return self._build_observation()

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        """Publica la acción, espera un timestep de sim-time, devuelve (s, r, d)."""
        arm_vel, base_cmd = self._split_and_scale_action(action)

        self.ros_node.send_joint_velocity(arm_vel)
        self.ros_node.publish_base_cmd(v_lin=base_cmd[0], w_ang=base_cmd[1])

        self._wait_sim_dt(self.timestep)

        state              = self._build_observation()
        odom               = self.ros_node.get_odom()
        reward, self.done  = self._compute_reward_and_done(odom)

        self._step_count += 1
        return state, float(reward), bool(self.done)

    def close(self, *_args, **_kwargs) -> None:
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


# --------------------------------------------------------------------------- #
#                              Smoke test                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    env = Env()
    try:
        obs = env.reset()
        print(f"obs_len={obs.size}  act_len={env.act_len}")
        for i in range(20):
            obs, reward, done = env.step({"arm": [0.0] * 6, "base": [0.0, 0.0]})
            print(f"step {i:3d}  reward={reward:.3f}  done={done}")
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
