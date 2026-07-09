#!/usr/bin/env python3
"""
base_env.py — Env de la base (oruga + flippers), Gym-like, sobre el bridge ROS2.

Responsabilidad de este modulo: TODO lo que define "el mundo" que ve la
politica — guia de navegacion, observacion, reward, terminacion, reset — igual
que en la version directa-MuJoCo original (BaseMuJoCoEnv). La diferencia es
que aqui la fisica no la posee este proceso: vive en mujoco_ros_bridge.py y se
habla con ella por ROS2.

    BaseRosEnv.reset()        -> obs
    BaseRosEnv.step(action)   -> obs, reward, done, info

Por dentro, en cada step():

    accion [v, ω, flipper×4]
        ── Twist ──────────▶ hardware_node/cmd_vel        (bridge)
        ── JointControl ───▶ /commands_hardware            (bridge)
    (se deja avanzar la fisica real del bridge ~1/control_hz segundos)
        pose/twist/joints  ◀── hardware_node/pose, state_vel, joint_states
    guia = global_navigator.step(xy, yaw)   -- A* + vortex APF
    obs    = guia + feedback cinematico
    reward = seguir velocidad/direccion objetivo + waypoint − castigos

train_base.py es el script general (pipeline): importa BaseRosEnv, instancia
la politica PPO y corre el bucle de entrenamiento — no conoce ROS ni reward.

REQUIERE, en OTRA terminal, el bridge corriendo:
    cd rl_ws && MUJOCO_GL=glfw python3 mujoco_ros_bridge.py
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from hardware.msg import JointControl
from std_srvs.srv import Trigger

from global_navigator import plan_route, GlobalNavigator, quat_to_yaw

_HERE    = os.path.dirname(os.path.abspath(__file__))
NAV_JSON = os.path.join(_HERE, "obstacles.json")

# ── Escalas de comando ───────────────────────
V_MAX_MPS   = 0.6      # linear.x  a v_norm = 1
W_MAX_RADPS = 1.5      # angular.z a w_norm = 1
FLIPPER_MAX = 3.1416   # rad a flip = 1
CONTROL_HZ  = 20.0     # frecuencia del lazo de control RL

# ── Mision (frame de aesir_complete.xml == frame de obstacles.json) ──────────
# Spawn = SPAWN_POSE del bridge; meta = ultimo pallet de obstacles.json.
START_XY: Tuple[float, float] = (-2.2, 4.2)
GOAL_XY:  Optional[Tuple[float, float]] = None   # None -> ultimo pallet del JSON
FINISH_DIST = 0.60
EPISODE_MAX_STEPS = 2500

FLIPPER_JOINTS = ["flipper_1_joint", "flipper_2_joint", "flipper_3_joint", "flipper_4_joint"]

# ── Pesos de reward ────────────────────────────────────────────────────────
# objetivo que da la guia del vortex.
W_DIRECTION    = 1.0     # encarar al objetivo (cos Δθ)
W_VELOCITY     = 1.5     # igualar la velocidad forward objetivo

WP_BONUS       = 50.0    # bonus al cruzar un waypoint       
TIME_PENALTY   = 0.02
FALL_PENALTY   = 100.0
STUCK_MAX      = 2.0
ENERGY_W       = 1e-4    # costo de energia (antes: 1e-9 * ctrl^2, ahora accion^2)
FLIPPER_JERK_W = 0.5
TILT_W         = 5.0

# ── Tamaños expuestos a la politica ──────────────────────────────────────────
OBS_DIM = 15   # guia(3) + twist_base(3) + flipper_qpos(4) + flipper_qvel(4) + upright(1)
ACT_DIM = 6    # v, ω, flipper×4


# ── Convencion "hardware" (espejo de topic_bridge_hardware.cpp) ──────────────
def hw_to_ros(rad: float) -> float:
    return math.fmod(rad, 2.0 * math.pi) - math.pi

def ros_to_hw(rad: float) -> float:
    return rad + math.pi

def quat_upright(quat_wxyz) -> float:
    """Componente z del eje z del chasis (R[2,2]): 1 = vertical, 0 = tumbado."""
    w, x, y, z = quat_wxyz
    return 1.0 - 2.0 * (x * x + y * y)


# ──────────────────────────── Nodo ROS2: E/S con el bridge ─────────────────
class _BridgeInterface(Node):
    """Publica comandos al bridge y cachea el ultimo feedback recibido."""

    def __init__(self):
        super().__init__("base_env_bridge_interface")
        self._lock = threading.Lock()

        self._pose_xy   = None                      # np.array([x, y])
        self._pose_z    = 0.0
        self._yaw       = 0.0
        self._upright   = 1.0
        self._twist     = np.zeros(3, dtype=np.float32)  # [v_fwd, v_lat, omega_z]
        self._flip_qpos = np.zeros(4, dtype=np.float32)
        self._flip_qvel = np.zeros(4, dtype=np.float32)

        self.cmd_vel_pub = self.create_publisher(Twist, "hardware_node/cmd_vel", 10)
        self.joint_pub   = self.create_publisher(JointControl, "/commands_hardware", 10)
        self.create_subscription(PoseStamped, "hardware_node/pose", self._pose_cb, 10)
        self.create_subscription(Twist, "hardware_node/state_vel", self._vel_cb, 10)
        self.create_subscription(JointState, "/hardware_node/joint_states", self._js_cb, 10)

        self.reset_cli = self.create_client(Trigger, "/mujoco_ros_bridge/reset_sim")

    # ── Callbacks de feedback ────────────────────────────────────────────────
    def _pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        quat = np.array([q.w, q.x, q.y, q.z])
        with self._lock:
            self._pose_xy = np.array([msg.pose.position.x, msg.pose.position.y])
            self._pose_z  = float(msg.pose.position.z)
            self._yaw     = quat_to_yaw(quat)
            self._upright = quat_upright(quat)

    def _vel_cb(self, msg: Twist):
        with self._lock:
            self._twist = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def _js_cb(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        qpos = np.zeros(4, dtype=np.float32)
        qvel = np.zeros(4, dtype=np.float32)
        for k, name in enumerate(FLIPPER_JOINTS):
            i = idx.get(name)
            if i is None:
                continue
            qpos[k] = hw_to_ros(msg.position[i]) if i < len(msg.position) else 0.0
            qvel[k] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        with self._lock:
            self._flip_qpos, self._flip_qvel = qpos, qvel

    # ── Snapshot de feedback ─────────────────────────────────────────────────
    def feedback(self) -> Optional[dict]:
        with self._lock:
            if self._pose_xy is None:
                return None
            return dict(
                xy=self._pose_xy.copy(), z=self._pose_z, yaw=self._yaw,
                upright=self._upright, twist=self._twist.copy(),
                flip_qpos=self._flip_qpos.copy(), flip_qvel=self._flip_qvel.copy(),
            )

    # ── Publicar accion ──────────────────────────────────────────────────────
    def publish_action(self, v_norm: float, w_norm: float, flippers: np.ndarray):
        tw = Twist()
        tw.linear.x  = float(np.clip(v_norm, -1.0, 1.0)) * V_MAX_MPS
        tw.angular.z = float(np.clip(w_norm, -1.0, 1.0)) * W_MAX_RADPS
        self.cmd_vel_pub.publish(tw)

        jc = JointControl()
        jc.header.stamp = self.get_clock().now().to_msg()
        jc.joint_names  = list(FLIPPER_JOINTS)
        jc.position     = [ros_to_hw(float(np.clip(f, -1.0, 1.0)) * FLIPPER_MAX) for f in flippers]
        self.joint_pub.publish(jc)

    def stop_robot(self):
        self.publish_action(0.0, 0.0, np.zeros(4))

    # ── Reset de episodio (servicio del bridge) ─────────────────────────────
    def reset_sim(self, timeout: float = 5.0) -> bool:
        if not self.reset_cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn("Servicio /mujoco_ros_bridge/reset_sim no disponible")
            return False
        fut = self.reset_cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.01)
        return fut.done() and fut.result() is not None and fut.result().success


# ──────────────────────────── Observacion y reward ─────────────────────────
def _build_obs(guidance: dict, fb: dict) -> np.ndarray:
    return np.concatenate([
        guidance["obs"],                                    # 3
        fb["twist"],                                        # 3  [v_fwd, v_lat, omega]
        fb["flip_qpos"],                                    # 4
        fb["flip_qvel"],                                    # 4
        [fb["upright"]],                                    # 1
    ]).astype(np.float32)


class _RewardState:
    """Estado entre pasos para el calculo de reward (progreso, stuck, jerk)."""
    def __init__(self):
        self.last_xy = None
        self.last_dist_to_target = 0.0
        self.last_wp = 0
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.stuck = 0

    def reset(self, xy: np.ndarray, dist_to_target: float):
        self.last_xy = xy.copy()
        self.last_dist_to_target = dist_to_target
        self.last_wp = 0
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.stuck = 0


def _compute_reward(fb: dict, guidance: dict, action: np.ndarray, rs: _RewardState) -> float:
    """Igual estructura que BaseMuJoCoEnv._reward (version directa-MuJoCo):
    caida letal -> castigos conservados (stuck, energia, jerk de flippers,
    inclinacion) -> progreso hacia el objetivo actual (misma formula con
    boost exponencial de proximidad) -> bonus al cruzar el objetivo (retorno
    temprano, igual que el pallet_bonus original). El objetivo ahora es el
    waypoint de la ruta A* en vez del pallet, y se suma ademas la recompensa
    NUEVA de encarar + igualar la velocidad que pide la guia del vortex."""
    xy = fb["xy"]
    dist_norm, sin_t, cos_t = guidance["obs"]
    target_xy = np.asarray(guidance["target"], dtype=np.float64)
    v_fwd = float(fb["twist"][0])

    # 1. Caida letal
    if fb["z"] < 0.10:
        return -FALL_PENALTY

    # 2. Inactividad 
    move_dist = float(np.linalg.norm(xy - rs.last_xy))
    rs.last_xy = xy.copy()
    if move_dist < 0.005:
        rs.stuck += 1
        penalty_stuck = min(STUCK_MAX, 0.01 * rs.stuck)
    else:
        rs.stuck = 0
        penalty_stuck = 0.0

    # 3. Costo de energia (conservado; antes 1e-9 * ctrl_crudo^2, ahora sobre
    #    la accion normalizada — mismo espiritu de penalizacion casi nula)
    action_cost = ENERGY_W * float(np.square(action).mean())

    # 4. Movimiento erratico de flippers
    current_flipper = action[2:6].astype(np.float32)
    flipper_pen = FLIPPER_JERK_W * float(np.square(current_flipper - rs.last_flip).mean())
    rs.last_flip = current_flipper.copy()

    # 5. Inclinacion 
    tilt_pen = max(0.0, 0.65 - float(fb["upright"])) * TILT_W

    penalties = penalty_stuck + action_cost + flipper_pen + tilt_pen

    # 6. Waypoint cruzado — al cruzarlo
    #    se devuelve solo el bonus mas las penalizaciones (sin progreso ni
    #    direccion/velocidad ese paso).
    if guidance["wp"] > rs.last_wp:
        rs.last_wp = guidance["wp"]
        rs.last_dist_to_target = float(np.linalg.norm(xy - target_xy))
        return WP_BONUS - penalties

    # 7. Progreso hacia el waypoint actual (delta_dist * boost
    #    exponencial de proximidad, misma formula que la version pallet)
    dist_to_target = float(np.linalg.norm(xy - target_xy))
    delta_dist = rs.last_dist_to_target - dist_to_target
    proximity_multiplier = float(np.exp(-dist_to_target))
    progress_reward = delta_dist * (50.0 + 100.0 * proximity_multiplier)
    rs.last_dist_to_target = dist_to_target

    # 8. Direccion y velocidad objetivo (la guia del vortex)
    direction_reward = W_DIRECTION * float(cos_t)
    v_des = V_MAX_MPS * float(dist_norm) * max(0.0, float(cos_t))
    speed_match = 1.0 - min(1.0, abs(v_fwd - v_des) / V_MAX_MPS)
    velocity_reward = W_VELOCITY * speed_match

    return progress_reward + direction_reward + velocity_reward - penalties - TIME_PENALTY


def _terminated(fb: dict, goal_xy: np.ndarray, ep_steps: int, max_steps: int) -> Tuple[bool, bool]:
    """Igual estructura que BaseMuJoCoEnv._terminated. Devuelve (done, reached_goal)."""
    if ep_steps >= max_steps:
        print("[base_env] ⏰ Episodio terminado por limite de pasos")
        return True, False
    if fb["upright"] < 0.20:
        print("[base_env] 🛑 Episodio terminado por caida (base demasiado inclinado)")
        return True, False
    if fb["z"] < 0.10:
        print("[base_env] 🛑 Episodio terminado por caida (base demasiado bajo)")
        return True, False
    if float(np.linalg.norm(fb["xy"] - goal_xy)) < FINISH_DIST:
        print("[base_env] 🏆 ¡Meta alcanzada!")
        return True, True
    return False, False


# ──────────────────────────── Env Gym-like ──────────────────────────────────
class BaseRosEnv:
    """Env de entrenamiento de la base, sobre mujoco_ros_bridge.py via ROS2.

    Uso (igual forma que la BaseMuJoCoEnv original):
        env = BaseRosEnv()
        obs = env.reset()
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
    """

    def __init__(self,
                 nav_json: str = NAV_JSON,
                 start_xy: Tuple[float, float] = START_XY,
                 goal_xy: Optional[Tuple[float, float]] = GOAL_XY,
                 control_hz: float = CONTROL_HZ,
                 max_steps: int = EPISODE_MAX_STEPS):

        self.obs_dim = OBS_DIM
        self.act_len = ACT_DIM
        self.dt = 1.0 / control_hz
        self.max_steps = max_steps

        # ── Ruta global (A*) una sola vez — geometria estatica del mapa ──────
        self.start_xy = np.array(start_xy, dtype=np.float64)
        if goal_xy is None:
            goal_xy = tuple(json.load(open(nav_json))["pallets"][-1]["center_xy"])
        self.goal_xy = np.array(goal_xy, dtype=np.float64)
        self.waypoints = plan_route(nav_json, tuple(self.start_xy), tuple(self.goal_xy))
        self.nav = GlobalNavigator(nav_json, waypoints=self.waypoints)
        print(f"[base_env] Ruta: {tuple(self.start_xy)} -> {tuple(self.goal_xy)}  "
              f"({len(self.waypoints)} waypoints)")

        # ── ROS2 ───────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self._node = _BridgeInterface()
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        print("[base_env] Esperando feedback del bridge en hardware_node/pose ...")
        while self._node.feedback() is None and rclpy.ok():
            time.sleep(0.05)
        print("[base_env] Bridge conectado.")

        self._rs = _RewardState()
        self._ep_steps = 0
        self._fb: Optional[dict] = None

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self._node.stop_robot()
        self._node.reset_sim()
        time.sleep(0.1)                       # deja llegar el primer feedback nuevo
        fb = self._node.feedback()
        while fb is None and rclpy.ok():
            time.sleep(0.02)
            fb = self._node.feedback()

        self.nav.reset(fb["xy"])
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        self._rs.reset(fb["xy"], float(np.linalg.norm(fb["xy"] - np.asarray(guidance["target"]))))
        self._ep_steps = 0
        self._fb = fb
        return _build_obs(guidance, fb)

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self._node.publish_action(action[0], action[1], action[2:6])
        time.sleep(self.dt)                   # deja avanzar la fisica del bridge

        fb = self._node.feedback()
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        reward = _compute_reward(fb, guidance, action, self._rs)

        self._ep_steps += 1
        done, reached = _terminated(fb, self.goal_xy, self._ep_steps, self.max_steps)

        obs = _build_obs(guidance, fb)
        info = {"wp": guidance["wp"], "reached": reached,
                "dist_goal": float(np.linalg.norm(fb["xy"] - self.goal_xy))}
        self._fb = fb
        return obs, reward, done, info

    # ── Cierre ───────────────────────────────────────────────────────────────
    def close(self):
        self._node.stop_robot()
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()
