#!/usr/bin/env python3
"""
base_ros_env.py — Env de la base (oruga + flippers), Gym-like, sobre el bridge ROS2.

Responsabilidad de este modulo: TODO lo que define "el mundo" que ve la
politica — guia de navegacion, observacion, reward, terminacion, reset — igual
que en la version directa-MuJoCo original (BaseMuJoCoEnv). La diferencia es
que aqui la fisica no la posee este proceso: vive en mujoco_sim_rosbridge.py y se
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
    cd rl_ws && MUJOCO_GL=glfw python3 base_training/mujoco_sim_rosbridge.py
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
from std_msgs.msg import Int32, Float32MultiArray

from rl_ws.global_navigator import (
    plan_route, GlobalNavigator, quat_to_yaw, quat_upright,
    build_platform_zone, plan_platform_route, plan_platform_route_with_obstacle,
)
from rl_ws.base_training.map_context import MapContext

# TODOS los parametros (escalas de comando, mision, pesos de reward, geometria
# de flippers, lookahead, OBS/ACT y la seccion ROS2) viven en base_training/
# config.py — el MISMO archivo que usa el pipeline de entrenamiento rapido, asi
# la politica ve identica tarea aqui que en train_fast.
from rl_ws.base_training.config import (
    NAV_JSON,
    V_MAX_MPS, W_MAX_RADPS, FLIPPER_MAX, FLIPPER_MIN_RAD, FLIPPER_MAX_RAD,
    START_XY, GOAL_XY, FINISH_DIST, EPISODE_MAX_STEPS,
    W_DIRECTION, W_VELOCITY, WP_BONUS, TIME_PENALTY, FALL_PENALTY,
    STUCK_MAX, ENERGY_W, FLIPPER_JERK_W, TILT_W, FLIPPER_COLLISION_W,
    ACCEL_W, ACCEL_DEADZONE, GUIDE_SPEED_SCALE, BACKWARD_W,
    FLIPPER_MOUNTS, FLIPPER_AXIS_SIGN, FLIPPER_L, FLIPPER_COLLISION_DIST,
    N_LOOKAHEAD, LOOKAHEAD_STEP, OBS_DIM, ACT_DIM,
    FLIPPER_JOINTS, CONTROL_HZ, SIM_SPEEDUP,
    USE_HEATMAP, OCTOMAP_BT_PATH, OCTOMAP_RESOLUTION, HEATMAP_RADIUS_M, HEATMAP_PIXELS,
    USE_VIRTUAL_OBSTACLE, GOAL_XY_RANGE, OBSTACLE_PENALTY,
)


# ── Convencion "hardware" (espejo de topic_bridge_hardware.cpp) ──────────────
def hw_to_ros(rad: float) -> float:
    return math.fmod(rad, 2.0 * math.pi) - math.pi

def ros_to_hw(rad: float) -> float:
    return rad + math.pi


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
        self._floor_contact = 0                          # nº de contactos robot<->piso
        self._obstacle_contact = 0                       # nº de contactos robot<->obstaculo

        self.cmd_vel_pub = self.create_publisher(Twist, "hardware_node/cmd_vel", 10)
        self.joint_pub   = self.create_publisher(JointControl, "/commands_hardware", 10)
        # Manda el obstaculo virtual del episodio al bridge (lo hace fisico/visible).
        self.obstacle_pub = self.create_publisher(Float32MultiArray, "/virtual_obstacle", 10)
        self.create_subscription(PoseStamped, "hardware_node/pose", self._pose_cb, 10)
        self.create_subscription(Twist, "hardware_node/state_vel", self._vel_cb, 10)
        self.create_subscription(JointState, "/hardware_node/joint_states", self._js_cb, 10)
        self.create_subscription(Int32, "/hardware_node/floor_contact", self._floor_cb, 10)
        self.create_subscription(Int32, "/hardware_node/obstacle_contact", self._obstacle_cb, 10)

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

    def _floor_cb(self, msg: Int32):
        with self._lock:
            self._floor_contact = int(msg.data)

    def _obstacle_cb(self, msg: Int32):
        with self._lock:
            self._obstacle_contact = int(msg.data)

    # ── Snapshot de feedback ─────────────────────────────────────────────────
    def feedback(self) -> Optional[dict]:
        with self._lock:
            if self._pose_xy is None:
                return None
            return dict(
                xy=self._pose_xy.copy(), z=self._pose_z, yaw=self._yaw,
                upright=self._upright, twist=self._twist.copy(),
                flip_qpos=self._flip_qpos.copy(), flip_qvel=self._flip_qvel.copy(),
                floor_contact=self._floor_contact,
                obstacle_contact=self._obstacle_contact,
            )

    # ── Publicar el obstaculo virtual del episodio al bridge ────────────────
    def publish_obstacle(self, vobs):
        """vobs = Obstacle2D o None. El bridge coloca/redimensiona el geom fisico
        (o lo esconde si None). data = [active, cx, cy, hx, hy]."""
        m = Float32MultiArray()
        if vobs is not None:
            m.data = [1.0, float(vobs.x), float(vobs.y), float(vobs.hx), float(vobs.hy)]
        else:
            m.data = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.obstacle_pub.publish(m)

    # ── Publicar accion ──────────────────────────────────────────────────────
    def publish_action(self, v_norm: float, w_norm: float, flippers: np.ndarray):
        tw = Twist()
        tw.linear.x  = float(np.clip(v_norm, -1.0, 1.0)) * V_MAX_MPS
        tw.angular.z = float(np.clip(w_norm, -1.0, 1.0)) * W_MAX_RADPS
        self.cmd_vel_pub.publish(tw)

        jc = JointControl()
        jc.header.stamp = self.get_clock().now().to_msg()
        jc.joint_names  = list(FLIPPER_JOINTS)
        jc.position     = [ros_to_hw(float(np.clip(np.clip(f, -1.0, 1.0) * FLIPPER_MAX,
                                                    FLIPPER_MIN_RAD, FLIPPER_MAX_RAD)))
                           for f in flippers]
        self.joint_pub.publish(jc)

    def stop_robot(self):
        self.publish_action(0.0, 0.0, np.zeros(4))

    # ── Reset de episodio (servicio del bridge) ─────────────────────────────
    def reset_sim(self, timeout: float = 5.0) -> bool:
        """Llama al servicio /mujoco_ros_bridge/reset_sim de forma THREAD-SAFE.

        El executor spinea en un hilo aparte (_spin_forever); llamar desde el
        hilo principal a wait_for_service() o hacer busy-poll de fut.done() toca
        internals del executor desde otro hilo -> race que re-lanzaba en el hilo
        de spin y lo mataba. En su lugar:
          1) Disponibilidad: se consulta el grafo con service_is_ready() (query
             puro, thread-safe), sin wait_for_service.
          2) Respuesta: se espera un threading.Event que dispara el done_callback
             del future — ese callback lo corre el executor (hilo de spin) cuando
             llega la respuesta; el hilo principal solo hace Event.wait(), nunca
             toca el executor. Sin race."""
        t0 = time.time()
        while not self.reset_cli.service_is_ready() and time.time() - t0 < timeout:
            time.sleep(0.05)
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn("Servicio /mujoco_ros_bridge/reset_sim no disponible")
            return False

        fut = self.reset_cli.call_async(Trigger.Request())
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout):
            self.get_logger().warn("reset_sim: timeout esperando respuesta del bridge")
            return False
        resp = fut.result()
        return resp is not None and resp.success


# ──────────────────────────── Observacion y reward ─────────────────────────
def _build_obs(guidance: dict, fb: dict, heatmap: Optional[np.ndarray] = None) -> np.ndarray:
    state = np.concatenate([
        guidance["obs"],                                    # 3   guia inmediata
        guidance["lookahead"],                              # 3*N puntos futuros
        fb["twist"],                                        # 3  [v_fwd, v_lat, omega]
        fb["flip_qpos"],                                    # 4
        fb["flip_qvel"],                                    # 4
        [fb["upright"]],                                    # 1
    ]).astype(np.float32)
    if heatmap is None:
        return state
    # MISMO orden que base_env.build_obs (usado por BaseMujocoEnv/train_fast):
    # estado plano + heatmap aplanado al final -- asi el checkpoint CNN entrenado
    # en un backend sirve tal cual en el otro.
    return np.concatenate([state, heatmap.astype(np.float32).ravel()])


class _RewardState:
    """Estado entre pasos para el calculo de reward (progreso, stuck, jerk)."""
    def __init__(self):
        self.last_xy = None
        self.last_dist_to_target = 0.0
        self.last_wp = 0
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.stuck = 0
        self.last_terms = {}          # desglose firmado del ultimo reward

    def reset(self, xy: np.ndarray, dist_to_target: float):
        self.last_xy = xy.copy()
        self.last_dist_to_target = dist_to_target
        self.last_wp = 0
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.stuck = 0
        self.last_terms = {}


def _flipper_tips(flip_qpos: np.ndarray) -> np.ndarray:
    """Posicion 3D (x,y,z) de la punta de cada flipper dado su angulo de pivote
    (rad, convencion ROS). Pivote sobre eje y:
        tip = mount + (axis_sign * L * sin(theta), 0, L * cos(theta))"""
    th = np.asarray(flip_qpos, dtype=np.float64)
    tips = FLIPPER_MOUNTS.copy()
    tips[:, 0] += FLIPPER_AXIS_SIGN * FLIPPER_L * np.sin(th)
    tips[:, 2] += FLIPPER_L * np.cos(th)
    return tips


def _flipper_collision_penalty(flip_qpos: np.ndarray) -> float:
    """Castigo por auto-colision de flippers: por cada par de puntas mas cercano
    que FLIPPER_COLLISION_DIST, castigo proporcional a la penetracion. Los pares
    que geometricamente NO pueden chocar (flippers de lados y opuestos) quedan
    siempre lejos del umbral, asi que no disparan. Colisionan sobre todo el par
    trasero (1,2) y el delantero (3,4) al girar uno hacia el otro."""
    tips = _flipper_tips(flip_qpos)
    pen = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(tips[i] - tips[j]))
            if d < FLIPPER_COLLISION_DIST:
                pen += FLIPPER_COLLISION_DIST - d
    return FLIPPER_COLLISION_W * pen


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

    # Desglose FIRMADO del reward (la suma de todos los terminos == reward total).
    terms = {k: 0.0 for k in ("fall", "obstacle", "stuck", "energy", "flipper_jerk",
                              "tilt", "flipper_collision", "accel", "wp_bonus", "progress",
                              "direction", "velocity", "backward", "time")}

    # 1. Caida letal (chasis muy bajo) o tocar piso (salirse de los pallets):
    if fb["z"] < 0.10 or fb.get("floor_contact", 0) > 0:
        terms["fall"] = -FALL_PENALTY
        rs.last_terms = terms
        return -FALL_PENALTY

    # 1b. Choque con el obstaculo virtual fisico (mismo castigo que train_fast).
    if fb.get("obstacle_contact", 0) > 0:
        terms["obstacle"] = -OBSTACLE_PENALTY
        rs.last_terms = terms
        return -OBSTACLE_PENALTY

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

    # 6. Auto-colision de flippers (dos flippers cuyas puntas se tocan) — se
    #    evalua sobre los angulos REALES medidos (fb["flip_qpos"]).
    flipper_collision_pen = _flipper_collision_penalty(fb["flip_qpos"])

    # 7. Aceleraciones FUERTES del chasis (cuidar la integridad del robot):
    #    cambio del twist base entre pasos, normalizado por [V_MAX, V_MAX, W_MAX].
    #    Zona muerta -> deja pasar la aceleracion normal; solo se castiga el
    #    exceso al cuadrado (jolts que dan;an el robot).
    twist = np.asarray(fb["twist"], dtype=np.float32)
    accel = (twist - rs.last_twist) / np.array([V_MAX_MPS, V_MAX_MPS, W_MAX_RADPS],
                                                dtype=np.float32)
    rs.last_twist = twist.copy()
    accel_pen = ACCEL_W * max(0.0, float(np.linalg.norm(accel)) - ACCEL_DEADZONE) ** 2

    penalties = (penalty_stuck + action_cost + flipper_pen + tilt_pen
                 + flipper_collision_pen + accel_pen)
    terms["stuck"]             = -penalty_stuck
    terms["energy"]            = -action_cost
    terms["flipper_jerk"]      = -flipper_pen
    terms["tilt"]              = -tilt_pen
    terms["flipper_collision"] = -flipper_collision_pen
    terms["accel"]             = -accel_pen

    # 7. Waypoint cruzado — al cruzarlo
    #    se devuelve solo el bonus mas las penalizaciones (sin progreso ni
    #    direccion/velocidad ese paso).
    if guidance["wp"] > rs.last_wp:
        rs.last_wp = guidance["wp"]
        rs.last_dist_to_target = float(np.linalg.norm(xy - target_xy))
        terms["wp_bonus"] = WP_BONUS
        rs.last_terms = terms
        return WP_BONUS - penalties

    # 8. Progreso hacia el waypoint actual (delta_dist * boost
    #    exponencial de proximidad, misma formula que la version pallet)
    dist_to_target = float(np.linalg.norm(xy - target_xy))
    delta_dist = rs.last_dist_to_target - dist_to_target
    proximity_multiplier = float(np.exp(-dist_to_target))
    progress_reward = delta_dist * (50.0 + 100.0 * proximity_multiplier)
    rs.last_dist_to_target = dist_to_target

    # 9. Alcanzar la velocidad y orientacion que pide el vortex + castigo por
    #    retroceder. La DISTANCIA al punto-guia es la velocidad deseada (lejos ->
    #    rapido, cerca -> lento, p.ej. al llegar) y cos_t la orientacion. Se premia
    #    acercarse a esa velocidad encarando la guia; retroceder (v_fwd<0) se
    #    castiga -> evita la oscilacion.
    d_guide = float(np.linalg.norm(np.asarray(guidance["vortex"], dtype=float) - xy))
    v_des = V_MAX_MPS * min(d_guide / GUIDE_SPEED_SCALE, 1.0)
    speed_reward = W_VELOCITY * (1.0 - abs(v_fwd - v_des) / V_MAX_MPS) * max(0.0, float(cos_t))
    # Encarar la guia: cos_t>0 premia ir de frente, cos_t<0 (de espaldas) castiga
    # -> rompe la simetria adelante/reversa del chasis y evita que aprenda a ir
    # en reversa hacia la meta cobrando el progreso (que es ciego a la orientacion).
    direction_reward = W_DIRECTION * float(cos_t)
    backward_pen = BACKWARD_W * max(0.0, -v_fwd) / V_MAX_MPS

    terms["progress"]  = progress_reward
    terms["direction"] = direction_reward
    terms["velocity"]  = speed_reward
    terms["backward"]  = -backward_pen
    terms["time"]      = -TIME_PENALTY
    rs.last_terms = terms

    return (progress_reward + direction_reward + speed_reward
            - backward_pen - penalties - TIME_PENALTY)


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
    if fb.get("floor_contact", 0) > 0:
        print("[base_env] 🛑 Episodio terminado: una parte del robot toco el piso")
        return True, False
    if fb.get("obstacle_contact", 0) > 0:
        print("[base_env] 🛑 Episodio terminado: choco con el obstaculo")
        return True, False
    if float(np.linalg.norm(fb["xy"] - goal_xy)) < FINISH_DIST:
        print("[base_env] 🏆 ¡Meta alcanzada!")
        return True, True
    return False, False


# ──────────────────────────── Env Gym-like ──────────────────────────────────
class BaseRosEnv:
    """Env de entrenamiento de la base, sobre mujoco_sim_rosbridge.py via ROS2.

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
                 max_steps: int = EPISODE_MAX_STEPS,
                 use_platform: bool = True):
        # use_platform=True  -> mundo NUEVO: plataforma plana (plataform.xml),
        #   ruta directa + obstaculo virtual + zona segura, IGUAL que el
        #   entrenamiento (BaseMujocoEnv). Es lo que ve la politica CNN actual.
        # use_platform=False -> mundo VIEJO: A* sobre los pallets/sticks de
        #   obstacles.json (mapa que ya no existe fisicamente). Se conserva para
        #   correr checkpoints entrenados con ese mapa.

        self.obs_dim = OBS_DIM
        self.act_len = ACT_DIM
        self.dt = 1.0 / control_hz
        self.max_steps = max_steps
        self.use_platform = use_platform
        self.nav_json = nav_json
        self.start_xy = np.array(start_xy, dtype=np.float64)

        if use_platform:
            # goal fijo si viene por config/arg; None -> random por episodio
            # (mismo rango que el entrenamiento, ±GOAL_XY_RANGE).
            self._fixed_goal = None if goal_xy is None else np.array(goal_xy, dtype=np.float64)
            self.platform_zone = build_platform_zone()   # zona segura (estatica), para el plot
        else:
            # Mundo viejo: goal None -> ultimo pallet del JSON; ruta A* fija.
            if goal_xy is None:
                goal_xy = tuple(json.load(open(nav_json))["pallets"][-1]["center_xy"])
            self._fixed_goal = np.array(goal_xy, dtype=np.float64)
            self.platform_zone = None

        # Ruta inicial (se replanifica en reset() desde el spawn real del bridge
        # cuando use_platform=True). virtual_obstacle=None en modo pallets.
        self.goal_xy = self._sample_goal()
        self.waypoints, self.virtual_obstacle = self._plan_route(self.start_xy, self.goal_xy)
        if use_platform:
            self.nav = GlobalNavigator(
                None, waypoints=self.waypoints,
                n_lookahead=N_LOOKAHEAD, lookahead_step=LOOKAHEAD_STEP,
                obstacles=[self.virtual_obstacle] if self.virtual_obstacle is not None else [],
                edges_zone=self.platform_zone)
        else:
            self.nav = GlobalNavigator(nav_json, waypoints=self.waypoints,
                                       n_lookahead=N_LOOKAHEAD, lookahead_step=LOOKAHEAD_STEP)
        print(f"[base_env] {'Plataforma' if use_platform else 'Pallets (A*)'}: "
              f"{tuple(self.start_xy)} -> {tuple(self.goal_xy)}  "
              f"({len(self.waypoints)} waypoints, "
              f"obstaculo={'si' if self.virtual_obstacle is not None else 'no'})")

        # Contexto de mapa (octomap -> heatmap egocentrico), MISMO patron que
        # BaseMujocoEnv (mujoco_sim_base.py) -- si USE_HEATMAP=True en config.py,
        # OBS_DIM ya incluye HEATMAP_PIXELS**2 y la politica es CNNActorCritic.
        self.map_ctx = None
        if USE_HEATMAP:
            self.map_ctx = MapContext(
                bt_path=OCTOMAP_BT_PATH,
                resolution=OCTOMAP_RESOLUTION,
                radius_m=HEATMAP_RADIUS_M,
                patch_pixels=HEATMAP_PIXELS,
            )

        # ── ROS2 ───────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self._node = _BridgeInterface()
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin_forever, daemon=True)
        self._spin_thread.start()

        print("[base_env] Esperando feedback del bridge en hardware_node/pose ...")
        while self._node.feedback() is None and rclpy.ok():
            time.sleep(0.05)
        print("[base_env] Bridge conectado.")

        self._rs = _RewardState()
        self._ep_steps = 0
        self._fb: Optional[dict] = None

    # ── Heatmap egocentrico (None si USE_HEATMAP=False) ─────────────────────
    def _get_heatmap(self, fb: dict):
        if self.map_ctx is None:
            return None
        return self.map_ctx.get_heatmap(robot_xy=fb["xy"], robot_z=fb["z"], robot_yaw=fb["yaw"])

    # ── Mision (goal + ruta) segun el mundo activo ──────────────────────────
    def _sample_goal(self) -> np.ndarray:
        """goal fijo si se configuro uno; si no (solo modo plataforma), random
        en ±GOAL_XY_RANGE cada episodio, como el entrenamiento."""
        if self._fixed_goal is not None:
            return self._fixed_goal.copy()
        return np.random.uniform(-GOAL_XY_RANGE, GOAL_XY_RANGE, 2)

    def _plan_route(self, spawn_xy, goal_xy):
        """Devuelve (waypoints, virtual_obstacle_or_None) para el mundo activo."""
        if not self.use_platform:
            return plan_route(self.nav_json, tuple(spawn_xy), tuple(goal_xy)), None
        if USE_VIRTUAL_OBSTACLE:
            return plan_platform_route_with_obstacle(tuple(spawn_xy), tuple(goal_xy))
        return plan_platform_route(tuple(spawn_xy), tuple(goal_xy)), None

    # ── Spin resiliente del executor ─────────────────────────────────────────
    def _spin_forever(self):
        """Spinea el executor tolerando errores: si una callback de ROS lanza
        (mensaje transitorio raro, etc.), el executor la re-lanza — con
        executor.spin() eso MATA el hilo y deja de llegar feedback, colgando el
        entrenamiento en silencio. Aqui la atrapamos, la logueamos y seguimos,
        para que un run de horas no muera por un mensaje puntual."""
        while rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                print(f"[base_env] callback ROS fallo (continuo): {e!r}")

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self._node.stop_robot()
        self._node.reset_sim()
        time.sleep(0.1)                       # deja llegar el primer feedback nuevo
        fb = self._node.feedback()
        while fb is None and rclpy.ok():
            time.sleep(0.02)
            fb = self._node.feedback()

        # Modo plataforma: nueva mision cada episodio (goal random + ruta directa
        # con obstaculo virtual) desde el spawn REAL que reporto el bridge, igual
        # que BaseMujocoEnv.reset(). Modo pallets: ruta fija de __init__.
        if self.use_platform:
            self.goal_xy = self._sample_goal()
            self.waypoints, self.virtual_obstacle = self._plan_route(fb["xy"], self.goal_xy)
            self.nav.replan(
                self.waypoints,
                obstacles=[self.virtual_obstacle] if self.virtual_obstacle is not None else [])

        # Manda el obstaculo del episodio al bridge -> aparece FISICO en MuJoCo,
        # igual que en train_fast (None en modo pallets -> el bridge lo esconde).
        self._node.publish_obstacle(self.virtual_obstacle)

        self.nav.reset(fb["xy"])
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        self._rs.reset(fb["xy"], float(np.linalg.norm(fb["xy"] - np.asarray(guidance["target"]))))
        self._ep_steps = 0
        self._fb = fb
        return _build_obs(guidance, fb, self._get_heatmap(fb))

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # accion = [v, ω, flipper×4]: la politica controla base + flippers.
        self._node.publish_action(action[0], action[1], action[2:6])
        time.sleep(self.dt / SIM_SPEEDUP)

        fb = self._node.feedback()
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        reward = _compute_reward(fb, guidance, action, self._rs)

        self._ep_steps += 1
        done, reached = _terminated(fb, self.goal_xy, self._ep_steps, self.max_steps)

        obs = _build_obs(guidance, fb, self._get_heatmap(fb))
        info = {"wp": guidance["wp"], "reached": reached,
                "dist_goal": float(np.linalg.norm(fb["xy"] - self.goal_xy)),
                "reward_terms": dict(self._rs.last_terms),
                # para visualizar la trayectoria en test_base:
                "xy": np.asarray(fb["xy"], dtype=float).copy(),        # pose real del robot
                "guide": np.asarray(guidance["vortex"], dtype=float).copy(),   # punto-guia vortex
                "target": np.asarray(guidance["target"], dtype=float).copy(),  # waypoint objetivo
                "lookahead_xy": np.asarray(guidance["lookahead_xy"], dtype=float)}  # 5 puntos futuros
        self._fb = fb
        return obs, reward, done, info

    # ── Cierre ───────────────────────────────────────────────────────────────
    def close(self):
        self._node.stop_robot()
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()
