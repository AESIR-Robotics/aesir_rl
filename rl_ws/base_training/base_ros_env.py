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
import threading
import time
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from hardware.msg import JointControl
from std_srvs.srv import Trigger
from std_msgs.msg import Int32, Float32MultiArray

from rl_ws.global_navigator import (
    plan_route, GlobalNavigator, quat_to_yaw, quat_upright, quat_to_grav_body,
    build_platform_zone, plan_platform_route, plan_platform_route_with_obstacle,
)
from rl_ws.base_training.map_context import MapContext
# Logica de tarea COMPARTIDA con el backend directo-MuJoCo (funciones puras de
# (fb, guidance)/config, sin ROS ni MuJoCo). Se importan en vez de espejarlas:
# antes este modulo tenia su propia copia de la observacion y el reward
# (_build_obs/_RewardState/_compute_reward) y habia que replicar a mano cada
# cambio, con el riesgo de que divergieran en silencio. El desglose por termino
# que imprime test_base.py lo publica la version compartida en rs.last_terms.
from rl_ws.base_training.base_env import (
    flipper_targets, flipper_edge,
    build_obs, RewardState, compute_reward, terminated,
)

# TODOS los parametros (escalas de comando, mision, geometria de flippers,
# lookahead, OBS/ACT y la seccion ROS2) viven en base_training/config.py — el
# MISMO archivo que usa el pipeline de entrenamiento rapido, asi la politica ve
# identica tarea aqui que en train_fast. Los PESOS del reward ya no se importan
# aqui: los consume base_env.compute_reward, que es quien calcula el reward.
from rl_ws.base_training.config import (
    NAV_JSON,
    V_MAX_MPS, W_MAX_RADPS,
    FLIPPER_MIN_RAD, FLIPPER_MAX_RAD, FLIPPER_HOME_RAD,
    START_XY, GOAL_XY, EPISODE_MAX_STEPS,
    N_LOOKAHEAD, LOOKAHEAD_STEP, OBS_DIM, ACT_DIM,
    FLIPPER_JOINTS, CONTROL_HZ, SIM_SPEEDUP,
    USE_HEATMAP, OCTOMAP_BT_PATH, OCTOMAP_RESOLUTION, HEATMAP_RADIUS_M, HEATMAP_PIXELS,
    USE_VIRTUAL_OBSTACLE, GOAL_XY_RANGE,
    TRACK_DEFS, ACTIVE_TRACKS,
)
# Pista ACTIVA (primera de ACTIVE_TRACKS): define el mundo que ve el env ROS.
# El bridge (mujoco_sim_rosbridge.py) simula esta MISMA pista. Para probar otra,
# cambia el orden de ACTIVE_TRACKS en config.py.
_ACTIVE_TRACK = TRACK_DEFS[ACTIVE_TRACKS[0]]


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
        self._grav_body = np.array([0., 0., -1.], dtype=np.float32)  # gravedad en cuerpo
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
            self._grav_body = quat_to_grav_body(quat)

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
                upright=self._upright, grav_body=self._grav_body.copy(),
                twist=self._twist.copy(),
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
    def publish_action(self, v_norm: float, w_norm: float, flip_rad: np.ndarray):
        """flip_rad = angulo objetivo (rad) de los 4 flippers, YA resuelto por
        base_env.flipper_targets. El caller lo calcula; aqui solo se recorta al
        limite por software y se convierte a convencion hardware."""
        tw = Twist()
        tw.linear.x  = float(np.clip(v_norm, -1.0, 1.0)) * V_MAX_MPS
        tw.angular.z = float(np.clip(w_norm, -1.0, 1.0)) * W_MAX_RADPS
        self.cmd_vel_pub.publish(tw)

        jc = JointControl()
        jc.header.stamp = self.get_clock().now().to_msg()
        jc.joint_names  = list(FLIPPER_JOINTS)
        jc.position     = [ros_to_hw(float(np.clip(f, FLIPPER_MIN_RAD, FLIPPER_MAX_RAD)))
                           for f in flip_rad]
        self.joint_pub.publish(jc)

    def stop_robot(self):
        self.publish_action(0.0, 0.0, np.full(4, FLIPPER_HOME_RAD))

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
                 use_platform: Optional[bool] = None):
        # use_platform: None (default) -> se DERIVA de la pista activa
        #   (ACTIVE_TRACKS[0]): kind="platform" -> True, kind="pallets" -> False.
        #   Asi el env ROS ve el MISMO mundo que el bridge y el entrenamiento.
        #   True  -> plataforma: ruta directa + obstaculo virtual + zona segura.
        #   False -> pallets: A* sobre los pallets/sticks del JSON (mision fija).
        if use_platform is None:
            use_platform = (_ACTIVE_TRACK.get("kind", "platform") == "platform")
        # nav_json y octomap de la pista activa (config ya los resolvio arriba,
        # pero en modo pallets usamos el JSON de esa pista explicitamente).
        if not use_platform:
            nav_json = _ACTIVE_TRACK.get("nav_json", nav_json)

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
        print(f"[base_env] pista={ACTIVE_TRACKS[0]} "
              f"({'plataforma' if use_platform else 'pallets A*'}): "
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

        self._rs = RewardState()
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
        self._rs.reset(fb["xy"], float(np.linalg.norm(fb["xy"] - np.asarray(guidance["target"]))), fb["yaw"])
        self._ep_steps = 0
        self._fb = fb
        return build_obs(guidance, fb, self._get_heatmap(fb))

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # accion = [v, ω, flipper×4]; los angulos salen de base_env.flipper_targets
        # (MISMA logica que el backend directo).
        flip_rad = flipper_targets(action)
        if flip_rad is None:                         # CONTROL_FLIPPERS=False -> reposo
            flip_rad = np.full(4, FLIPPER_HOME_RAD)
        self._node.publish_action(action[0], action[1], flip_rad)
        time.sleep(self.dt / SIM_SPEEDUP)

        fb = self._node.feedback()
        fb["flipper_edge"] = flipper_edge(self.map_ctx, fb)
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        reward = compute_reward(fb, guidance, action, self._rs, self.goal_xy)

        self._ep_steps += 1
        # MISMA terminacion que el entrenamiento (base_env.terminated): tocar el
        # obstaculo virtual NO corta el episodio -- solo cuesta OBSTACLE_PENALTY
        # en el reward (ver _count_obstacle_contacts) -- y aplica el timeout por
        # no-progreso. Antes esto era una copia local que si cortaba por
        # obstaculo y no tenia timeout, asi que el test media otra tarea que la
        # entrenada y hundia el success rate.
        done, reached, reason = terminated(
            fb, self.goal_xy, self._ep_steps, self.max_steps, self._rs)
        if reason:
            print(f"[base_ros_env] {'🏆' if reached else '🛑'} {reason}")

        obs = build_obs(guidance, fb, self._get_heatmap(fb))
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
