"""
arm_env.py  —  Env MuJoCo para Modelo A (brazo 6-DOF + garra).

Observaciones — mismo formato dict que los demás envs:
  images      : 3 cámaras RGB con flip → (9, H, W)
  lidar       : 7 rangefinders normalizados → (7,)
  joint_states: qpos+qvel del brazo + dedos → (16,)

Acciones (8 valores en [-1, 1]):
  [0..2]  velocidad lineal del end-effector (vx, vy, vz), en el frame arm_base_link
  [3..5]  velocidad angular del end-effector (wx, wy, wz), en el frame arm_base_link
  [6]     dedo izquierdo
  [7]     dedo derecho

La acción es un twist cartesiano del end-effector, no velocidad articular: se
convierte a velocidad articular vía Jacobiano + pseudo-inversa amortiguada
(damped least squares) en cartesian_twist_to_joint_vel(), exactamente el mismo
cálculo que hace MoveIt Servo en deployment real (ver ros_bridge.py /
send_arm_cartesian_vel). La base permanece quieta en este env.

MAX_LINEAR_VEL / MAX_ANGULAR_VEL deben coincidir con scale.linear /
scale.rotational en servo_params.yaml (ambos 1.0 por defecto), para que una
acción a=1.0 produzca la misma velocidad cartesiana en sim y en deployment.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco

# ──────────────────────────── Constantes ───────────────────────────────────
CAMERA_NAMES        = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W  = 84, 84
NUM_LIDAR           = 7
LIDAR_MAX           = 15.0
LIDAR_SPIN_VEL      = 20.0

ARM_JOINTS  = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
               "pos_joint_4", "pos_joint_5", "pos_joint_6"]
FINGER_L    = "pos_left_finger"
FINGER_R    = "pos_right_finger"
LIDAR_SPIN  = "vel_lidar_spin"

# Actuadores observados en joint_states (brazo + dedos)
OBS_ACTUATORS = ARM_JOINTS + [FINGER_L, FINGER_R]

MAX_LINEAR_VEL  = 1.0   # m/s   — debe igualar scale.linear en servo_params.yaml
MAX_ANGULAR_VEL = 1.0   # rad/s — debe igualar scale.rotational en servo_params.yaml
DLS_DAMPING     = 0.05  # damping de la pseudo-inversa cerca de singularidades
JOINT_RANGE     = 3.1416

REST_ANGLES = {
    "joint_1": -0.314, "joint_2": -3.14, "joint_3": 3.14,
    "joint_4":  0.0,   "joint_5":  0.0,  "joint_6": 0.0,
}

CONTROL_DECIMATION  = 10
EPISODE_MAX_STEPS   = 500


# ──────────────────────────── IK cartesiana compartida ─────────────────────
def cartesian_twist_to_joint_vel(model, data, ee_id: int, arm_dof_adr,
                                  base_id: int, twist_base: np.ndarray,
                                  damping: float = DLS_DAMPING) -> np.ndarray:
    """
    Convierte un twist cartesiano del end-effector (expresado en el frame de
    base_id) a velocidades articulares del brazo, vía Jacobiano + pseudo-inversa
    amortiguada (damped least squares). Mismo cálculo que hace MoveIt Servo.

    Usado por ArmMuJoCoEnv y por combined_env.py — para no duplicar la IK.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
    J = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])

    # frame de base_id -> world (si base_id no se mueve durante el step, da igual)
    R = data.xmat[base_id].reshape(3, 3)
    twist_world = np.concatenate([R @ twist_base[:3], R @ twist_base[3:]])

    J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6))
    return J_pinv @ twist_world


class ArmMuJoCoEnv:
    """
    Env para Modelo A.

    Observación: dict {"images": (9,H,W), "lidar": (7,), "joint_states": (16,)}
    — mismo formato que BaseMuJoCoEnv y AesirMuJoCoEnv.

    Acción: (8,) en [-1,1]
      [0..2] vel. lineal EE  [3..5] vel. angular EE  [6] dedo izq  [7] dedo der
    """

    def __init__(self,
                 xml_path: str,
                 camera_names: List[str] = CAMERA_NAMES,
                 image_hw: Tuple[int, int] = (CAMERA_H, CAMERA_W),
                 control_decimation: int = CONTROL_DECIMATION,
                 max_steps: int = EPISODE_MAX_STEPS,
                 render: bool = False):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.max_steps          = max_steps
        self.control_decimation = control_decimation
        self._step_count        = 0
        self._dt                = self.model.opt.timestep * control_decimation

        # ── cámaras ────────────────────────────────────────────────────────
        self.image_h, self.image_w = image_hw
        self.renderer     = mujoco.Renderer(self.model,
                                            height=self.image_h,
                                            width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)

        # ── lidar ──────────────────────────────────────────────────────────
        self.num_lidar = NUM_LIDAR
        self.lidar_adr = []
        for i in range(NUM_LIDAR):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} no encontrado")
            self.lidar_adr.append(int(self.model.sensor_adr[sid]))
        self.id_lidar_spin = self._aid(LIDAR_SPIN)

        # ── actuadores del brazo ───────────────────────────────────────────
        self.ids_arm   = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l = self._aid(FINGER_L)
        self.id_fing_r = self._aid(FINGER_R)

        # ── joint_states: qpos+qvel de brazo + dedos ──────────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)

        # Direcciones de qvel de las 6 articulaciones del brazo (mismo orden que ARM_JOINTS),
        # usadas para extraer las columnas del Jacobiano correspondientes al brazo.
        self._arm_dof_adr = self._qvel_adr[:6]

        # ── end-effector body ──────────────────────────────────────────────
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "logitech_gripper_assembly"
        )
        if self.ee_id < 0:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")

        # ── base body ──────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── tamaños expuestos — mismos atributos que los demás envs ────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (NUM_LIDAR,)
        self.joint_len   = 2 * len(self._obs_act_ids)
        self.joint_shape = (self.joint_len,)
        self.act_len     = 8

        # ── integrador de posición del brazo (resultado de integrar la IK) ──
        self._joint_pos = np.zeros(6, dtype=np.float64)

        # goal externo (opcional, para reward)
        self.goal_pos: Optional[np.ndarray] = None

        # ── viewer ─────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 2.0
            self.viewer.cam.elevation = -20

    # ── utilidad ───────────────────────────────────────────────────────────
    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuador no encontrado: '{name}'")
        return i

    # ── Aplicar accion: twist cartesiano -> IK -> integrador (≡ MoveIt Servo) ──
    def _apply_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)

        twist_base = a[:6] * np.array([MAX_LINEAR_VEL] * 3 + [MAX_ANGULAR_VEL] * 3)
        qvel_arm = cartesian_twist_to_joint_vel(
            self.model, self.data, self.ee_id, self._arm_dof_adr, self.base_id, twist_base
        )

        delta = qvel_arm * self._dt
        self._joint_pos = np.clip(self._joint_pos + delta, -JOINT_RANGE, JOINT_RANGE)
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]

        self.data.ctrl[self.id_fing_l] = float(np.clip((a[6] + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_fing_r] = float(np.clip((a[7] + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

    # ── Observaciones ──────────────────────────────────────────────────────
    def _read_cameras(self) -> np.ndarray:
        """→ (9, H, W) float32 en [0,1].
        np.flip(..., axis=(0,1)) equivale a cv2.flip(img, -1).
        """
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            img = np.flip(img, axis=(0, 1))
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        """→ (7,) float32 en [0,1]."""
        lidar = np.empty(NUM_LIDAR, dtype=np.float32)
        for i, adr in enumerate(self.lidar_adr):
            d = float(self.data.sensordata[adr])
            if d <= 0.0 or d >= LIDAR_MAX:
                d = LIDAR_MAX
            lidar[i] = d / LIDAR_MAX
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        """→ (16,) float32: qpos+qvel de joint_1..6 + dedos."""
        qpos = self.data.qpos[self._qpos_adr]
        qvel = self.data.qvel[self._qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    # ── Reward ────────────────────────────────────────────────────────────
    def _reward(self) -> float:
        # Minimizar distancia EE → goal si existe
        if self.goal_pos is not None and self.ee_id >= 0:
            dist = float(np.linalg.norm(self.data.xpos[self.ee_id] - self.goal_pos))
            rew  = -2.0 * dist + 0.01
        else:
            rew = 0.01

        # action_cost sobre todos los actuadores del brazo
        action_cost = 1e-3 * float(np.square(self.data.ctrl[self._obs_act_ids]).mean())
        return rew - action_cost

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        return self._step_count >= self.max_steps

    # ── API pública ────────────────────────────────────────────────────────
    def reset(self, keep_base: bool = True) -> Dict[str, np.ndarray]:
        if keep_base:
            base_qpos = self.data.qpos[:7].copy()
            base_qvel = self.data.qvel[:6].copy()
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = base_qpos
            self.data.qvel[:6] = base_qvel
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0] = -1.5
            self.data.qpos[1] =  3.5
            self.data.qpos[2] =  0.2

        for nombre, angulo in REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo

        self._joint_pos = np.array([REST_ANGLES[f"joint_{i+1}"] for i in range(6)])
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

        return self._observation()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        self._step_count += 1
        obs  = self._observation()
        rew  = self._reward()
        done = self._terminated()
        return obs, rew, done, {}

    def set_goal(self, goal_xyz: np.ndarray):
        self.goal_pos = np.array(goal_xyz, dtype=np.float64)

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass


# ──────────────────────────── Env vía ROS2 real ─────────────────────────────
# Mismos pesos de reward que ArmMuJoCoEnv._reward (2.0 / 0.01), nombrados aqui
# para no repetir literales magicos en ArmServoEnv.
W_GOAL  = 2.0
W_ALIVE = 0.01

# Mismos joints que ARM_JOINTS/FINGER_L/FINGER_R pero en convencion ROS (sin
# el prefijo "pos_" de los actuadores MuJoCo) — los nombres que aparecen en
# /joint_states real.
ROS_ARM_JOINTS    = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ROS_FINGER_JOINTS = ["left_finger_joint", "right_finger_joint"]
ROS_OBS_JOINTS    = ROS_ARM_JOINTS + ROS_FINGER_JOINTS

SERVO_CONTROL_DT      = 0.1   # segundos reales por step() — no se puede acelerar
SERVO_EPISODE_MAX_STEPS = 200  # 200*0.1s = 20s reales por episodio
SERVO_EE_FRAME   = "tool_link"
SERVO_BASE_FRAME = "arm_base_link"


class ArmServoEnv:
    """
    Misma interfaz reset()/step()/set_goal() que ArmMuJoCoEnv, mismos pesos de
    reward y misma semantica de accion (8 valores en [-1,1]: twist cartesiano
    del EE + 2 dedos) — pero controlando el brazo a traves del stack ROS2 real
    (MoveIt Servo + ros2_control) en vez de llamar mujoco.mj_step() directo.

    Requiere, en otras terminales, ANTES de instanciar este env:
        ros2 launch robot_moveit_config bringup_viz.launch.py
        cd rl_ws && python3 mujoco_ros_bridge.py

    Diferencias obligadas frente a ArmMuJoCoEnv (no son elecciones de diseño,
    son restricciones del control via ROS2):
      - step() bloquea SERVO_CONTROL_DT segundos de tiempo REAL por llamada —
        no hay mj_step() directo que acelerar, el fisico corre en otro proceso.
      - reset() no puede teletransportar el brazo via Servo (solo velocidad
        incremental) — usa el servicio /mujoco_ros_bridge/reset_arm, que sí
        teletransporta el estado fisico dentro de mujoco_ros_bridge.py.
      - Observacion: solo joint_states (16,) — no existe bridge de
        camaras/lidar via ROS2, asi que no hay "images"/"lidar" en el dict.
        Por eso necesita una red mas simple (ver train_arm_servo.py).
      - Posicion del EE para el reward: via TF (arm_base_link -> tool_link),
        no via data.xpos directo (no hay acceso directo a MuJoCo aqui).
    """

    def __init__(self, node_name: str = "arm_servo_env", max_steps: int = SERVO_EPISODE_MAX_STEPS):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        from geometry_msgs.msg import TwistStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from std_srvs.srv import Trigger
        import tf2_ros

        self._rclpy = rclpy
        self._TwistStamped = TwistStamped
        self._Float64MultiArray = Float64MultiArray
        self._Trigger = Trigger

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(node_name)

        self.max_steps   = max_steps
        self._step_count = 0
        self.goal_pos: Optional[np.ndarray] = None
        self._joint_state = None

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                          history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self._pub_twist  = self._node.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", qos)
        self._pub_finger = self._node.create_publisher(Float64MultiArray, "/gripper_controller/commands", 10)
        self._node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self._reset_cli = self._node.create_client(Trigger, "/mujoco_ros_bridge/reset_arm")

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        self.joint_len = 2 * len(ROS_OBS_JOINTS)
        self.act_len   = 8

    # ── ROS plumbing ──────────────────────────────────────────────────────
    def _joint_state_cb(self, msg) -> None:
        self._joint_state = msg

    def _spin_for(self, duration_s: float) -> None:
        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            self._rclpy.spin_once(self._node, timeout_sec=max(0.0, t_end - time.monotonic()))

    def _wait_for_joint_state(self, timeout_s: float = 5.0) -> None:
        t0 = time.monotonic()
        while self._joint_state is None:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(
                    "No llego /joint_states — verifica que bringup_viz.launch.py "
                    "y mujoco_ros_bridge.py esten corriendo."
                )

    def _ee_position(self) -> Optional[np.ndarray]:
        try:
            tf = self._tf_buffer.lookup_transform(
                SERVO_BASE_FRAME, SERVO_EE_FRAME, self._rclpy.time.Time()
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=np.float64)
        except Exception:
            return None

    # ── Observacion ───────────────────────────────────────────────────────
    def _read_joint_state(self) -> np.ndarray:
        msg = self._joint_state
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        qpos = np.zeros(len(ROS_OBS_JOINTS), dtype=np.float32)
        qvel = np.zeros(len(ROS_OBS_JOINTS), dtype=np.float32)
        for k, name in enumerate(ROS_OBS_JOINTS):
            idx = name_to_idx.get(name)
            if idx is not None:
                if idx < len(msg.position): qpos[k] = msg.position[idx]
                if idx < len(msg.velocity): qvel[k] = msg.velocity[idx]
        return np.concatenate([qpos, qvel])

    def _observation(self) -> Dict[str, np.ndarray]:
        return {"joint_states": self._read_joint_state()}

    # ── Reward — misma formula que ArmMuJoCoEnv._reward ────────────────────
    def _reward(self, action: np.ndarray) -> float:
        ee = self._ee_position()
        if self.goal_pos is not None and ee is not None:
            dist = float(np.linalg.norm(ee - self.goal_pos))
            rew = -W_GOAL * dist + W_ALIVE
        else:
            rew = W_ALIVE
        action_cost = 1e-3 * float(np.square(action).mean())
        return rew - action_cost

    # ── API publica — misma firma que ArmMuJoCoEnv ─────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        while not self._reset_cli.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("Esperando /mujoco_ros_bridge/reset_arm...")
        future = self._reset_cli.call_async(self._Trigger.Request())
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)

        self._joint_state = None
        self._wait_for_joint_state()
        self._step_count = 0
        return self._observation()

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)

        twist = self._TwistStamped()
        twist.header.stamp = self._node.get_clock().now().to_msg()
        twist.header.frame_id = SERVO_BASE_FRAME
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = (float(v) for v in a[0:3])
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = (float(v) for v in a[3:6])
        self._pub_twist.publish(twist)

        finger_msg = self._Float64MultiArray()
        finger_msg.data = [
            float(np.clip((a[6] + 1.0) / 2.0 * 0.03, 0.0, 0.03)),
            float(np.clip((a[7] + 1.0) / 2.0 * 0.03, 0.0, 0.03)),
        ]
        self._pub_finger.publish(finger_msg)

        self._spin_for(SERVO_CONTROL_DT)

        self._step_count += 1
        obs  = self._observation()
        rew  = self._reward(a)
        done = self._step_count >= self.max_steps
        return obs, rew, done, {}

    def set_goal(self, goal_xyz: np.ndarray) -> None:
        self.goal_pos = np.array(goal_xyz, dtype=np.float64)

    def close(self) -> None:
        twist = self._TwistStamped()
        twist.header.frame_id = SERVO_BASE_FRAME
        self._pub_twist.publish(twist)
        self._node.destroy_node()
