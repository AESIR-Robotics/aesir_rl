#!/usr/bin/env python3
"""
mujoco_hardware_bridge.py — Reemplazo de hardware_loopback.py con fisica real de MuJoCo.

hardware_loopback.py hace eco instantaneo: lo que se comanda es lo que se reporta,
sin fisica real. Este nodo en cambio:

  1. Recibe /commands_hardware (hardware.msg.JointControl) — lo que publica
     TopicBridgeHardware::write() (posicion ya en convencion "hardware" [0,2pi]).
     Incluye brazo (joint_1..6), flippers (flipper_1..4_joint) y gripper
     (left/right_finger_joint) — ver target_joints en topic_bridge_hardware.cpp.
  2. Convierte a convencion ROS y aplica como setpoint a los actuadores position
     correspondientes en MuJoCo (ver JOINT_TO_ACTUATOR).
  3. Step fisico real (masa, gravedad, colisiones) a la tasa del modelo.
  4. Publica el estado articular RESULTANTE (no el comando) en
     /hardware_node/joint_states, reconvertido a convencion "hardware" — lo que
     TopicBridgeHardware::read() espera.

Asi, RViz/MoveIt muestran el estado real simulado por MuJoCo (con inercia,
gravedad, etc.), no un espejo instantaneo del comando. Se abre tambien un
viewer de MuJoCo para ver la misma simulacion fisicamente.

NO lances hardware_loopback.py al mismo tiempo — son alternativos, ambos
publican en /hardware_node/joint_states.

Uso (con el resto del stack ya corriendo: bringup_viz.launch.py):
    cd rl_ws
    MUJOCO_GL=glfw python3 base_training/mujoco_sim_rosbridge.py
"""
import os
import sys
import math
import threading

import numpy as np
import mujoco
import mujoco.viewer
import pyoctomap

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from hardware.msg import JointControl
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger

from tf2_ros import TransformBroadcaster
from std_msgs.msg import Int32, Header

# Raiz del proyecto (aesir_rl) al path para resolver `rl_ws.*` corriendo directo.
# (base_training/ -> rl_ws/ -> aesir_rl/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Constantes compartidas (rutas del XML, spawn, SIM_SPEEDUP, cinematica y
# limites de rampa) + las MISMAS funciones de rampa AVR446 que usa el
# entrenamiento directo (robot_control) -> identica respuesta dinamica.
import rl_ws.base_training.config as C
from rl_ws.base_training.robot_control import ramp_toward_position, ramp_toward_velocity

# PISTA ACTIVA: el bridge simula la PRIMERA pista de config.ACTIVE_TRACKS
# (mismo mundo, spawn y octomap que usaria el entrenamiento en esa pista).
# Para probar otra pista por ROS, cambia el orden de ACTIVE_TRACKS en config.
_TRACK = C.TRACK_DEFS[C.ACTIVE_TRACKS[0]]
XML_PATH = C.XML_PATH                     # == _TRACK["xml"]

# Nombre de joint ROS -> actuador MuJoCo.
JOINT_TO_ACTUATOR = {
    "joint_1": "pos_joint_1", "joint_2": "pos_joint_2", "joint_3": "pos_joint_3",
    "joint_4": "pos_joint_4", "joint_5": "pos_joint_5", "joint_6": "pos_joint_6",
    "flipper_1_joint": "pos_flipper_1", "flipper_2_joint": "pos_flipper_2",
    "flipper_3_joint": "pos_flipper_3", "flipper_4_joint": "pos_flipper_4",
    "left_finger_joint": "pos_left_finger", "right_finger_joint": "pos_right_finger",
}

PHYSICS_HZ  = C.PHYSICS_HZ    # debe igualar update_rate en ros2_controllers.yaml
SIM_SPEEDUP = C.SIM_SPEEDUP   # mismo valor que duerme BaseRosEnv (dt/SIM_SPEEDUP)

# Spawn del chasis para el reset de episodio RL: XY/yaw de la pista activa (o la
# mision de config si la pista no define el suyo), Z segun el TECHO de su terreno
# (steps/ramps son mas altos) — mismo spawn que usa BaseMujocoEnv al entrenar en
# esa pista. El maze SI trae el suyo: su entrada no coincide con START_XY, que es
# el arranque de la mision de pallets (otra pista, otro frame).
_SPAWN_XY = _TRACK.get("spawn_xy", C.START_XY)
SPAWN_POSE = (float(_SPAWN_XY[0]), float(_SPAWN_XY[1]),
              float(_TRACK.get("spawn_z", C.SPAWN_Z)),
              float(_TRACK.get("spawn_yaw", C.SPAWN_YAW)))
SPAWN_SETTLE_STEPS = int(_TRACK.get("settle_steps", C.SPAWN_SETTLE_STEPS))

# ── Limites de movimiento tipo AVR446 (rampa trapezoidal), en radianes ──────
# Brazo desde config.ARM_JOINT_LIMITS; flippers con los MISMOS limites que el
# entrenamiento directo (FLIPPER_MAX_VEL / FLIPPER_MAX_ACCEL).
POSITION_JOINT_LIMITS = dict(C.ARM_JOINT_LIMITS)
POSITION_JOINT_LIMITS.update({
    name: dict(max_vel=C.FLIPPER_MAX_VEL, max_accel=C.FLIPPER_MAX_ACCEL)
    for name in C.FLIPPER_JOINTS
})


# ── Conversion de convencion, espejo exacto de topic_bridge_hardware.cpp ───────
def hw_to_ros(rad: float) -> float:
    return math.fmod(rad, 2.0 * math.pi) - math.pi

def ros_to_hw(rad: float) -> float:
    return rad + math.pi



def duty_to_omega(duty: float, omega_max: float) -> float:
    """Traduce una señal de potencia normalizada (-1..1, tipo PWM/duty de un
    driver de motor) a velocidad angular (rad/s) para un actuador de
    velocidad de MuJoCo. omega_max = velocidad sin carga nominal del motor,
    en rad/s (RPM_datasheet * 2*pi/60, dividido por la relacion de reduccion
    si el motor tiene caja reductora). No hace falta modelar la caida de
    velocidad bajo carga a mano — el limite de fuerza del actuador
    (forcerange/actuatorfrcrange) + kv ya reproduce ese efecto."""
    return duty * omega_max


def points_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Arma un PointCloud2 (XYZ float32) a mano, sin depender de sensor_msgs_py
    (no siempre disponible segun la distro/instalacion)."""
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


class MujocoHardwareBridge(Node):
    def __init__(self):
        super().__init__("mujoco_hardware_bridge")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._substeps = max(1, round(SIM_SPEEDUP * (1.0 / PHYSICS_HZ) / self.model.opt.timestep))

        self._aid, self._qpos_adr, self._qvel_adr = {}, {}, {}
        for ros_name, act_name in JOINT_TO_ACTUATOR.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise ValueError(f"Actuador no encontrado: '{act_name}'")
            jid = int(self.model.actuator_trnid[aid, 0])
            self._aid[ros_name]       = aid
            self._qpos_adr[ros_name]  = self.model.jnt_qposadr[jid]
            self._qvel_adr[ros_name]  = self.model.jnt_dofadr[jid]

        self._arm_dof_adr = [self._qvel_adr[n] for n in
                              ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")]

        self._left_wheel_aid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_track_left")
        self._right_wheel_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_track_right")
        self._vel_ramp_target  = {self._left_wheel_aid: 0.0, self._right_wheel_aid: 0.0}
        self._vel_ramp_current = {self._left_wheel_aid: 0.0, self._right_wheel_aid: 0.0}

        # Estado de rampa de posicion (AVR446-style) para brazo + pivotes de
        # flipper. El gripper (left/right_finger_joint) queda fuera: pasa
        # directo, no es un joint tipo stepper con perfil trapezoidal.
        self._pos_ramp_target = {n: float(self.data.qpos[self._qpos_adr[n]]) for n in POSITION_JOINT_LIMITS}
        self._pos_ramp_pos    = dict(self._pos_ramp_target)
        self._pos_ramp_vel    = {n: 0.0 for n in POSITION_JOINT_LIMITS}

        # DOFs/qpos del freejoint del chasis, para leer velocidad base y publicarla
        # como Twist (igual que hardware_node/state_vel en el driver real).
        base_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
        self._base_qpos_adr = self.model.jnt_qposadr[base_jid]
        self._base_dof_adr  = self.model.jnt_dofadr[base_jid]

        # Deteccion de contacto robot <-> PISO (el suelo a evitar, no los pallets).
        # "maze_floor" es el piso CAMINABLE de la pista maze -- se excluye para
        # que caminar sobre el maze no cuente como caida (ver mismo criterio en
        # mujoco_sim_base.py). "fatal_floor", debajo de maze_floor, SI cuenta.
        self._floor_gids = {
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_PLANE
            and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "") != "maze_floor"
        }
        robot_root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "footprint_link")
        self._robot_gids = {
            gid for gid in range(self.model.ngeom)
            if int(self.model.body_rootid[self.model.geom_bodyid[gid]]) == robot_root
        }

        # Geom del obstaculo virtual FISICO (plataform.xml), lo teletransporta y
        # redimensiona _obstacle_cb en cada episodio -- MISMO geom que reposiciona
        # BaseMujocoEnv.reset() en el entrenamiento rapido.
        self._obstacle_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "virtual_obstacle")
        self._obstacle_gids = {self._obstacle_gid} if self._obstacle_gid >= 0 else set()

        self.feedback_pub = self.create_publisher(
            JointState, "/hardware_node/joint_states", 10
        )
        self.floor_contact_pub = self.create_publisher(
            Int32, "/hardware_node/floor_contact", 10
        )
        self.obstacle_contact_pub = self.create_publisher(
            Int32, "/hardware_node/obstacle_contact", 10
        )
        self.command_sub = self.create_subscription(
            JointControl, "/commands_hardware", self._command_cb, 10
        )
        self.obstacle_sub = self.create_subscription(
            Float32MultiArray, "/virtual_obstacle", self._obstacle_cb, 10
        )
        self.vel_state_pub = self.create_publisher(
            Twist, "hardware_node/state_vel", 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, "hardware_node/cmd_vel", self._cmd_vel_cb, 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, "hardware_node/pose", 10
        )


        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Point cloud del octomap para MoveIt (brazo) ──────────────────────
        # Carga UNA sola vez (mapa estatico, no se vuelve a leer el .bt). En
        # cada tick de su propio timer (independiente de PHYSICS_HZ) se filtra
        # por distancia XY al robot (sin Z, pediste circunferencia 2D) y se
        # publica en frame_id="map" -- el MISMO frame fijo que ya usa pose_pub
        # arriba, para que el robot se mueva DENTRO del pointcloud en vez de
        # que el pointcloud se mueva con el robot.
        self.get_logger().info(f"Cargando octomap para MoveIt: {C.OCTOMAP_BT_PATH}")
        _tree = pyoctomap.OcTree(C.OCTOMAP_RESOLUTION)
        if not _tree.readBinary(C.OCTOMAP_BT_PATH):
            raise RuntimeError(f"No se pudo leer {C.OCTOMAP_BT_PATH}")
        _map_points, _ = _tree.extractPointCloud()
        self._map_points = np.asarray(_map_points, dtype=np.float32)
        self.get_logger().info(
            f"{len(self._map_points)} puntos cargados -- publicando en "
            f"'{C.OCTOMAP_POINTCLOUD_TOPIC}' (frame='{C.OCTOMAP_POINTCLOUD_FRAME}', "
            f"radio XY={C.POINTCLOUD_RADIUS_M}m, {C.POINTCLOUD_PUBLISH_HZ}Hz)"
        )

        self.pointcloud_pub = self.create_publisher(
            PointCloud2, C.OCTOMAP_POINTCLOUD_TOPIC, 10
        )
        self._pointcloud_timer = self.create_timer(
            1.0 / C.POINTCLOUD_PUBLISH_HZ, self._publish_pointcloud
        )

        # Reset rapido para RL: teletransporta el brazo a C.ARM_REST_POSE
        self._reset_srv = self.create_service(
            Trigger, "/mujoco_ros_bridge/reset_arm", self._reset_arm_cb
        )
        self._reset_sim_srv = self.create_service(
            Trigger, "/mujoco_ros_bridge/reset_sim", self._reset_sim_cb
        )

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 1.5

        self._timer = self.create_timer(1.0 / PHYSICS_HZ, self._physics_step)
        self.get_logger().info(
            f"MuJoCo hardware bridge activo. pista={C.ACTIVE_TRACKS[0]}  "
            f"XML={XML_PATH}  spawn_z={SPAWN_POSE[2]}  substeps={self._substeps}"
        )

    def _obstacle_cb(self, msg: Float32MultiArray) -> None:
        """Coloca el obstaculo virtual FISICO donde lo planeo el env este episodio.
        Espejo de BaseMujocoEnv.reset(): reescribe model.geom_pos/geom_size del
        geom 'virtual_obstacle'. active<=0 -> lo baja fuera de juego (z=-5)."""
        if self._obstacle_gid < 0:
            return
        d = list(msg.data)
        if len(d) < 5:
            return
        active, cx, cy, hx, hy = d[0], d[1], d[2], d[3], d[4]
        with self._lock:
            if active > 0.5:
                self.model.geom_pos[self._obstacle_gid] = [
                    cx, cy, C.PLATFORM_SURFACE_Z + C.VIRTUAL_OBSTACLE_HEIGHT_HALF]
                self.model.geom_size[self._obstacle_gid] = [
                    hx, hy, C.VIRTUAL_OBSTACLE_HEIGHT_HALF]
            else:
                self.model.geom_pos[self._obstacle_gid, 2] = -5.0   # fuera de juego

    def _reset_arm_cb(self, request, response) -> Trigger.Response:
        with self._lock:
            for name, angle in C.ARM_REST_POSE.items():
                qpos_adr = self._qpos_adr[name]
                qvel_adr = self._qvel_adr[name]
                self.data.qpos[qpos_adr] = angle
                self.data.qvel[qvel_adr] = 0.0
                self.data.ctrl[self._aid[name]] = angle
                if name in POSITION_JOINT_LIMITS:
                    self._pos_ramp_target[name] = angle
                    self._pos_ramp_pos[name]    = angle
                    self._pos_ramp_vel[name]    = 0.0
            mujoco.mj_forward(self.model, self.data)
        response.success = True
        response.message = "Arm reset to C.ARM_REST_POSE"
        return response

    def _reset_sim_cb(self, request, response) -> Trigger.Response:
        """Reset de episodio completo para RL: mj_resetData, chasis al SPAWN_POSE,
        brazo en reposo, velocidades y rampas a cero, y unos substeps para
        asentar. Deja la sim lista para un episodio nuevo y determinista."""
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)

            # Chasis al spawn (qpos del freejoint: x y z + quat wxyz).
            sx, sy, sz, syaw = SPAWN_POSE
            a = self._base_qpos_adr
            self.data.qpos[a + 0] = sx
            self.data.qpos[a + 1] = sy
            self.data.qpos[a + 2] = sz
            self.data.qpos[a + 3] = math.cos(0.5 * syaw)   # w
            self.data.qpos[a + 4] = 0.0
            self.data.qpos[a + 5] = 0.0
            self.data.qpos[a + 6] = math.sin(0.5 * syaw)   # z

            # Brazo en reposo (qpos + ctrl + rampa).
            for name, angle in C.ARM_REST_POSE.items():
                self.data.qpos[self._qpos_adr[name]] = angle
                self.data.qvel[self._qvel_adr[name]] = 0.0
                self.data.ctrl[self._aid[name]] = angle

            # Rampas de posicion a la qpos actual, rampas de velocidad a cero.
            for name in POSITION_JOINT_LIMITS:
                p = float(self.data.qpos[self._qpos_adr[name]])
                self._pos_ramp_target[name] = p
                self._pos_ramp_pos[name]    = p
                self._pos_ramp_vel[name]    = 0.0
            for aid in self._vel_ramp_target:
                self._vel_ramp_target[aid]  = 0.0
                self._vel_ramp_current[aid] = 0.0
                self.data.ctrl[aid] = 0.0

            mujoco.mj_forward(self.model, self.data)
            for _ in range(SPAWN_SETTLE_STEPS):
                self.data.qfrc_applied[self._arm_dof_adr] = self.data.qfrc_bias[self._arm_dof_adr]
                mujoco.mj_step(self.model, self.data)

        response.success = True
        response.message = f"Sim reset to spawn {SPAWN_POSE}"
        return response

    def _command_cb(self, msg: JointControl) -> None:
        with self._lock:
            for name, pos_hw in zip(msg.joint_names, msg.position):
                aid = self._aid.get(name)
                if aid is None:
                    continue
                target = hw_to_ros(pos_hw)
                if name in POSITION_JOINT_LIMITS:
                    self._pos_ramp_target[name] = target
                else:
                    self.data.ctrl[aid] = target  # gripper: passthrough directo, sin rampa

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """ Cinematica diferencial: v_lado = v_lineal +- w*separacion/2, luego a
        rad/s de rueda dividiendo por el radio. El limite de velocidad ya lo
        aplica MuJoCo via ctrlrange en vel_track_left/right; aqui solo se
        fija el objetivo — la rampa de aceleracion se aplica en
        _physics_step via ramp_toward_velocity. El resto de ruedas y
        rodillos de flipper del mismo lado siguen automaticamente por las
        equality constraints del XML. """
        v, w  = msg.linear.x, msg.angular.z
        omega_left  = (v - w * C.TRACK_SEPARATION / 2.0) / C.WHEEL_RADIUS
        omega_right = (v + w * C.TRACK_SEPARATION / 2.0) / C.WHEEL_RADIUS
        with self._lock:
            self._vel_ramp_target[self._left_wheel_aid]  = omega_left
            self._vel_ramp_target[self._right_wheel_aid] = omega_right

    def _physics_step(self) -> None:
        dt = self.model.opt.timestep
        with self._lock:
            for _ in range(self._substeps):
                for name, lim in POSITION_JOINT_LIMITS.items():
                    pos, vel = ramp_toward_position(
                        self._pos_ramp_pos[name], self._pos_ramp_vel[name],
                        self._pos_ramp_target[name], lim["max_vel"], lim["max_accel"], dt,
                    )
                    self._pos_ramp_pos[name], self._pos_ramp_vel[name] = pos, vel
                    self.data.ctrl[self._aid[name]] = pos

                for aid, target_vel in self._vel_ramp_target.items():
                    new_vel = ramp_toward_velocity(
                        self._vel_ramp_current[aid], target_vel, C.VELOCITY_MAX_ACCEL, dt,
                    )
                    self._vel_ramp_current[aid] = new_vel
                    self.data.ctrl[aid] = new_vel

                self.data.qfrc_applied[self._arm_dof_adr] = self.data.qfrc_bias[self._arm_dof_adr]
                mujoco.mj_step(self.model, self.data)

            state = JointState()
            state.header.stamp = self.get_clock().now().to_msg()
            state.name     = list(self._aid.keys())
            state.position = [ros_to_hw(float(self.data.qpos[self._qpos_adr[n]])) for n in state.name]
            state.velocity = [float(self.data.qvel[self._qvel_adr[n]]) for n in state.name]
            self.feedback_pub.publish(state)

            lin_world = self.data.qvel[self._base_dof_adr:self._base_dof_adr + 3]
            ang_body  = self.data.qvel[self._base_dof_adr + 3:self._base_dof_adr + 6]
            quat      = self.data.qpos[self._base_qpos_adr + 3:self._base_qpos_adr + 7]
            quat_inv  = np.zeros(4)
            mujoco.mju_negQuat(quat_inv, quat)
            lin_body = np.zeros(3)
            mujoco.mju_rotVecQuat(lin_body, lin_world, quat_inv)

            vel_msg = Twist()
            vel_msg.linear.x  = float(lin_body[0])
            vel_msg.linear.y  = float(lin_body[1])
            vel_msg.linear.z  = float(lin_body[2])
            vel_msg.angular.x = float(ang_body[0])
            vel_msg.angular.y = float(ang_body[1])
            vel_msg.angular.z = float(ang_body[2])
            self.vel_state_pub.publish(vel_msg)

            pos = self.data.qpos[self._base_qpos_adr:self._base_qpos_adr + 3]
            pose_msg = PoseStamped()
            pose_msg.header.stamp = state.header.stamp
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            pose_msg.pose.orientation.w = float(quat[0])
            pose_msg.pose.orientation.x = float(quat[1])
            pose_msg.pose.orientation.y = float(quat[2])
            pose_msg.pose.orientation.z = float(quat[3])
            self.pose_pub.publish(pose_msg)

            t = TransformStamped()
            t.header.stamp = state.header.stamp
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            t.transform.rotation.w = float(quat[0])
            self.tf_broadcaster.sendTransform(t)

            n_floor = 0
            n_obstacle = 0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if ((g1 in self._floor_gids and g2 in self._robot_gids) or
                        (g2 in self._floor_gids and g1 in self._robot_gids)):
                    n_floor += 1
                elif ((g1 in self._obstacle_gids and g2 in self._robot_gids) or
                        (g2 in self._obstacle_gids and g1 in self._robot_gids)):
                    n_obstacle += 1
            self.floor_contact_pub.publish(Int32(data=n_floor))
            self.obstacle_contact_pub.publish(Int32(data=n_obstacle))

        if self.viewer.is_running():
            self.viewer.sync()
        else:
            rclpy.shutdown()

    def _publish_pointcloud(self) -> None:
        """Filtra el mapa por distancia XY (sin Z) al robot y publica. Timer
        propio, independiente de _physics_step -- el mapa no cambia, no hace
        falta recalcular a PHYSICS_HZ."""
        with self._lock:
            robot_xy = np.array(
                [self.data.qpos[self._base_qpos_adr], self.data.qpos[self._base_qpos_adr + 1]],
                dtype=np.float32,
            )
            stamp = self.get_clock().now().to_msg()

        dist_xy = np.linalg.norm(self._map_points[:, :2] - robot_xy, axis=1)
        nearby = self._map_points[dist_xy <= C.POINTCLOUD_RADIUS_M]

        msg = points_to_pointcloud2(nearby, C.OCTOMAP_POINTCLOUD_FRAME, stamp)
        self.pointcloud_pub.publish(msg)

    def destroy_node(self) -> None:
        try:
            self.viewer.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = MujocoHardwareBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()