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
    MUJOCO_GL=glfw python3 mujoco_hardware_bridge.py
"""
import os
import math
import threading

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from hardware.msg import JointControl
from std_srvs.srv import Trigger

_HERE    = os.path.dirname(os.path.abspath(__file__))
_FULL    = os.path.join(_HERE, "../models/aesir_complete.xml")
_ROBOT   = os.path.join(_HERE, "../models/aesir_mujoco_robot.xml")
XML_PATH = _FULL if os.path.exists(_FULL) else _ROBOT

# Nombre de joint ROS -> actuador MuJoCo.
JOINT_TO_ACTUATOR = {
    "joint_1": "pos_joint_1", "joint_2": "pos_joint_2", "joint_3": "pos_joint_3",
    "joint_4": "pos_joint_4", "joint_5": "pos_joint_5", "joint_6": "pos_joint_6",
    "flipper_1_joint": "pos_flipper_1", "flipper_2_joint": "pos_flipper_2",
    "flipper_3_joint": "pos_flipper_3", "flipper_4_joint": "pos_flipper_4",
    "left_finger_joint": "pos_left_finger", "right_finger_joint": "pos_right_finger",
}

PHYSICS_HZ = 100.0  # debe igualar update_rate en ros2_controllers.yaml

# Cinematica diferencial de las orugas: separacion entre eje izquierdo/derecho
# (wheel_l/r_* estan en y=+-0.18, ver aesir_mujoco_robot.xml) y radio de rueda
# motriz (geom cylinder size[0] de wheel_l_*/wheel_r_*).
TRACK_SEPARATION = 0.36
WHEEL_RADIUS     = 0.15

# Pose de reset del brazo — verificada contra el modelo: cond(J)=8.08 (lejos de
# singularidad) y sin self-colision (ncon=8, igual al baseline sin brazo).
ARM_RESET_POSE = {
    "joint_1": 0.0, "joint_2": -1.0, "joint_3": 1.8,
    "joint_4": 0.0, "joint_5": -1.4, "joint_6": 0.0,
}

# Spawn del chasis para el reset de episodio RL (frame de aesir_complete.xml,
# mismo frame que obstacles.json). x, y, z, yaw(rad).
SPAWN_POSE = (-2.2, 4.2, 0.20, 0.0)
SPAWN_SETTLE_STEPS = 50  # substeps de fisica para asentar tras teletransportar

# ── Limites de movimiento tipo AVR446 (rampa trapezoidal), en radianes ──────
# PLACEHOLDER: max_vel (rad/s) y max_accel (rad/s^2) — reemplazar con los
# valores reales del motor/reductor (datasheet: steps/rev + RPM max -> rad/s,
# rad/s^2 segun la reduccion mecanica) cuando esten disponibles.
POSITION_JOINT_LIMITS = {
    "joint_1": dict(max_vel=6.0, max_accel=10.0),
    "joint_2": dict(max_vel=3.0, max_accel=10.0),
    "joint_3": dict(max_vel=6.0, max_accel=10.0),
    "joint_4": dict(max_vel=9.0, max_accel=10.0),
    "joint_5": dict(max_vel=9.0, max_accel=10.0),
    "joint_6": dict(max_vel=9.0, max_accel=10.0),
    "flipper_1_joint": dict(max_vel=3.0, max_accel=10.0),
    "flipper_2_joint": dict(max_vel=3.0, max_accel=10.0),
    "flipper_3_joint": dict(max_vel=3.0, max_accel=10.0),
    "flipper_4_joint": dict(max_vel=3.0, max_accel=10.0),
}

# PLACEHOLDER: aceleracion maxima (rad/s^2) de los actuadores de VELOCIDAD
# (ruedas de oruga + rodillos de flipper). Aqui no hay "posicion objetivo",
# solo se limita que tan rapido puede cambiar la velocidad comandada — el
# limite de velocidad en si ya lo da el ctrlrange de cada actuador en el XML.
VELOCITY_MAX_ACCEL = 15.0


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


def _ramp_toward_position(pos: float, vel: float, target: float,
                           max_vel: float, max_accel: float, dt: float):
    """Un paso de rampa trapezoidal (equivalente continuo del calculo de
    tiempo-entre-pasos de AVR446): acelera/frena respetando max_accel, sin
    superar max_vel, y frena a tiempo para llegar a 'target' con velocidad
    cero — no hay overshoot ni oscilacion."""
    error = target - pos
    direction = 1.0 if error >= 0.0 else -1.0
    stoppable_speed = math.sqrt(max(2.0 * max_accel * abs(error), 0.0))
    desired_vel = direction * min(max_vel, stoppable_speed)

    max_dv = max_accel * dt
    dv = max(-max_dv, min(max_dv, desired_vel - vel))
    new_vel = vel + dv
    new_pos = pos + new_vel * dt

    if (target - new_pos) * error < 0.0: 
        return target, 0.0
    return new_pos, new_vel


def _ramp_toward_velocity(vel: float, target_vel: float, max_accel: float, dt: float) -> float:
    """Igual que _ramp_toward_position pero sin objetivo de posicion — solo
    limita la aceleracion con la que la velocidad comandada puede cambiar
    (para ruedas/rodillos, que giran continuamente sin un angulo objetivo)."""
    max_dv = max_accel * dt
    dv = max(-max_dv, min(max_dv, target_vel - vel))
    return vel + dv


class MujocoHardwareBridge(Node):
    def __init__(self):
        super().__init__("mujoco_hardware_bridge")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._substeps = max(1, round((1.0 / PHYSICS_HZ) / self.model.opt.timestep))

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

        """ Orugas: UN actuador de velocidad por lado (vel_track_left/right),
        que mueve el joint "maestro" de cada lado (drive_l_1/drive_r_1). El
        resto de ruedas y rodillos de flipper de ese lado quedan acoplados
        rigidamente al maestro via <equality><joint .../></equality> en el
        XML (polycoef = relacion de radios/engranaje) — no hace falta
        comandarlos por separado, MuJoCo los sincroniza via fuerzas de
        restriccion. El limite de velocidad ya esta en el XML (ctrlrange)."""
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

        self.feedback_pub = self.create_publisher(
            JointState, "/hardware_node/joint_states", 10
        )
        self.command_sub = self.create_subscription(
            JointControl, "/commands_hardware", self._command_cb, 10
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

        # Reset rapido para RL: teletransporta el brazo a ARM_RESET_POSE
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
            f"MuJoCo hardware bridge activo. XML={XML_PATH}  substeps={self._substeps}"
        )

    def _reset_arm_cb(self, request, response) -> Trigger.Response:
        with self._lock:
            for name, angle in ARM_RESET_POSE.items():
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
        response.message = "Arm reset to ARM_RESET_POSE"
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
            for name, angle in ARM_RESET_POSE.items():
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
        _physics_step via _ramp_toward_velocity. El resto de ruedas y
        rodillos de flipper del mismo lado siguen automaticamente por las
        equality constraints del XML. """
        v, w  = msg.linear.x, msg.angular.z
        omega_left  = (v - w * TRACK_SEPARATION / 2.0) / WHEEL_RADIUS
        omega_right = (v + w * TRACK_SEPARATION / 2.0) / WHEEL_RADIUS
        with self._lock:
            self._vel_ramp_target[self._left_wheel_aid]  = omega_left
            self._vel_ramp_target[self._right_wheel_aid] = omega_right

    def _physics_step(self) -> None:
        dt = self.model.opt.timestep
        with self._lock:
            for _ in range(self._substeps):
                for name, lim in POSITION_JOINT_LIMITS.items():
                    pos, vel = _ramp_toward_position(
                        self._pos_ramp_pos[name], self._pos_ramp_vel[name],
                        self._pos_ramp_target[name], lim["max_vel"], lim["max_accel"], dt,
                    )
                    self._pos_ramp_pos[name], self._pos_ramp_vel[name] = pos, vel
                    self.data.ctrl[self._aid[name]] = pos

                for aid, target_vel in self._vel_ramp_target.items():
                    new_vel = _ramp_toward_velocity(
                        self._vel_ramp_current[aid], target_vel, VELOCITY_MAX_ACCEL, dt,
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

        if self.viewer.is_running():
            self.viewer.sync()
        else:
            rclpy.shutdown()

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
