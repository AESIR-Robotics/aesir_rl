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

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from hardware.msg import JointControl
from std_srvs.srv import Trigger

_HERE    = os.path.dirname(os.path.abspath(__file__))
_FULL    = os.path.join(_HERE, "../models/aesir_complete.xml")
_ROBOT   = os.path.join(_HERE, "../models/aesir_mujoco_robot.xml")
XML_PATH = _FULL if os.path.exists(_FULL) else _ROBOT

# Nombre de joint ROS (igual que en rescue_robot.ros2_control.xacro) -> actuador MuJoCo.
# OJO: en el XML de MuJoCo los flippers se llaman "flipper_joint_N", pero en ROS son
# "flipper_N_joint" — el orden de los tokens esta invertido entre los dos modelos.
JOINT_TO_ACTUATOR = {
    "joint_1": "pos_joint_1", "joint_2": "pos_joint_2", "joint_3": "pos_joint_3",
    "joint_4": "pos_joint_4", "joint_5": "pos_joint_5", "joint_6": "pos_joint_6",
    "flipper_1_joint": "pos_flipper_1", "flipper_2_joint": "pos_flipper_2",
    "flipper_3_joint": "pos_flipper_3", "flipper_4_joint": "pos_flipper_4",
    "left_finger_joint": "pos_left_finger", "right_finger_joint": "pos_right_finger",
}

PHYSICS_HZ = 100.0  # debe igualar update_rate en ros2_controllers.yaml

# Pose de reset del brazo — verificada contra el modelo: cond(J)=8.08 (lejos de
# singularidad) y sin self-colision (ncon=8, igual al baseline sin brazo).
ARM_RESET_POSE = {
    "joint_1": 0.0, "joint_2": -1.0, "joint_3": 1.8,
    "joint_4": 0.0, "joint_5": -1.4, "joint_6": 0.0,
}


# ── Conversion de convencion, espejo exacto de topic_bridge_hardware.cpp ───────
def hw_to_ros(rad: float) -> float:
    return math.fmod(rad, 2.0 * math.pi) - math.pi

def ros_to_hw(rad: float) -> float:
    return rad + math.pi


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

        self.feedback_pub = self.create_publisher(
            JointState, "/hardware_node/joint_states", 10
        )
        self.command_sub = self.create_subscription(
            JointControl, "/commands_hardware", self._command_cb, 10
        )

        # Reset rapido para RL: teletransporta el brazo a ARM_RESET_POSE sin
        # tener que mover el brazo de verdad a traves de Servo (eso tomaria
        # varios segundos por episodio, impractico para entrenar).
        self._reset_srv = self.create_service(
            Trigger, "/mujoco_ros_bridge/reset_arm", self._reset_arm_cb
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
            mujoco.mj_forward(self.model, self.data)
        response.success = True
        response.message = "Arm reset to ARM_RESET_POSE"
        return response

    def _command_cb(self, msg: JointControl) -> None:
        with self._lock:
            for name, pos_hw in zip(msg.joint_names, msg.position):
                aid = self._aid.get(name)
                if aid is not None:
                    self.data.ctrl[aid] = hw_to_ros(pos_hw)

    def _physics_step(self) -> None:
        with self._lock:
            for _ in range(self._substeps):
                self.data.qfrc_applied[self._arm_dof_adr] = self.data.qfrc_bias[self._arm_dof_adr]
                mujoco.mj_step(self.model, self.data)

            state = JointState()
            state.header.stamp = self.get_clock().now().to_msg()
            state.name     = list(self._aid.keys())
            state.position = [ros_to_hw(float(self.data.qpos[self._qpos_adr[n]])) for n in state.name]
            state.velocity = [float(self.data.qvel[self._qvel_adr[n]]) for n in state.name]
            self.feedback_pub.publish(state)

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
