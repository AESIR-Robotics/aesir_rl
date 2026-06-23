#!/usr/bin/env python3
"""
test_pipeline.py — Prueba de la secuencia completa via ROS2 + MoveIt 2.

Replica la misma secuencia de test_actuators.py pero usando el pipeline real:
  Brazo   → MoveGroup action  (move_group → arm_controller/follow_joint_trajectory)
  Base    → /diff_drive_controller/cmd_vel  (geometry_msgs/Twist)
  Flippers→ /flipper_controller/commands    (std_msgs/Float64MultiArray)

PREREQUISITOS (deben estar corriendo antes de lanzar este script):
  1. MuJoCo con mujoco_ros2_control:
       ros2 launch robot_moveit_config bringup.launch.py
  2. MoveIt 2 (move_group):
       ros2 launch robot_moveit_config move_group.launch.py

Uso:
    source /opt/ros/humble/setup.bash
    source <workspace>/install/setup.bash
    python3 test_pipeline.py
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

# ── Constantes de la arquitectura ────────────────────────────────────────────
ARM_JOINTS   = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
FLIP_JOINTS  = ["flipper_1_joint", "flipper_2_joint",
                "flipper_3_joint", "flipper_4_joint"]
ARM_GROUP    = "arm"

# Estados SRDF definidos en rescue_robot.srdf
NAMED_STATES = {
    "home":      [0.0, -2.8625,  2.8625, -1.5708, -1.5708, 0.0],
    "extension": [0.0, -0.6766,  0.6766,  1.5708,  1.5708, 0.0],
    "arm_up":    [0.0,  1.4,    -1.2,     0.0,     1.57,   0.0],
}

BASE_LINEAR_VEL  = 0.4   # m/s  (límite en ros2_controllers.yaml: 0.5)
BASE_ANGULAR_VEL = 0.9   # rad/s (límite: 1.0)
PHASE_DURATION   = 3.0   # segundos de wall-clock por fase


# ─────────────────────────────────────────────────────────────────────────────
class PipelineTestNode(Node):
    """Nodo ROS2 que encapsula todos los publishers / action clients del test."""

    def __init__(self) -> None:
        super().__init__("aesir_pipeline_test")

        # Base
        self._pub_base = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel", 10
        )
        # Flippers
        self._pub_flip = self.create_publisher(
            Float64MultiArray, "/flipper_controller/commands", 10
        )
        # Brazo — MoveGroup action client
        self._arm_client = ActionClient(self, MoveGroup, "move_group")

        self.get_logger().info("PipelineTestNode iniciado.")

    # ── Base ────────────────────────────────────────────────────────────────
    def cmd_base(self, v_lin: float, w_ang: float) -> None:
        msg = Twist()
        msg.linear.x  = float(v_lin)
        msg.angular.z = float(w_ang)
        self._pub_base.publish(msg)

    def stop_base(self) -> None:
        self.cmd_base(0.0, 0.0)

    # ── Flippers ────────────────────────────────────────────────────────────
    def cmd_flippers(self, positions: list[float]) -> None:
        """positions: [flip1, flip2, flip3, flip4] en radianes."""
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]
        self._pub_flip.publish(msg)

    def stop_flippers(self) -> None:
        self.cmd_flippers([0.0, 0.0, 0.0, 0.0])

    # ── Brazo — MoveGroup ────────────────────────────────────────────────────
    def _build_joint_goal(
        self,
        target_positions: list[float],
        vel_scale: float = 0.5,
        accel_scale: float = 0.5,
        planning_time: float = 5.0,
    ) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                   = ARM_GROUP
        req.allowed_planning_time        = planning_time
        req.num_planning_attempts        = 3
        req.max_velocity_scaling_factor     = vel_scale
        req.max_acceleration_scaling_factor = accel_scale

        constraints = Constraints()
        for name, pos in zip(ARM_JOINTS, target_positions):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.replan    = False
        return goal

    def send_arm_goal(
        self,
        target_positions: list[float],
        vel_scale: float = 0.5,
        label: str = "",
    ) -> bool:
        """Envía goal al brazo y bloquea hasta completar. Retorna True si OK."""
        if not self._arm_client.server_is_ready():
            self.get_logger().warning("MoveGroup no está listo — saltando goal de brazo.")
            return False

        goal_msg = self._build_joint_goal(target_positions, vel_scale=vel_scale)
        self.get_logger().info(
            f"  → Enviando goal{' (' + label + ')' if label else ''}: "
            f"{[f'{v:.2f}' for v in target_positions]}"
        )

        future = self._arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("  ✗ Goal rechazado o timeout de aceptación.")
            return False

        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("  ✗ Goal rechazado por el servidor.")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done():
            self.get_logger().error("  ✗ Timeout esperando resultado.")
            return False

        ec = result_future.result().result.error_code.val
        if ec == 1:  # MoveItErrorCodes.SUCCESS
            self.get_logger().info("  ✓ Goal completado.")
            return True
        else:
            self.get_logger().error(f"  ✗ MoveIt error code: {ec}")
            return False

    def wait_for_arm_server(self, timeout_sec: float = 15.0) -> bool:
        self.get_logger().info("Esperando servidor move_group…")
        ready = self._arm_client.wait_for_server(timeout_sec=timeout_sec)
        if ready:
            self.get_logger().info("✓ move_group disponible.")
        else:
            self.get_logger().warning("✗ move_group no respondió en el timeout.")
        return ready


# ─────────────────────────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    print(f"\n{'─' * 58}")
    print(f"  PROBANDO (pipeline): {title}")
    print(f"{'─' * 58}")


def run_sequence(node: PipelineTestNode) -> None:

    node.wait_for_arm_server(timeout_sec=15.0)

    # ── 1. BASE — avance recto ────────────────────────────────────────────
    _banner("RUEDAS — avance recto")
    t0 = time.monotonic()
    while time.monotonic() - t0 < PHASE_DURATION:
        node.cmd_base(BASE_LINEAR_VEL, 0.0)
        time.sleep(0.05)
    node.stop_base()

    # ── 2. BASE — giro rápido ─────────────────────────────────────────────
    _banner("RUEDAS — giro en sitio (rápido)")
    t0 = time.monotonic()
    while time.monotonic() - t0 < PHASE_DURATION:
        node.cmd_base(0.0, BASE_ANGULAR_VEL)
        time.sleep(0.05)
    node.stop_base()

    # ── 3. FLIPPERS — oscilación sinusoidal ───────────────────────────────
    _banner("FLIPPERS — barrido sinusoidal via flipper_controller")
    t0 = time.monotonic()
    while time.monotonic() - t0 < PHASE_DURATION:
        t = time.monotonic() - t0
        angle = 1.2 * math.sin(2 * math.pi * t / PHASE_DURATION)
        node.cmd_flippers([angle, angle, angle, angle])
        time.sleep(0.05)

    # ── 4. FLIPPERS — nivel del suelo ─────────────────────────────────────
    _banner("FLIPPERS — bajando a nivel del suelo (0 rad)")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        node.cmd_flippers([0.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)

    # ── 5. BRAZO — posición HOME (via MoveIt) ─────────────────────────────
    _banner("BRAZO — ir a HOME (MoveIt planifica trayectoria)")
    node.send_arm_goal(NAMED_STATES["home"], vel_scale=0.5, label="home")

    # ── 6. BRAZO — onda de joints (deltas consecutivos, como el agente RL) ─
    _banner("BRAZO — onda de joints via MoveGroup (igual que el agente RL)")
    amplitudes = np.array([1.0, 0.8, 0.8, 0.6, 0.6, 0.4])
    base_pos   = np.array(NAMED_STATES["home"])
    N_STEPS    = 6
    for step in range(N_STEPS):
        t      = step / max(N_STEPS - 1, 1)
        target = base_pos + amplitudes * np.sin(np.pi * t + np.arange(6) * np.pi / 3)
        node.send_arm_goal(
            target.tolist(), vel_scale=0.7,
            label=f"wave step {step + 1}/{N_STEPS}",
        )

    # ── 7. BRAZO — posición ARM-UP (garra visible) ────────────────────────
    _banner("BRAZO — posición ARM-UP (garra visible, MoveIt)")
    node.send_arm_goal(NAMED_STATES["arm_up"], vel_scale=0.4, label="arm_up")

    # ── 8. BRAZO — posición EXTENSION (nombre del SRDF) ──────────────────
    _banner("BRAZO — posición EXTENSION (estado nombrado del SRDF)")
    node.send_arm_goal(NAMED_STATES["extension"], vel_scale=0.3, label="extension")

    print("\n" + "═" * 58)
    print("  ✓ Secuencia completa del pipeline finalizada.")
    print("  El brazo queda en posición EXTENSION.")
    print("═" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    rclpy.init()
    node = PipelineTestNode()

    # Spin en hilo daemon para que los callbacks de action client funcionen.
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name="spin"
    )
    spin_thread.start()

    try:
        run_sequence(node)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        node.stop_base()
        node.stop_flippers()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
