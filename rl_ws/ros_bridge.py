"""ros_bridge.py — Backend ROS 2 para el agente RL del robot Aesir.

Suscripciones (lo que lee del simulador):
    /joint_states          sensor_msgs/JointState
    /odom                  nav_msgs/Odometry
    /scan                  sensor_msgs/LaserScan
    /clock                 rosgraph_msgs/Clock

Publicaciones (lo que manda el agente):
    /diff_drive_controller/cmd_vel          geometry_msgs/Twist
        → base: velocidad lineal y angular
    /flipper_controller/commands            std_msgs/Float64MultiArray
        → posición de los 4 flippers (sin pasar por MoveIt)
    /servo_node/delta_joint_cmds            control_msgs/JointJog
        → brazo: velocidades articulares → MoveIt Servo → IK → ros2_control

El brazo SIEMPRE pasa por MoveIt Servo (send_arm_jog).
La base y flippers van directo a sus controllers (sin MoveIt).
"""
from __future__ import annotations

import threading
from typing import Optional, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64MultiArray, Header
from rosgraph_msgs.msg import Clock

# JointJog es el mensaje que MoveIt Servo acepta para comandos articulares
try:
    from control_msgs.msg import JointJog
    _HAS_JOINT_JOG = True
except ImportError:
    _HAS_JOINT_JOG = False


# ── QoS ────────────────────────────────────────────────────────────────────
def _sensor_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )

def _reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )

def _clock_qos() -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


# ── Node ───────────────────────────────────────────────────────────────────
class RosCommunicationNode(Node):
    """Nodo ROS 2 thread-safe que conecta el agente RL con MuJoCo + MoveIt."""

    DEFAULT_ARM_JOINTS = [
        "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_6",
    ]
    DEFAULT_FLIPPER_JOINTS = [
        "flipper_joint_1", "flipper_joint_2",
        "flipper_joint_3", "flipper_joint_4",
    ]

    def __init__(
        self,
        node_name: str = "rl_communication_bridge",
        arm_joint_names: Optional[List[str]] = None,
        flipper_joint_names: Optional[List[str]] = None,
        enable_scan: bool = True,
        use_moveit_servo: bool = True,   # False → publica directo al velocity controller
    ) -> None:
        super().__init__(node_name)

        self._lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._latest_odom:        Optional[Odometry]   = None
        self._latest_scan:        Optional[LaserScan]  = None
        self._sim_time_s: float = 0.0

        self._arm_joint_names     = list(arm_joint_names     or self.DEFAULT_ARM_JOINTS)
        self._flipper_joint_names = list(flipper_joint_names or self.DEFAULT_FLIPPER_JOINTS)
        self._use_moveit_servo    = use_moveit_servo and _HAS_JOINT_JOG

        if use_moveit_servo and not _HAS_JOINT_JOG:
            self.get_logger().warning(
                "control_msgs no disponible — el brazo se controlará "
                "directo al velocity controller (sin MoveIt Servo)."
            )

        # ── Publishers ─────────────────────────────────────────────────────

        # Base: cmd_vel → diff_drive_controller
        self._pub_base_cmd = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel", 10
        )

        # Flippers: posición directa → flipper_controller
        self._pub_flipper = self.create_publisher(
            Float64MultiArray,
            "/flipper_controller/commands", 10
        )

        # Brazo — dos modos:
        if self._use_moveit_servo:
            # ✅ Correcto para deployment con MoveIt:
            # Publica JointJog al servo_node, que hace IK y manda al velocity controller
            self._pub_arm_servo = self.create_publisher(
                JointJog,
                "/servo_node/delta_joint_cmds",
                _reliable_qos(10),
            )
            self._pub_arm_vel = None
            self.get_logger().info("Brazo: MoveIt Servo (/servo_node/delta_joint_cmds)")
        else:
            # Fallback: velocidades directas (sin IK, solo para pruebas)
            self._pub_arm_servo = None
            self._pub_arm_vel = self.create_publisher(
                Float64MultiArray,
                "/joint_group_velocity_controller/commands", 10
            )
            self.get_logger().info("Brazo: direct velocity controller (sin MoveIt)")

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, _sensor_qos()
        )
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, _sensor_qos()
        )
        if enable_scan:
            self.create_subscription(
                LaserScan, "/scan", self._scan_cb, _sensor_qos(5)
            )
        self.create_subscription(
            Clock, "/clock", self._clock_cb, _clock_qos()
        )

        self.get_logger().info(f"[{node_name}] bridge activo.")

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = msg

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._latest_scan = msg

    def _clock_cb(self, msg: Clock) -> None:
        with self._lock:
            self._sim_time_s = msg.clock.sec + msg.clock.nanosec * 1e-9

    # ── Getters ────────────────────────────────────────────────────────────
    def get_sim_time(self) -> float:
        with self._lock:
            return self._sim_time_s

    def get_joint_state(self) -> Optional[dict]:
        """Devuelve {position, velocity} del brazo ordenado por arm_joint_names."""
        with self._lock:
            msg = self._latest_joint_state
        if msg is None:
            return None
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = np.full(len(self._arm_joint_names), np.nan)
        vel = np.full(len(self._arm_joint_names), np.nan)
        for k, name in enumerate(self._arm_joint_names):
            idx = name_to_idx.get(name)
            if idx is not None:
                if idx < len(msg.position): pos[k] = msg.position[idx]
                if idx < len(msg.velocity): vel[k] = msg.velocity[idx]
        return {"position": pos, "velocity": vel}

    def get_odom(self) -> Optional[dict]:
        with self._lock:
            msg = self._latest_odom
        if msg is None:
            return None
        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        return {
            "position":    np.array([p.x, p.y, p.z]),
            "orientation": np.array([q.x, q.y, q.z, q.w]),
            "linear_vel":  np.array([lv.x, lv.y, lv.z]),
            "angular_vel": np.array([av.x, av.y, av.z]),
        }

    def get_scan(self, n_rays: Optional[int] = None,
                 max_range: float = 15.0) -> Optional[np.ndarray]:
        with self._lock:
            msg = self._latest_scan
        if msg is None:
            return None
        ranges = np.clip(
            np.where(np.isfinite(np.asarray(msg.ranges)), msg.ranges, max_range),
            0.0, max_range
        )
        if n_rays is not None and ranges.size != n_rays and ranges.size > 0:
            ranges = ranges[np.linspace(0, ranges.size - 1, n_rays).astype(int)]
        return ranges.astype(np.float64)

    # ── Publicadores de acción ──────────────────────────────────────────────

    def publish_base_cmd(self, v_lin: float, w_ang: float) -> None:
        """Publica velocidad lineal y angular de la base."""
        msg = Twist()
        msg.linear.x  = float(v_lin)
        msg.angular.z = float(w_ang)
        self._pub_base_cmd.publish(msg)

    def publish_flipper_positions(self, positions_4: np.ndarray) -> None:
        """Publica posición objetivo de los 4 flippers [rad]."""
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions_4]
        self._pub_flipper.publish(msg)

    def send_arm_jog(self, vel_6d, frame_id: str = "arm_base_link") -> None:
        """
        Publica velocidades articulares del brazo [rad/s].

        Si use_moveit_servo=True (default):
            → publica JointJog a /servo_node/delta_joint_cmds
            → MoveIt Servo hace la cinemática y manda al ros2_control
        Si use_moveit_servo=False:
            → publica Float64MultiArray directo al velocity controller
        """
        vel = np.asarray(vel_6d, dtype=np.float64).reshape(-1)
        if vel.size != len(self._arm_joint_names):
            raise ValueError(
                f"send_arm_jog: esperaba {len(self._arm_joint_names)} valores, "
                f"recibió {vel.size}"
            )

        if self._use_moveit_servo and self._pub_arm_servo is not None:
            msg = JointJog()
            msg.header = Header()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            msg.joint_names     = list(self._arm_joint_names)
            msg.velocities      = [float(v) for v in vel]
            msg.duration        = 0.1   # segundos de duración del comando
            self._pub_arm_servo.publish(msg)
        elif self._pub_arm_vel is not None:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in vel]
            self._pub_arm_vel.publish(msg)

    # Alias del nombre anterior para no romper código que lo llame
    def send_joint_velocity(self, vel_6d) -> None:
        self.send_arm_jog(vel_6d)

    def stop_robot(self) -> None:
        """Para base, brazo y flippers."""
        self.publish_base_cmd(0.0, 0.0)
        self.send_arm_jog([0.0] * len(self._arm_joint_names))
        self.publish_flipper_positions(np.zeros(4))
