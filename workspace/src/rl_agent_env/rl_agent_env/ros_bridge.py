"""Backend ROS 2 asíncrono para el ambiente RL del robot Aesir.

Suscripciones
    /joint_states          sensor_msgs/JointState   — brazo + flippers
    /odom                  nav_msgs/Odometry        — pose + twist de la base
    /scan                  sensor_msgs/LaserScan    — LiDAR rotatorio (lidar_publisher)
    /clock                 rosgraph_msgs/Clock      — tiempo de simulación MuJoCo

Publicaciones
    /diff_drive_controller/cmd_vel          geometry_msgs/Twist
    /joint_group_velocity_controller/commands  std_msgs/Float64MultiArray
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock


# --------------------------------------------------------------------------- #
#                              QoS presets                                    #
# --------------------------------------------------------------------------- #
def _sensor_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
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


# --------------------------------------------------------------------------- #
#                          RosCommunicationNode                               #
# --------------------------------------------------------------------------- #
class RosCommunicationNode(Node):
    """Nodo ROS 2 thread-safe que une el simulador con el trainer PPO.

    Parámetros
    ----------
    node_name : str
    arm_joint_names : list[str], opcional
        Filtro y orden para get_joint_state(). Si es None usa DEFAULT_ARM_JOINTS.
    enable_scan : bool
        Suscribirse a /scan y exponer datos por get_scan().
    """

    DEFAULT_ARM_JOINTS = [
        "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_6",
    ]

    def __init__(
        self,
        node_name: str = "rl_communication_bridge",
        arm_joint_names: Optional[list] = None,
        enable_scan: bool = True,
    ) -> None:
        super().__init__(node_name)

        self._lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._latest_odom: Optional[Odometry]          = None
        self._latest_scan: Optional[LaserScan]         = None
        self._sim_time_s: float                        = 0.0

        self._arm_joint_names = list(arm_joint_names or self.DEFAULT_ARM_JOINTS)

        # ---- Publishers --------------------------------------------------- #
        self._pub_base_cmd = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel", 10
        )
        # Control de velocidad del brazo — directo al JointGroupVelocityController
        self._pub_arm_vel = self.create_publisher(
            Float64MultiArray,
            "/joint_group_velocity_controller/commands", 10
        )

        # ---- Subscribers -------------------------------------------------- #
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

        self.get_logger().info(
            f"[{node_name}] bridge activo. Joints: {self._arm_joint_names}"
        )

    # ------------------------------------------------------------------ #
    #                       Callbacks de suscripción                     #
    # ------------------------------------------------------------------ #
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
        secs = msg.clock.sec + msg.clock.nanosec * 1e-9
        with self._lock:
            self._sim_time_s = secs

    # ------------------------------------------------------------------ #
    #                  Accessores thread-safe (trainer)                  #
    # ------------------------------------------------------------------ #
    def get_sim_time(self) -> float:
        with self._lock:
            return self._sim_time_s

    def get_joint_state(self) -> Optional[dict]:
        """Devuelve {"position": ndarray, "velocity": ndarray} ordenado por arm_joint_names."""
        with self._lock:
            msg = self._latest_joint_state
        if msg is None:
            return None

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = np.full(len(self._arm_joint_names), np.nan, dtype=np.float64)
        vel = np.full(len(self._arm_joint_names), np.nan, dtype=np.float64)
        for k, name in enumerate(self._arm_joint_names):
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            if idx < len(msg.position):
                pos[k] = msg.position[idx]
            if idx < len(msg.velocity):
                vel[k] = msg.velocity[idx]
        return {"position": pos, "velocity": vel}

    def get_odom(self) -> Optional[dict]:
        """Devuelve pose+twist de la base como arrays numpy."""
        with self._lock:
            msg = self._latest_odom
        if msg is None:
            return None

        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        return {
            "position":    np.array([p.x, p.y, p.z],      dtype=np.float64),
            "orientation": np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
            "linear_vel":  np.array([lv.x, lv.y, lv.z],   dtype=np.float64),
            "angular_vel": np.array([av.x, av.y, av.z],    dtype=np.float64),
        }

    def get_scan(self, n_rays: Optional[int] = None,
                 max_range: float = 15.0) -> Optional[np.ndarray]:
        """Devuelve rangos del LiDAR como ndarray.

        Si n_rays difiere del tamaño del scan, submuestrea uniformemente.
        """
        with self._lock:
            msg = self._latest_scan
        if msg is None:
            return None

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ranges = np.where(np.isfinite(ranges), ranges, max_range)
        ranges = np.clip(ranges, 0.0, max_range)

        if n_rays is not None and ranges.size != n_rays and ranges.size > 0:
            idx    = np.linspace(0, ranges.size - 1, n_rays).astype(np.int64)
            ranges = ranges[idx]
        return ranges

    # ------------------------------------------------------------------ #
    #               Publicación de acciones (trainer)                    #
    # ------------------------------------------------------------------ #
    def send_joint_velocity(self, vel_6d) -> None:
        """Publica velocidades de joints [rad/s] al JointGroupVelocityController.

        Parámetros
        ----------
        vel_6d : array-like de 6 valores en rad/s, ordenados como arm_joint_names.
        """
        vel = np.asarray(vel_6d, dtype=np.float64).reshape(-1)
        if vel.size != len(self._arm_joint_names):
            raise ValueError(
                f"send_joint_velocity: esperaba {len(self._arm_joint_names)} "
                f"valores, recibió {vel.size}"
            )
        msg      = Float64MultiArray()
        msg.data = [float(v) for v in vel]
        self._pub_arm_vel.publish(msg)

    def publish_base_cmd(self, v_lin: float, w_ang: float) -> None:
        """Publica Twist planar en /diff_drive_controller/cmd_vel."""
        msg           = Twist()
        msg.linear.x  = float(v_lin)
        msg.angular.z = float(w_ang)
        self._pub_base_cmd.publish(msg)

    def stop_robot(self) -> None:
        """Para base y brazo. Llamar en reset y shutdown."""
        self.publish_base_cmd(0.0, 0.0)
        self.send_joint_velocity([0.0] * len(self._arm_joint_names))
