"""Asynchronous ROS 2 backend for the RL environment.

This module owns every rclpy primitive (node, pubs, subs, clock). It is meant
to run inside a *background thread* spun by :class:`rl_agent_env.rl_env.Env`.
The PPO trainer (PyTorch) lives in the main thread and only ever calls the
thread-safe accessors exposed here -- never touches the rclpy executor.

Topics
------
Subscriptions
    /joint_states          sensor_msgs/JointState   - arm + flipper feedback
    /odom                  nav_msgs/Odometry        - base pose & twist
    /scan                  sensor_msgs/LaserScan    - optional 2-D LiDAR
    /clock                 rosgraph_msgs/Clock      - MuJoCo sim time

Publications
    /servo_node/delta_twist_cmds   geometry_msgs/TwistStamped   - arm EE delta
    /diff_drive_controller/cmd_vel geometry_msgs/Twist          - base cmd
"""
from __future__ import annotations

import math
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

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from rosgraph_msgs.msg import Clock


# --------------------------------------------------------------------------- #
#                              QoS presets                                    #
# --------------------------------------------------------------------------- #
def _sensor_qos(depth: int = 10) -> QoSProfile:
    """Best-effort sensor stream (joint_states, odom, scan)."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _clock_qos() -> QoSProfile:
    """ROS 2 standard /clock QoS: reliable + transient_local + depth 1."""
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
    """Thread-safe ROS 2 node bridging the sim with the PPO trainer.

    All shared buffers are guarded by an internal ``threading.Lock`` so the
    PPO main thread (calling ``get_*`` from :class:`Env.step`) and the
    rclpy executor thread (calling the subscriber callbacks) can interact
    without races.

    Parameters
    ----------
    node_name : str
        ROS node name. Defaults to ``"rl_communication_bridge"``.
    arm_joint_names : list[str], optional
        Filter applied to JointState messages. When provided, ``get_joint_state``
        returns positions/velocities ordered by this list. When ``None``, the
        most recent message is returned as-is.
    enable_scan : bool
        If True, subscribe to ``/scan`` and expose ranges through
        :meth:`get_scan`.
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

        # ---- shared state ------------------------------------------------- #
        self._lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_scan: Optional[LaserScan] = None
        self._sim_time_s: float = 0.0  # seconds (from /clock)

        # Track joint ordering for repeatable observations.
        self._arm_joint_names = list(arm_joint_names or self.DEFAULT_ARM_JOINTS)

        # ---- publishers --------------------------------------------------- #
        self._pub_arm_twist = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
        self._pub_base_cmd = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel", 10
        )

        # ---- subscribers -------------------------------------------------- #
        self.create_subscription(
            JointState, "/joint_states",
            self._joint_state_cb, _sensor_qos(),
        )
        self.create_subscription(
            Odometry, "/odom",
            self._odom_cb, _sensor_qos(),
        )
        if enable_scan:
            self.create_subscription(
                LaserScan, "/scan",
                self._scan_cb, _sensor_qos(5),
            )
        self.create_subscription(
            Clock, "/clock",
            self._clock_cb, _clock_qos(),
        )

        self.get_logger().info(
            f"[{node_name}] bridge up. Tracking joints: {self._arm_joint_names}"
        )

    # ------------------------------------------------------------------ #
    #                       Subscriber callbacks                         #
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
        # Convert builtin_interfaces/Time to float seconds.
        secs = msg.clock.sec + msg.clock.nanosec * 1e-9
        with self._lock:
            self._sim_time_s = secs

    # ------------------------------------------------------------------ #
    #                Thread-safe accessors (trainer-facing)              #
    # ------------------------------------------------------------------ #
    def get_sim_time(self) -> float:
        """Return the latest /clock value in seconds."""
        with self._lock:
            return self._sim_time_s

    def get_joint_state(self) -> Optional[dict]:
        """Return ``{"position": np.ndarray, "velocity": np.ndarray}``.

        Reordered according to ``arm_joint_names`` when possible.
        Missing joints are filled with NaN (so the trainer can tell something
        was unobserved).
        """
        with self._lock:
            msg = self._latest_joint_state
        if msg is None:
            return None

        if self._arm_joint_names:
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

        return {
            "position": np.asarray(msg.position, dtype=np.float64),
            "velocity": np.asarray(msg.velocity, dtype=np.float64),
        }

    def get_odom(self) -> Optional[dict]:
        """Return base pose+twist as a flat dict of numpy arrays."""
        with self._lock:
            msg = self._latest_odom
        if msg is None:
            return None

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        return {
            "position":    np.array([p.x, p.y, p.z],          dtype=np.float64),
            "orientation": np.array([q.x, q.y, q.z, q.w],     dtype=np.float64),
            "linear_vel":  np.array([lv.x, lv.y, lv.z],       dtype=np.float64),
            "angular_vel": np.array([av.x, av.y, av.z],       dtype=np.float64),
        }

    def get_scan(self, n_rays: Optional[int] = None,
                 max_range: float = 10.0) -> Optional[np.ndarray]:
        """Return LaserScan ranges as a numpy array.

        If ``n_rays`` is given, the raw scan is uniformly subsampled to that
        size (useful to keep the obs vector at a fixed length).
        """
        with self._lock:
            msg = self._latest_scan
        if msg is None:
            return None

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        # Replace inf / NaN with max_range, clip to valid window.
        ranges = np.where(np.isfinite(ranges), ranges, max_range)
        ranges = np.clip(ranges, 0.0, max_range)

        if n_rays is not None and ranges.size != n_rays and ranges.size > 0:
            idx = np.linspace(0, ranges.size - 1, n_rays).astype(np.int64)
            ranges = ranges[idx]
        return ranges

    # ------------------------------------------------------------------ #
    #                  Action publishing (trainer-facing)                #
    # ------------------------------------------------------------------ #
    def publish_arm_twist(self, twist_6d, frame_id: str = "arm_base_link") -> None:
        """Publish a 6-D Cartesian delta to MoveIt Servo.

        Parameters
        ----------
        twist_6d : Sequence[float]
            ``[vx, vy, vz, wx, wy, wz]`` in the planning frame.
        frame_id : str
            Frame attached to the TwistStamped header. Must match
            ``robot_link_command_frame`` in servo_params.yaml.
        """
        if len(twist_6d) != 6:
            raise ValueError(
                f"publish_arm_twist expects 6 components, got {len(twist_6d)}"
            )

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.twist.linear.x  = float(twist_6d[0])
        msg.twist.linear.y  = float(twist_6d[1])
        msg.twist.linear.z  = float(twist_6d[2])
        msg.twist.angular.x = float(twist_6d[3])
        msg.twist.angular.y = float(twist_6d[4])
        msg.twist.angular.z = float(twist_6d[5])
        self._pub_arm_twist.publish(msg)

    def publish_base_cmd(self, v_lin: float, w_ang: float) -> None:
        """Publish a planar Twist on /diff_drive_controller/cmd_vel."""
        msg = Twist()
        msg.linear.x  = float(v_lin)
        msg.angular.z = float(w_ang)
        self._pub_base_cmd.publish(msg)

    def stop_robot(self) -> None:
        """Convenience: zero out base and arm commands. Useful on reset/shutdown."""
        self.publish_base_cmd(0.0, 0.0)
        self.publish_arm_twist([0.0] * 6)
