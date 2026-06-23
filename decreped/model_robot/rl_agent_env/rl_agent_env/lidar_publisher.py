"""Nodo ROS 2 publicador de LiDAR para el robot Aesir.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan


# Velocidad angular del lidar_spin [rad/s] — igual que sensors.py (20 rad/s)
LIDAR_SPIN_VEL = 20.0
PUBLISH_HZ     = 30.0
CUTOFF_M       = 15.0   # distancia máxima de los rangefinders


class LidarPublisherNode(Node):
    """Instancia MuJoCo sombra que publica /scan desde los rangefinders."""

    def __init__(self) -> None:
        super().__init__("lidar_publisher")

        self.declare_parameter("scene_xml", "")
        xml_path = self.get_parameter("scene_xml").value
        if not xml_path:
            raise RuntimeError("Parámetro 'scene_xml' vacío. Pásalo desde el launch file.")

        self.get_logger().info(f"[lidar_publisher] Cargando escena: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)

        # Índice qpos del joint lidar_spin
        self._lidar_spin_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lidar_spin"
        )
        if self._lidar_spin_jid < 0:
            self.get_logger().warning("Joint 'lidar_spin' no encontrado en el modelo.")
        self._lidar_spin_qpos_adr = (
            int(self.model.jnt_qposadr[self._lidar_spin_jid])
            if self._lidar_spin_jid >= 0 else -1
        )

        # Índices sensordata para lidar_0..lidar_6
        self._sensor_addrs: list[int] = []
        for i in range(7):
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}"
            )
            if sid < 0:
                self.get_logger().warning(f"Sensor 'lidar_{i}' no encontrado.")
                self._sensor_addrs.append(-1)
            else:
                self._sensor_addrs.append(int(self.model.sensor_adr[sid]))
        self.get_logger().info(
            f"[lidar_publisher] sensor_adr lidar_0..6 = {self._sensor_addrs}"
        )

        # Mapa joint_name → qpos_adr (para actualizar desde /joint_states)
        self._joint_qpos: dict[str, int] = {}
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                self._joint_qpos[name] = int(self.model.jnt_qposadr[jid])

        # Índice qpos del freejoint (base del robot)
        self._base_qpos_adr = -1
        base_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint"
        )
        if base_jid >= 0:
            self._base_qpos_adr = int(self.model.jnt_qposadr[base_jid])

        # Ángulo interno del lidar_spin (rad), avanza con el reloj de pared
        self._spin_angle: float = 0.0
        self._last_spin_t: float = time.monotonic()

        # Ángulos de elevación de los 7 rayos [rad] (euler Y en el XML)
        ray_elevations = [0.0, 0.2793, 0.5585, 0.8378, 1.1170, 1.3963, 1.6755]

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Suscriptores
        self.create_subscription(JointState, "/joint_states", self._joint_cb, sensor_qos)
        self.create_subscription(Odometry,   "/odom",         self._odom_cb,  sensor_qos)

        # Publicador
        self._pub = self.create_publisher(LaserScan, "/scan", sensor_qos)

        # Mensaje LaserScan plantilla (campos fijos)
        self._scan_msg = LaserScan()
        self._scan_msg.header.frame_id = "sensor_lidar"
        self._scan_msg.angle_min       = float(ray_elevations[0])
        self._scan_msg.angle_max       = float(ray_elevations[-1])
        self._scan_msg.angle_increment = float(ray_elevations[1] - ray_elevations[0])
        self._scan_msg.time_increment  = 0.0
        self._scan_msg.scan_time       = 1.0 / PUBLISH_HZ
        self._scan_msg.range_min       = 0.01
        self._scan_msg.range_max       = CUTOFF_M

        self.create_timer(1.0 / PUBLISH_HZ, self._publish_scan)
        self.get_logger().info("[lidar_publisher] Listo.")

    # ------------------------------------------------------------------ #
    #                        Callbacks de suscripción                    #
    # ------------------------------------------------------------------ #
    def _joint_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            adr = self._joint_qpos.get(name)
            if adr is not None:
                self.data.qpos[adr] = float(pos)

    def _odom_cb(self, msg: Odometry) -> None:
        if self._base_qpos_adr < 0:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # MuJoCo freejoint: [x, y, z, qw, qx, qy, qz]
        # ROS odometry quaternion: [qx, qy, qz, qw]
        adr = self._base_qpos_adr
        self.data.qpos[adr + 0] = float(p.x)
        self.data.qpos[adr + 1] = float(p.y)
        self.data.qpos[adr + 2] = 0.10          # terreno plano, altura fija
        self.data.qpos[adr + 3] = float(q.w)    # qw
        self.data.qpos[adr + 4] = float(q.x)    # qx
        self.data.qpos[adr + 5] = float(q.y)    # qy
        self.data.qpos[adr + 6] = float(q.z)    # qz

    # ------------------------------------------------------------------ #
    #                        Timer: publicar /scan                       #
    # ------------------------------------------------------------------ #
    def _publish_scan(self) -> None:
        # Avanzar ángulo de spin con tiempo de pared
        now = time.monotonic()
        dt  = now - self._last_spin_t
        self._last_spin_t = now
        self._spin_angle += LIDAR_SPIN_VEL * dt
        self._spin_angle %= (2.0 * math.pi)

        if self._lidar_spin_qpos_adr >= 0:
            self.data.qpos[self._lidar_spin_qpos_adr] = self._spin_angle

        # Actualizar cinemática y sensores (SIN avanzar física)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_sensor(self.model, self.data)

        # Leer los 7 rangefinders
        ranges: list[float] = []
        for adr in self._sensor_addrs:
            if adr < 0:
                ranges.append(CUTOFF_M)
                continue
            val = float(self.data.sensordata[adr])
            # MuJoCo devuelve -1 si no hay impacto; mapear a CUTOFF
            ranges.append(val if 0.0 < val < CUTOFF_M else CUTOFF_M)

        msg = self._scan_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ranges = ranges
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = LidarPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
