#!/usr/bin/env python3
"""
Aesir pipeline diagnostic — live dashboard + scripted commands.

Shows the full signal chain end-to-end:

    Python action
        ↓  /servo_node/delta_twist_cmds  (TwistStamped)
    MoveIt Servo
        ↓  /joint_group_velocity_controller/commands  (Float64MultiArray)
    ros2_control / MuJoCo
        ↓  /joint_states  (JointState)  +  /odom  (Odometry)
    ros_bridge (rl_env)

Modes (--mode):
  monitor  — only subscribe and display, send nothing.
  zero     — publish zeros to arm + base every cycle (confirms controllers
             are alive and do NOT drift).
  step     — constant TwistStamped (vx=STEP_VEL) for ARM_AXIS; constant
             v_lin for base.  Good for a quick open-loop sanity check.
  sweep    — sinusoidal sweep of all 6 Cartesian DOFs sequentially.
             Each DOF is exercised for SWEEP_PERIOD seconds then moves to
             the next.  Use this to map the Servo → MuJoCo latency.
  base     — drives the base only (linear + angular) in a square pattern.

Usage:
    # Build first:
    #   colcon build --packages-select robot_moveit_config
    #   source install/setup.bash

    ros2 run robot_moveit_config diagnose_pipeline.py
    ros2 run robot_moveit_config diagnose_pipeline.py --ros-args -p mode:=sweep
    ros2 run robot_moveit_config diagnose_pipeline.py --ros-args -p mode:=step -p step_vel:=0.05
    ros2 run robot_moveit_config diagnose_pipeline.py --ros-args -p mode:=base

Keyboard while running:
    CTRL-C  — clean shutdown
"""

from __future__ import annotations

import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int8

# Servo status integer -> label (moveit_servo StatusCode enum)
_SERVO_STATUS = {
    0: "OK          ",
    1: "SLOW_singular",
    2: "HALT_singular",
    3: "SLOW_leave_sing",
    4: "HALT_leave_sing",
    5: "SLOW_collision",
    6: "HALT_collision",
    7: "JOINT_BOUND ",
}

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# ANSI helpers
_CLR  = "\033[2J\033[H"   # clear screen + cursor home
_BOLD = "\033[1m"
_RST  = "\033[0m"
_GRN  = "\033[32m"
_YLW  = "\033[33m"
_RED  = "\033[31m"
_CYN  = "\033[36m"


def _fmt_f(v: float, width: int = 7, prec: int = 3) -> str:
    return f"{v:+{width}.{prec}f}"


class PipelineDiagnostic(Node):
    """Single node that publishes commands and displays the full pipeline."""

    TIMER_HZ = 10.0   # publish + render rate

    def __init__(self) -> None:
        super().__init__("aesir_pipeline_diagnostic")

        # ---- Parameters -------------------------------------------------- #
        self.declare_parameter("mode",       "monitor")
        self.declare_parameter("step_vel",   0.05)    # m/s or rad/s for step mode
        self.declare_parameter("sweep_amp",  0.10)    # amplitude for sine sweep (m/s)
        self.declare_parameter("sweep_period", 6.0)   # seconds per DOF in sweep
        self.declare_parameter("base_lin",   0.15)    # m/s linear for base test
        self.declare_parameter("base_ang",   0.3)     # rad/s angular for base test

        self.mode        = self.get_parameter("mode").value
        self.step_vel    = self.get_parameter("step_vel").value
        self.sweep_amp   = self.get_parameter("sweep_amp").value
        self.sweep_period = self.get_parameter("sweep_period").value
        self.base_lin    = self.get_parameter("base_lin").value
        self.base_ang    = self.get_parameter("base_ang").value

        # ---- Publishers -------------------------------------------------- #
        self._pub_arm = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
        self._pub_base = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel", 10
        )

        # ---- Subscribers (buffered, thread-safe via rclpy default mutex) - #
        self._js:   Optional[JointState]         = None
        self._odom: Optional[Odometry]           = None
        self._servo_out: Optional[Float64MultiArray] = None
        self._servo_status: int                  = -1
        self._sim_time: float                    = 0.0

        self.create_subscription(JointState, "/joint_states",
                                 self._cb_js,   10)
        self.create_subscription(Odometry, "/odom",
                                 self._cb_odom, 10)
        self.create_subscription(
            Float64MultiArray,
            "/joint_group_velocity_controller/commands",
            self._cb_servo_out, 10,
        )
        self.create_subscription(Int8, "/servo_node/status",
                                 self._cb_servo_status, 10)

        # ---- State for scripted patterns ---------------------------------- #
        self._t0           = time.monotonic()
        self._last_cmd_arm = TwistStamped()
        self._last_cmd_base = Twist()
        self._sweep_dof    = 0    # which of the 6 Cartesian DOFs is active
        self._base_phase   = 0    # square-wave phase counter for base test

        # ---- Main timer -------------------------------------------------- #
        self.create_timer(1.0 / self.TIMER_HZ, self._tick)

        self.get_logger().info(
            f"Pipeline diagnostic started — mode='{self.mode}'  "
            f"(CTRL-C to quit)"
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #
    def _cb_js(self, msg: JointState)              -> None: self._js   = msg
    def _cb_odom(self, msg: Odometry)              -> None: self._odom = msg
    def _cb_servo_out(self, msg: Float64MultiArray)-> None: self._servo_out   = msg
    def _cb_servo_status(self, msg: Int8)          -> None: self._servo_status = msg.data

    # ------------------------------------------------------------------ #
    #  Command generation                                                 #
    # ------------------------------------------------------------------ #
    def _build_arm_twist(self, t: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base_link"

        lx = ly = lz = ax = ay = az = 0.0

        if self.mode == "zero":
            pass

        elif self.mode == "step":
            lx = self.step_vel

        elif self.mode == "sweep":
            # Cycle through 6 DOFs: lx ly lz ax ay az
            dof_idx = self._sweep_dof % 6
            amp     = self.sweep_amp * math.sin(2 * math.pi * t / self.sweep_period)
            dofs    = [0.0] * 6
            dofs[dof_idx] = amp
            lx, ly, lz, ax, ay, az = dofs
            # Advance DOF every sweep_period seconds
            if t > (self._sweep_dof + 1) * self.sweep_period:
                self._sweep_dof += 1

        # base mode: arm stays at zero
        msg.twist.linear.x  = lx
        msg.twist.linear.y  = ly
        msg.twist.linear.z  = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        return msg

    def _build_base_twist(self, t: float) -> Twist:
        msg = Twist()

        if self.mode in ("zero", "sweep", "step"):
            pass  # base stays at zero unless mode=base

        elif self.mode == "base":
            # Simple square pattern: fwd → turn → fwd → turn ...
            phase_dur = 3.0   # seconds per phase
            phase     = int(t / phase_dur) % 4
            if phase in (0, 2):   # straight
                msg.linear.x  =  self.base_lin
                msg.angular.z =  0.0
            else:                 # turn
                msg.linear.x  =  0.0
                msg.angular.z =  self.base_ang if phase == 1 else -self.base_ang

        return msg

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #
    def _render(self, t: float) -> None:
        arm_cmd = self._last_cmd_arm
        base_cmd = self._last_cmd_base

        lines = [
            f"{_CLR}{_BOLD}{'━'*62}",
            f"  Aesir Pipeline Diagnostic — mode: {_CYN}{self.mode}{_RST}{_BOLD}",
            f"{'━'*62}{_RST}",
            f"  Elapsed: {t:7.2f} s    "
            f"Servo status: {self._servo_status_str()}",
            "",
            f"{_BOLD}── 1. SENT → /servo_node/delta_twist_cmds ──────────────{_RST}",
            f"     linear  x={_fmt_f(arm_cmd.twist.linear.x)} "
            f"y={_fmt_f(arm_cmd.twist.linear.y)} "
            f"z={_fmt_f(arm_cmd.twist.linear.z)}",
            f"     angular x={_fmt_f(arm_cmd.twist.angular.x)} "
            f"y={_fmt_f(arm_cmd.twist.angular.y)} "
            f"z={_fmt_f(arm_cmd.twist.angular.z)}",
        ]

        if self.mode in ("base", "zero"):
            lines += [
                "",
                f"{_BOLD}── 2. SENT → /diff_drive_controller/cmd_vel ────────────{_RST}",
                f"     linear.x={_fmt_f(base_cmd.linear.x)}   "
                f"angular.z={_fmt_f(base_cmd.angular.z)}",
            ]

        lines += [
            "",
            f"{_BOLD}── 3. MoveIt Servo OUT → /joint_group_velocity_controller/commands {_RST}",
        ]
        if self._servo_out is not None and self._servo_out.data:
            vals = self._servo_out.data
            row  = "  ".join(
                f"{_GRN}{j}: {_fmt_f(v, 6, 3)}{_RST}"
                for j, v in zip(ARM_JOINTS, vals)
            )
            lines.append(f"     {row}")
        else:
            lines.append(f"     {_YLW}(no data yet){_RST}")

        lines += [
            "",
            f"{_BOLD}── 4. MuJoCo → /joint_states ───────────────────────────{_RST}",
        ]
        if self._js is not None:
            js   = self._js
            # Build index map for robustness (order not guaranteed by broadcaster)
            idx  = {n: i for i, n in enumerate(js.name)}
            pos_str = "  ".join(
                f"{j[6:]}: {_fmt_f(js.position[idx[j]], 6, 3)}"
                if j in idx else f"{j[6:]}: {_YLW}n/a{_RST}"
                for j in ARM_JOINTS
            )
            vel_str = "  ".join(
                f"{j[6:]}: {_fmt_f(js.velocity[idx[j]], 6, 3)}"
                if j in idx and js.velocity else f"{j[6:]}: {_YLW}n/a{_RST}"
                for j in ARM_JOINTS
            )
            lines.append(f"     pos: {pos_str}")
            lines.append(f"     vel: {vel_str}")
        else:
            lines.append(f"     {_YLW}(no data yet — is MuJoCo running?){_RST}")

        lines += [
            "",
            f"{_BOLD}── 5. MuJoCo → /odom ───────────────────────────────────{_RST}",
        ]
        if self._odom is not None:
            p = self._odom.pose.pose.position
            t_lin = self._odom.twist.twist.linear
            t_ang = self._odom.twist.twist.angular
            lines.append(
                f"     pos: x={_fmt_f(p.x)} y={_fmt_f(p.y)} z={_fmt_f(p.z)}"
            )
            lines.append(
                f"     vel: vx={_fmt_f(t_lin.x)} vy={_fmt_f(t_lin.y)}"
                f"   wz={_fmt_f(t_ang.z)}"
            )
        else:
            lines.append(
                f"     {_YLW}(no data yet — is diff_drive_controller active?){_RST}"
            )

        lines.append(f"\n{_BOLD}  CTRL-C to quit{_RST}")
        print("\n".join(lines), flush=True)

    def _servo_status_str(self) -> str:
        s = self._servo_status
        label = _SERVO_STATUS.get(s, f"unknown ({s})")
        if s == 0:
            return f"{_GRN}{label}{_RST}"
        elif s == -1:
            return f"{_YLW}(no status topic yet){_RST}"
        else:
            return f"{_RED}{label}{_RST}"

    # ------------------------------------------------------------------ #
    #  Main timer callback                                                #
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        t   = time.monotonic() - self._t0

        if self.mode != "monitor":
            arm_msg  = self._build_arm_twist(t)
            base_msg = self._build_base_twist(t)
            self._pub_arm.publish(arm_msg)
            self._pub_base.publish(base_msg)
            self._last_cmd_arm  = arm_msg
            self._last_cmd_base = base_msg

        self._render(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send a zero command on exit so the robot doesn't keep moving.
        try:
            stop_arm = TwistStamped()
            stop_arm.header.stamp = node.get_clock().now().to_msg()
            node._pub_arm.publish(stop_arm)
            node._pub_base.publish(Twist())
            time.sleep(0.1)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
