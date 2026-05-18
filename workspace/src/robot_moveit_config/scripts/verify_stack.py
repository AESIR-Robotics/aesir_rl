#!/usr/bin/env python3
"""
verify_stack.py — Integration verification for the Aesir dual-agent RL stack.

Tests three modules in order.  Each module builds on the previous one.

──────────────────────────────────────────────────────────────────────────────
MODULE 1 — MoveIt alive
  1.1  /controller_manager/list_controllers service responds
  1.2  joint_group_velocity_controller      → state: active
  1.3  diff_drive_controller                → state: active
  1.4  flipper_controller                   → state: active
  1.5  joint_state_broadcaster              → state: active
  1.6  move_group action server             → reachable within timeout
       (skip with --skip-plan; we don't launch move_group)

MODULE 2 — MoveIt ↔ MuJoCo connection (data flows both ways)
  2.1  /joint_states published  → arm joints (joint_1..6) present
  2.2  /joint_states published  → flipper joints (flipper_joint_*) present
  2.3  /odom published          → diff_drive_controller is forwarding MuJoCo odometry
  2.4  Send arm velocity cmd    → joint_1 velocity updates in /joint_states
  2.5  Send base twist cmd      → odom linear velocity updates
  2.6  MoveGroup plan to HOME   → plan succeeds (plan_only=True, no motion executed)

MODULE 3 — Agent A / Agent B actuations (exact paths used by train_ppo.py)
  3.1  [Agent A — arm]       /joint_group_velocity_controller/commands → joints move
  3.2  [Agent A — grip]      pos_left/right_finger absent from /joint_states
                             (MuJoCo-direct: not exported via ros2_control)
  3.3  [Agent B — base]      /diff_drive_controller/cmd_vel → /odom vx ≠ 0
  3.4  [Agent B — flippers]  /flipper_controller/commands  → flipper pos changes
  3.5  [Agent B — flip-whls] vel_flip* absent from /joint_states
                             (MuJoCo-direct: not exported via ros2_control)
──────────────────────────────────────────────────────────────────────────────

PREREQUISITE (run in separate terminal BEFORE this script):
    ros2 launch robot_moveit_config bringup.launch.py

Usage:
    cd /home/<user>/aesir_rl/workspace
    source install/setup.bash

    # Full verification (all 3 modules)
    python3 src/robot_moveit_config/scripts/verify_stack.py

    # Only specific module(s):
    python3 src/robot_moveit_config/scripts/verify_stack.py --module 1
    python3 src/robot_moveit_config/scripts/verify_stack.py --module 2
    python3 src/robot_moveit_config/scripts/verify_stack.py --module 3

    # Skip the MoveGroup plan test (no MoveIt planning pipeline needed):
    python3 src/robot_moveit_config/scripts/verify_stack.py --skip-plan
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint


# ─────────────────────────── Constants ─────────────────────────────────────
ARM_JOINTS     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
FLIPPER_JOINTS = ["flipper_joint_1", "flipper_joint_2",
                  "flipper_joint_3", "flipper_joint_4"]

# Joints that must NOT appear in /joint_states (MuJoCo-direct only)
MUJOCO_DIRECT_JOINTS = [
    "pos_left_finger", "pos_right_finger",     # Agent A grip
    "vel_flip1_back", "vel_flip1_front",        # Agent B flipper wheels
    "vel_flip2_back", "vel_flip2_front",
    "vel_flip3_back", "vel_flip3_front",
    "vel_flip4_back", "vel_flip4_front",
]

# Controllers we actually launch (train_agents.launch.py). We do NOT spawn
# `arm_controller` (JointTrajectoryController for move_group's
# FollowJointTrajectory action) because MoveIt Servo claims joint_1..6 via
# `joint_group_velocity_controller`, and a joint can only be commanded by one
# active controller at a time.
EXPECTED_CONTROLLERS = [
    "joint_group_velocity_controller",
    "diff_drive_controller",
    "flipper_controller",
    "joint_state_broadcaster",
]

HOME_POSITION  = [0.0, -2.8625, 2.8625, -1.5708, -1.5708, 0.0]
ARM_GROUP      = "arm"

TOPIC_WAIT_S      = 6.0    # seconds to wait for a topic to appear
CMD_SETTLE_S      = 1.2    # seconds to wait after sending a command
SERVICE_TIMEOUT_S = 8.0
MOVEIT_TIMEOUT_S  = 15.0
PLAN_TIMEOUT_S    = 20.0

# ANSI colour helpers
_GRN  = "\033[32m"
_RED  = "\033[31m"
_YLW  = "\033[33m"
_CYN  = "\033[36m"
_BOLD = "\033[1m"
_RST  = "\033[0m"


# ─────────────────────────── Result collector ───────────────────────────────
class Results:
    def __init__(self):
        self._items: List[Tuple[str, bool, str]] = []

    def add(self, test_id: str, passed: bool, detail: str = "") -> bool:
        self._items.append((test_id, passed, detail))
        icon  = f"{_GRN}✓{_RST}" if passed else f"{_RED}✗{_RST}"
        color = _GRN if passed else _RED
        print(f"  {icon}  {color}{test_id}{_RST}"
              + (f"  — {detail}" if detail else ""))
        return passed

    def summary(self) -> bool:
        total  = len(self._items)
        passed = sum(1 for _, ok, _ in self._items if ok)
        failed = total - passed
        bar    = "═" * 60
        print(f"\n{_BOLD}{bar}")
        if failed == 0:
            print(f"  {_GRN}ALL {total} TESTS PASSED{_RST}")
        else:
            print(f"  {_RED}{failed}/{total} TESTS FAILED{_RST}")
            for tid, ok, det in self._items:
                if not ok:
                    print(f"    {_RED}✗ {tid}{_RST}"
                          + (f": {det}" if det else ""))
        print(f"{_BOLD}{bar}{_RST}\n")
        return failed == 0


# ─────────────────────────── ROS node ───────────────────────────────────────
class VerifyNode(Node):
    """Single ROS2 node that performs all verifications."""

    def __init__(self):
        super().__init__("aesir_verify_stack")

        # Latest messages (written in callbacks, read in test methods)
        self._js:   Optional[JointState] = None
        self._odom: Optional[Odometry]   = None
        self._arm_cmd_echo: Optional[Float64MultiArray] = None
        self._lock = threading.Lock()

        # Subscribers
        self.create_subscription(JointState, "/joint_states",
                                 self._cb_js,   10)
        # diff_drive_controller publishes on `~/odom` → namespaced to its
        # controller name, NOT plain /odom.
        self.create_subscription(Odometry, "/diff_drive_controller/odom",
                                 self._cb_odom, 10)
        self.create_subscription(
            Float64MultiArray,
            "/joint_group_velocity_controller/commands",
            self._cb_arm_echo, 10,
        )

        # Publishers
        self._pub_arm_vel = self.create_publisher(
            Float64MultiArray, "/joint_group_velocity_controller/commands", 10)
        # diff_drive_controller subscribes on `~/cmd_vel_unstamped` (Twist)
        # when `use_stamped_vel: false` in ros2_controllers.yaml; for
        # `use_stamped_vel: true` it would be `~/cmd_vel` (TwistStamped).
        self._pub_base = self.create_publisher(
            Twist, "/diff_drive_controller/cmd_vel_unstamped", 10)
        self._pub_flip = self.create_publisher(
            Float64MultiArray, "/flipper_controller/commands", 10)

        # MoveGroup action client
        self._mg_client = ActionClient(self, MoveGroup, "move_group")

        # Controller manager service client
        self._ctrl_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers")

    # ── Callbacks ───────────────────────────────────────────────────────────
    def _cb_js(self, msg: JointState)              -> None:
        with self._lock: self._js = msg

    def _cb_odom(self, msg: Odometry)              -> None:
        with self._lock: self._odom = msg

    def _cb_arm_echo(self, msg: Float64MultiArray) -> None:
        with self._lock: self._arm_cmd_echo = msg

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _wait_for_js(self, timeout: float = TOPIC_WAIT_S) -> Optional[JointState]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._js is not None:
                    return self._js
            time.sleep(0.05)
        return None

    def _wait_for_odom(self, timeout: float = TOPIC_WAIT_S) -> Optional[Odometry]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._odom is not None:
                    return self._odom
            time.sleep(0.05)
        return None

    def _get_js(self) -> Optional[JointState]:
        with self._lock: return self._js

    def _get_odom(self) -> Optional[Odometry]:
        with self._lock: return self._odom

    def _zero_arm(self):
        msg      = Float64MultiArray()
        msg.data = [0.0] * 6
        self._pub_arm_vel.publish(msg)

    def _zero_base(self):
        self._pub_base.publish(Twist())

    def _zero_flippers(self):
        msg      = Float64MultiArray()
        msg.data = [0.0] * 4
        self._pub_flip.publish(msg)

    # ── MODULE 1: MoveIt alive ──────────────────────────────────────────────
    def m1_controller_service(self, res: Results) -> bool:
        ok = self._ctrl_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S)
        return res.add("1.1 controller_manager service",
                       ok, "" if ok else "service not available")

    def m1_controllers_active(self, res: Results) -> bool:
        if not self._ctrl_client.service_is_ready():
            for name in EXPECTED_CONTROLLERS:
                res.add(f"1.x {name}", False, "controller_manager unavailable")
            return False

        future = self._ctrl_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_S)
        if not future.done() or future.result() is None:
            for name in EXPECTED_CONTROLLERS:
                res.add(f"1.x {name}", False, "service call timeout")
            return False

        active = {c.name: c.state for c in future.result().controller}
        all_ok = True
        for i, name in enumerate(EXPECTED_CONTROLLERS, start=2):
            state = active.get(name, "NOT FOUND")
            ok    = state == "active"
            all_ok &= ok
            res.add(f"1.{i} {name}", ok, f"state={state}")
        return all_ok

    def m1_moveit_server(self, res: Results, skip: bool) -> bool:
        if skip:
            print(f"  {_YLW}~  1.7 move_group action server  [skipped]{_RST}")
            return True
        ok = self._mg_client.wait_for_server(timeout_sec=MOVEIT_TIMEOUT_S)
        return res.add("1.7 move_group action server",
                       ok, "" if ok else "server not reachable within timeout")

    # ── MODULE 2: MoveIt ↔ MuJoCo ──────────────────────────────────────────
    def m2_joint_states_arm(self, res: Results) -> Optional[JointState]:
        js = self._wait_for_js()
        if js is None:
            res.add("2.1 /joint_states (arm joints)", False,
                    "/joint_states not published")
            return None
        names  = set(js.name)
        miss   = [j for j in ARM_JOINTS if j not in names]
        ok     = len(miss) == 0
        detail = "" if ok else f"missing: {miss}"
        res.add("2.1 /joint_states — arm joints", ok, detail)
        return js

    def m2_joint_states_flippers(self, res: Results, js: Optional[JointState]) -> bool:
        if js is None:
            return res.add("2.2 /joint_states — flipper joints", False,
                           "/joint_states unavailable")
        names = set(js.name)
        miss  = [j for j in FLIPPER_JOINTS if j not in names]
        ok    = len(miss) == 0
        return res.add("2.2 /joint_states — flipper joints",
                       ok, "" if ok else f"missing: {miss}")

    def m2_odom(self, res: Results) -> Optional[Odometry]:
        odom = self._wait_for_odom()
        ok   = odom is not None
        res.add("2.3 /odom published", ok,
                "" if ok else "/odom not published — is diff_drive_controller active?")
        return odom

    def m2_arm_cmd_roundtrip(self, res: Results) -> bool:
        """Send a sustained arm velocity command and verify joint_1 POSITION
        drifts.
        """
        js_before = self._get_js()
        if js_before is None:
            return res.add("2.4 arm cmd → joint_1 position drift", False,
                           "/joint_states unavailable")

        idx_map = {n: i for i, n in enumerate(js_before.name)}
        j1_idx  = idx_map.get("joint_1")
        if j1_idx is None:
            return res.add("2.4 arm cmd → joint_1 position drift", False,
                           "joint_1 not in /joint_states")

        p_before = js_before.position[j1_idx] if js_before.position else 0.0

        # Hold a sizable velocity for ~2s so the integrated position drift is
        # well above noise.  We do NOT zero before reading so the equilibrium
        # offset doesn't snap back during the measurement.
        cmd      = Float64MultiArray()
        cmd.data = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        for _ in range(20):
            self._pub_arm_vel.publish(cmd)
            time.sleep(0.1)

        js_after = self._get_js()
        self._zero_arm()

        if js_after is None:
            return res.add("2.4 arm cmd → joint_1 position drift", False,
                           "no /joint_states after command")

        idx_map_a = {n: i for i, n in enumerate(js_after.name)}
        j1_idx_a  = idx_map_a.get("joint_1")
        p_after   = (js_after.position[j1_idx_a]
                     if js_after.position and j1_idx_a is not None else p_before)
        delta     = p_after - p_before
        changed   = abs(delta) > 0.005
        return res.add("2.4 arm cmd → joint_1 position drift",
                       changed,
                       f"joint_1 pos: {p_before:.4f} → {p_after:.4f}  "
                       f"(Δ={delta:+.4f})"
                       + ("" if changed else "  [DID NOT MOVE — controller "
                          "not reaching MJCF actuator]"))

    def m2_base_cmd_roundtrip(self, res: Results) -> bool:
        """Send sustained base Twist and check /odom linear velocity.
        """
        odom_before = self._get_odom()
        vx_before   = 0.0
        if odom_before is not None:
            vx_before = odom_before.twist.twist.linear.x

        cmd          = Twist()
        cmd.linear.x = 0.25

        # Hold the command for ~1.5s at 10 Hz so it outlives cmd_vel_timeout.
        for _ in range(15):
            self._pub_base.publish(cmd)
            time.sleep(0.1)

        odom_after = self._get_odom()
        self._zero_base()

        if odom_after is None:
            return res.add("2.5 base cmd → /odom velocity", False,
                           "no /odom after command")

        vx_after = odom_after.twist.twist.linear.x
        changed  = abs(vx_after - vx_before) > 0.01
        return res.add("2.5 base cmd → /odom velocity",
                       changed,
                       f"vx: {vx_before:.4f} → {vx_after:.4f}"
                       + ("" if changed else "  [DID NOT CHANGE — check diff_drive_controller]"))

    def m2_moveit_plan(self, res: Results, skip: bool) -> bool:
        if skip:
            print(f"  {_YLW}~  2.6 MoveGroup plan to HOME  [skipped]{_RST}")
            return True
        if not self._mg_client.server_is_ready():
            return res.add("2.6 MoveGroup plan to HOME", False,
                           "move_group action server not ready")

        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name            = ARM_GROUP
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 3

        constraints = Constraints()
        for name, pos in zip(ARM_JOINTS, HOME_POSITION):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.planning_options.plan_only = True   # plan but DO NOT execute

        future = self._mg_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=PLAN_TIMEOUT_S)
        if not future.done() or future.result() is None:
            return res.add("2.6 MoveGroup plan to HOME", False,
                           "goal acceptance timeout")

        handle = future.result()
        if not handle.accepted:
            return res.add("2.6 MoveGroup plan to HOME", False,
                           "goal rejected by move_group")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=PLAN_TIMEOUT_S)
        if not result_future.done():
            return res.add("2.6 MoveGroup plan to HOME", False, "result timeout")

        ec = result_future.result().result.error_code.val
        ok = ec == 1  # MoveItErrorCodes.SUCCESS
        return res.add("2.6 MoveGroup plan to HOME",
                       ok, f"MoveIt error_code={ec}" + ("" if ok else " (1=SUCCESS)"))

    # ── MODULE 3: Agent actuations ──────────────────────────────────────────
    def m3_agent_a_arm(self, res: Results) -> bool:
        """Agent A arm path: /joint_group_velocity_controller/commands.
        """
        js_before = self._get_js()
        if js_before is None:
            return res.add("3.1 [Agent A arm] joint_group_velocity_controller",
                           False, "/joint_states unavailable")

        idx = {n: i for i, n in enumerate(js_before.name)}
        j1  = idx.get("joint_1")
        if j1 is None:
            return res.add("3.1 [Agent A arm] joint_group_velocity_controller",
                           False, "joint_1 not in /joint_states")
        p0  = js_before.position[j1] if (js_before.position and j1 is not None) else 0.0

        cmd      = Float64MultiArray()
        cmd.data = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        for _ in range(20):
            self._pub_arm_vel.publish(cmd)
            time.sleep(0.1)

        js_after = self._get_js()
        self._zero_arm()

        if js_after is None:
            return res.add("3.1 [Agent A arm] joint_group_velocity_controller",
                           False, "no /joint_states after command")

        idx_a = {n: i for i, n in enumerate(js_after.name)}
        j1_a  = idx_a.get("joint_1")
        p1    = (js_after.position[j1_a]
                 if (js_after.position and j1_a is not None) else p0)
        delta = p1 - p0
        ok    = abs(delta) > 0.005
        return res.add("3.1 [Agent A arm] joint_group_velocity_controller",
                       ok,
                       f"joint_1 pos {p0:.4f} → {p1:.4f}  (Δ={delta:+.4f})")

    def m3_agent_a_grip_direct(self, res: Results) -> bool:
        """
        Agent A — grip (pos_left_finger, pos_right_finger):
        These actuators are MuJoCo-direct only (not exported via ros2_control).
        Verify they do NOT appear in /joint_states.
        """
        js = self._get_js()
        if js is None:
            return res.add("3.2 [Agent A grip] MuJoCo-direct (absent from /joint_states)",
                           False, "/joint_states unavailable")
        names     = set(js.name)
        # Grip joints in MuJoCo use the actuator name; ros2_control would expose finger joints
        grip_keys = ["pos_left_finger", "pos_right_finger",
                     "left_finger_joint", "right_finger_joint"]
        found     = [k for k in grip_keys if k in names]
        ok        = len(found) == 0
        detail    = ("confirmed absent — MuJoCo-direct ✓" if ok
                     else f"FOUND in /joint_states: {found} — review ros2_control xacro")
        return res.add("3.2 [Agent A grip] MuJoCo-direct (absent from /joint_states)",
                       ok, detail)

    def m3_agent_b_base(self, res: Results) -> bool:
        """Agent B base path: /diff_drive_controller/cmd_vel_unstamped.

        Like 2.5, we have to hold the Twist at 10 Hz to outlive the
        controller's cmd_vel_timeout (0.5 s).
        """
        odom0 = self._get_odom()
        vx0   = odom0.twist.twist.linear.x if odom0 else 0.0

        cmd          = Twist()
        cmd.linear.x = 0.30
        for _ in range(15):
            self._pub_base.publish(cmd)
            time.sleep(0.1)

        odom1 = self._get_odom()
        self._zero_base()

        if odom1 is None:
            return res.add("3.3 [Agent B base] /diff_drive_controller/cmd_vel",
                           False, "/odom unavailable")
        vx1 = odom1.twist.twist.linear.x
        ok  = abs(vx1 - vx0) > 0.01
        return res.add("3.3 [Agent B base] /diff_drive_controller/cmd_vel",
                       ok, f"odom vx {vx0:.4f} → {vx1:.4f}")

    def m3_agent_b_flippers(self, res: Results) -> bool:
        """
        Agent B — flippers:  /flipper_controller/commands
        Send position command, verify flipper joints change in /joint_states.
        Note: in train_ppo.py flippers go MuJoCo-direct; this tests the ROS path
        used when bringup is running (same physical joints, different command path).
        """
        js_before = self._get_js()
        if js_before is None:
            return res.add("3.4 [Agent B flippers] /flipper_controller/commands",
                           False, "/joint_states unavailable")

        idx_b = {n: i for i, n in enumerate(js_before.name)}
        f1_b  = idx_b.get("flipper_joint_1")
        p0    = js_before.position[f1_b] if (f1_b is not None and js_before.position) else None
        if p0 is None:
            return res.add("3.4 [Agent B flippers] /flipper_controller/commands",
                           False, "flipper_joint_1 not in /joint_states")

        cmd      = Float64MultiArray()
        cmd.data = [0.4, 0.4, 0.4, 0.4]
        self._pub_flip.publish(cmd)
        time.sleep(CMD_SETTLE_S)

        js_after  = self._get_js()
        self._zero_flippers()

        if js_after is None:
            return res.add("3.4 [Agent B flippers] /flipper_controller/commands",
                           False, "no /joint_states after command")

        idx_a = {n: i for i, n in enumerate(js_after.name)}
        f1_a  = idx_a.get("flipper_joint_1")
        p1    = js_after.position[f1_a] if (f1_a is not None and js_after.position) else p0
        ok    = abs(p1 - p0) > 0.01
        return res.add("3.4 [Agent B flippers] /flipper_controller/commands",
                       ok,
                       f"flipper_1 pos {p0:.4f} → {p1:.4f}")

    def m3_agent_b_fwheels_direct(self, res: Results) -> bool:
        """
        Agent B — flipper wheels (vel_flip*):
        MuJoCo-direct only — must NOT appear in /joint_states.
        """
        js = self._get_js()
        if js is None:
            return res.add(
                "3.5 [Agent B flip-wheels] MuJoCo-direct (absent from /joint_states)",
                False, "/joint_states unavailable")
        names    = set(js.name)
        fw_keys  = [j for j in MUJOCO_DIRECT_JOINTS if "flip" in j]
        found    = [k for k in fw_keys if k in names]
        ok       = len(found) == 0
        detail   = ("confirmed absent — MuJoCo-direct ✓" if ok
                    else f"FOUND in /joint_states: {found}")
        return res.add(
            "3.5 [Agent B flip-wheels] MuJoCo-direct (absent from /joint_states)",
            ok, detail)


# ─────────────────────────── Runner ────────────────────────────────────────
def _banner(title: str) -> None:
    bar = "─" * 60
    print(f"\n{_BOLD}{bar}")
    print(f"  {_CYN}{title}{_RST}")
    print(f"{_BOLD}{bar}{_RST}")


def run(args: argparse.Namespace) -> bool:
    rclpy.init()
    node = VerifyNode()

    # Spin in background so callbacks are processed while we run tests
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name="verify_spin"
    )
    spin_thread.start()

    res   = Results()
    mods  = set(args.module) if args.module else {1, 2, 3}

    try:
        # ── MODULE 1 ───────────────────────────────────────────────────────
        if 1 in mods:
            _banner("MODULE 1 — MoveIt alive")
            svc_ok = node.m1_controller_service(res)
            node.m1_controllers_active(res)
            node.m1_moveit_server(res, skip=args.skip_plan)

        # ── MODULE 2 ───────────────────────────────────────────────────────
        if 2 in mods:
            _banner("MODULE 2 — MoveIt ↔ MuJoCo connection")
            js = node.m2_joint_states_arm(res)
            node.m2_joint_states_flippers(res, js)
            node.m2_odom(res)
            node.m2_arm_cmd_roundtrip(res)
            node.m2_base_cmd_roundtrip(res)
            node.m2_moveit_plan(res, skip=args.skip_plan)

        # ── MODULE 3 ───────────────────────────────────────────────────────
        if 3 in mods:
            _banner("MODULE 3 — Agent A / Agent B actuations")
            node.m3_agent_a_arm(res)
            node.m3_agent_a_grip_direct(res)
            node.m3_agent_b_base(res)
            node.m3_agent_b_flippers(res)
            node.m3_agent_b_fwheels_direct(res)

    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        # Always leave the robot stopped
        node._zero_arm()
        node._zero_base()
        node._zero_flippers()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)

    return res.summary()


# ─────────────────────────── Entry point ───────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the Aesir RL stack (MoveIt + MuJoCo + agents)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--module", type=int, nargs="+", choices=[1, 2, 3],
        metavar="N",
        help="Run only the specified module(s).  Default: run all three.",
    )
    p.add_argument(
        "--skip-plan", action="store_true",
        help="Skip tests that require the MoveIt planning pipeline "
             "(move_group action server).  Useful when only the controllers "
             "and the raw topics need to be verified.",
    )
    return p.parse_args()


if __name__ == "__main__":
    ok = run(_parse())
    sys.exit(0 if ok else 1)
