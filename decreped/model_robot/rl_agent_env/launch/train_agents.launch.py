"""End-to-end launch for the Aesir RL stack.

Brings up, in dependency order:

    A. robot_state_publisher    (xacro -> URDF, /tf, /tf_static)
    B. mujoco_ros2_control      (physics + ros2_control hardware bridge)
    C. controller spawners      (joint_state_broadcaster,
                                 joint_group_velocity_controller for the arm,
                                 diff_drive_controller for the base,
                                 flipper_controller)
    D. moveit_servo             (TwistStamped -> Float64MultiArray -> arm vel)
    E. rl_agent_env             (RL env node, owns PPO training loop)

Run:
    ros2 launch rl_agent_env train_agents.launch.py
    ros2 launch rl_agent_env train_agents.launch.py mujoco_scene:=/abs/path/to/scene.xml
"""
import os
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _static_paths():
    """Return a dict of absolute paths resolved at parse time."""
    moveit_pkg = get_package_share_directory("robot_moveit_config")
    desc_pkg   = get_package_share_directory("aesir_robot_description")
    return {
        "urdf_xacro":   os.path.join(moveit_pkg, "config", "rescue_robot.urdf.xacro"),
        "srdf":         os.path.join(moveit_pkg, "config", "rescue_robot.srdf"),
        "kinematics":   os.path.join(moveit_pkg, "config", "kinematics.yaml"),
        "joint_limits": os.path.join(moveit_pkg, "config", "joint_limits.yaml"),
        "servo":        os.path.join(moveit_pkg, "config", "servo_params.yaml"),
        "controllers":  os.path.join(moveit_pkg, "config", "ros2_controllers.yaml"),
        "default_scene": os.path.join(desc_pkg,  "launch", "aesir_complete.xml"),
    }


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_nodes(context, *args, **kwargs):
    """OpaqueFunction: resolved at launch time so LaunchConfiguration values are available."""
    paths         = _static_paths()
    mujoco_scene  = context.perform_substitution(LaunchConfiguration("mujoco_scene"))
    use_sim_time  = context.perform_substitution(LaunchConfiguration("use_sim_time"))
    log_level     = context.perform_substitution(LaunchConfiguration("log_level"))

    # ── Process xacro with the runtime mujoco_scene path ─────────────────────
    # The <param name="mujoco_model"> inside rescue_robot.ros2_control.xacro
    # is filled by the xacro arg "mujoco_model", which we inject here.
    robot_description_content = xacro.process_file(
        paths["urdf_xacro"],
        mappings={
            "mujoco_model": mujoco_scene,
        },
    ).toxml()
    robot_description = {"robot_description": robot_description_content}

    # SRDF stays as raw XML string under the canonical parameter name.
    with open(paths["srdf"]) as f:
        srdf_content = f.read()
    robot_description_semantic = {"robot_description_semantic": srdf_content}

    # kinematics.yaml is a flat dict of {group_name: kinematics_solver_cfg},
    # wrapped under `robot_description_kinematics` so MoveIt finds it.
    kinematics_param = {
        "robot_description_kinematics": _load_yaml(paths["kinematics"]),
    }

    # joint_limits.yaml is a dict with `default_velocity_scaling_factor`,
    # `default_acceleration_scaling_factor`, and `joint_limits`.
    # MoveIt reads these from the `robot_description_planning` namespace.
    joint_limits_param = {
        "robot_description_planning": _load_yaml(paths["joint_limits"]),
    }

    # servo_params.yaml uses the standard `/**: ros__parameters: moveit_servo: …`
    # wildcard. Extract the inner block so we can pass it as a Python dict; the
    # launch system will serialize nested dicts into dotted parameter names so
    # moveit_servo sees `moveit_servo.move_group_name`, etc.
    servo_yaml = _load_yaml(paths["servo"])
    servo_params = servo_yaml["/**"]["ros__parameters"]

    sim_time_param = {"use_sim_time": use_sim_time == "true"}

    # ── A. robot_state_publisher ──────────────────────────────────────────────
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description, sim_time_param],
    )

    # ── B. mujoco_ros2_control ────────────────────────────────────────────────
    # Uses the custom ros2_control_node bundled with the package.
    # The MuJoCo model path is embedded in the URDF <param name="mujoco_model">
    # by the xacro processing above, so NO extra parameter needed here.
    # Do NOT set name=: the executable creates a node named "controller_manager"
    # internally; overriding the name would move all services away from
    # /controller_manager/* breaking the spawners.
    mujoco_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            robot_description,
            paths["controllers"],       # yaml path string → auto-loaded
            sim_time_param,
            {"headless": True},
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    # ── C. controller spawners ────────────────────────────────────────────────
    # jsb_spawner retries until /controller_manager comes up (mujoco_node init)
    jsb_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    arm_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_group_velocity_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    diff_drive_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["diff_drive_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    flipper_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["flipper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # arm/diff_drive/flipper spawners fire after jsb exits (jsb is short-lived)
    spawn_rest_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=jsb_spawner,
            on_exit=[arm_spawner, diff_drive_spawner, flipper_spawner],
        )
    )

    # ── D. moveit_servo ───────────────────────────────────────────────────────
    # Parameters MUST be properly loaded YAML dicts. Previously this passed
    # the YAML *file contents* as a string under "servo_params", which caused
    # the servo node to fall back to its compiled-in defaults — including
    # move_group_name="panda_arm" — and crash with
    #   "Invalid move group name: 'panda_arm'".
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            kinematics_param,
            joint_limits_param,
            sim_time_param,
        ],
    )
    start_servo_after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[servo_node])
    )

    # ── E. RL environment ─────────────────────────────────────────────────────
    rl_env_node = Node(
        package="rl_agent_env",
        executable="rl_env",
        name="rl_communication_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time_param],
        arguments=["--ros-args", "--log-level", log_level],
    )
    start_rl_after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[rl_env_node])
    )

    return [
        rsp_node,
        mujoco_node,
        jsb_spawner,
        spawn_rest_after_jsb,
        start_servo_after_arm,
        start_rl_after_arm,
    ]


def generate_launch_description() -> LaunchDescription:
    paths = _static_paths()

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use /clock from MuJoCo simulation.",
        ),
        DeclareLaunchArgument(
            "log_level", default_value="info",
        ),
        DeclareLaunchArgument(
            "mujoco_scene",
            default_value=paths["default_scene"],
            description=(
                "Absolute path to the MuJoCo XML scene. "
                "For full RL training: "
                "src/aesir_robot_description/launch/aesir_complete.xml"
            ),
        ),
        LogInfo(msg=["[rl_agent_env] Launching Aesir end-to-end RL stack."]),
        OpaqueFunction(function=_build_nodes),
    ])
