cmake_minimum_required(VERSION 3.22)
project(robot_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

# -----------------------------------------------------------------------------
# DISCONNECTED: aesir_plugin / TopicBridgeHardware
# -----------------------------------------------------------------------------
# These dependencies were required by the topic-bridge hardware plugin.
# The integration path now goes through mujoco_ros2_control/MujocoSystem
# (see rescue_robot.ros2_control.xacro), so the plugin is not built and
# pluginlib does not export it. Source files in src/ are left in place so
# you can re-enable everything by uncommenting the block below.
# -----------------------------------------------------------------------------
# find_package(hardware_interface REQUIRED)
# find_package(pluginlib REQUIRED)
# find_package(hardware REQUIRED)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

# -----------------------------------------------------------------------------
# arm_command_node: small MoveIt utility, unrelated to the hardware plugin.
# -----------------------------------------------------------------------------
add_executable(arm_command_node src/arm_command_node.cpp)
ament_target_dependencies(arm_command_node
  rclcpp
  geometry_msgs
  sensor_msgs
  moveit_ros_planning_interface)
install(TARGETS arm_command_node DESTINATION lib/${PROJECT_NAME})

# -----------------------------------------------------------------------------
# DISCONNECTED: TopicBridgeHardware shared library + pluginlib export
# -----------------------------------------------------------------------------
# add_library(${PROJECT_NAME} SHARED src/topic_bridge_hardware.cpp)
# ament_target_dependencies(${PROJECT_NAME}
#   rclcpp
#   hardware
#   hardware_interface
#   pluginlib
#   sensor_msgs
# )
# pluginlib_export_plugin_description_file(hardware_interface aesir_plugin.xml)
# install(TARGETS ${PROJECT_NAME} DESTINATION lib)

install(PROGRAMS
  scripts/test_teleop.py
  scripts/test_teleop_joints.py
  scripts/hardware_loopback.py
  scripts/diagnose_pipeline.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()
