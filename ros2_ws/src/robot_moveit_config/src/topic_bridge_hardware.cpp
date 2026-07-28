#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include <vector>
#include <mutex>

namespace aesir_plugin
{

// Conversion helpers
//  ros_to_hw : ROS [-π, π]  →  Hardware [0, 2π]   (add π)
//  hw_to_ros : Hardware [0, 2π]  →  ROS [-π, π]   (subtract π, then normalise)
static inline double ros_to_hw(double rad)
{
  return rad + M_PI;
}

static inline double hw_to_ros(double rad)
{
  // Subtract π then normalise to [-π, π] in case hardware sends values
  // slightly outside [0, 2π]
  return std::fmod(rad - M_PI + M_PI, 2.0 * M_PI) - M_PI;
}

class TopicBridgeHardware : public hardware_interface::SystemInterface
{
private:
  rclcpp::Node::SharedPtr node_;

  // Publisher for standard JointState commands
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;

  // Subscriber for incoming physical hardware feedback
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;

  // Memory for Position, Velocity, and Effort
  std::vector<double> hw_commands_;     // position commands
  std::vector<double> hw_vel_cmds_;     // velocity commands
  std::vector<double> hw_eff_cmds_;     // effort commands

  std::vector<double> hw_states_;            // position states
  std::vector<double> hw_vel_states_;        // velocity states
  std::vector<double> hw_eff_states_;        // effort states
  std::vector<double> latest_hw_states_;     // lastest position states
  std::vector<double> latest_hw_vel_states_; // velocity states

  std::mutex state_mutex_; // Protects the state arrays from being read and written at the exact same time

  const std::vector<std::string> target_joints = {
      "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
  };

  static constexpr double DEFAULT_ACCELERATION = 3.14159265;

public:
  RCLCPP_SHARED_PTR_DEFINITIONS(TopicBridgeHardware)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override
  {
    if (hardware_interface::SystemInterface::on_init(info) !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
      return hardware_interface::CallbackReturn::ERROR;
    }

    const size_t nr_joints = info_.joints.size();

    hw_states_.resize(nr_joints, 0.0);
    hw_vel_states_.resize(nr_joints, 0.0);
    hw_eff_states_.resize(nr_joints, 0.0);

    latest_hw_states_.resize(nr_joints, 0.0);
    latest_hw_vel_states_.resize(nr_joints, 0.0);

    hw_commands_.resize(nr_joints, 0.0);
    hw_vel_cmds_.resize(nr_joints, 0.0);
    hw_eff_cmds_.resize(nr_joints, 0.0);

    // -----------------------------------------------------------------
    // Node setup
    // -----------------------------------------------------------------
    node_ = std::make_shared<rclcpp::Node>("hw_bridge_node");

    // Read initial values from URDF if present
    // initial_value is expressed in ROS radians [-π, π], but store as hardware
    for (size_t i = 0; i < nr_joints; ++i) {
      for (const auto& state_interface : info_.joints[i].state_interfaces) {
        if (state_interface.name == hardware_interface::HW_IF_POSITION) {
          if (!state_interface.initial_value.empty()) {
            try {
              double initial_val = std::stod(state_interface.initial_value);
              
              hw_states_[i]        = initial_val;
              latest_hw_states_[i] = initial_val;
              hw_commands_[i]      = initial_val;

              RCLCPP_INFO(
                node_->get_logger(),
                "Joint '%s' initialized to %.4f from URDF",
                info_.joints[i].name.c_str(),
                initial_val
              );
            } catch (const std::exception& e) {
              RCLCPP_WARN(
                node_->get_logger(),
                "Failed to parse initial_value for joint '%s': %s",
                info_.joints[i].name.c_str(),
                e.what()
              );
            }
          }
          break;
        }
      }
    }

    // Publisher: standard JointState commands
    command_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      "/commands_hardware", 10);

    // Subscriber: Listen to physical hardware encoders
    feedback_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      "/hardware_node/joint_states", 
      10,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg)
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        // Match the incoming joint names to the correct index in our arrays
        for (size_t i = 0; i < msg->name.size(); ++i) {
          for (size_t j = 0; j < info_.joints.size(); ++j) {
            if (info_.joints[j].name == msg->name[i]) {
              if (i < msg->position.size()) {
                // Hardware [0, 2π]  →  ROS [-π, π]
                latest_hw_states_[j] = hw_to_ros(msg->position[i]);
              }
              if (i < msg->velocity.size()) {
                latest_hw_vel_states_[j] = msg->velocity[i];
              }
              break; 
            }
          }
        }
      });

    RCLCPP_INFO(node_->get_logger(),
      "TopicBridgeHardware initialised with %zu joints. "
      "Commands are published as JointState; effort values are used as acceleration.",
      nr_joints);

    return hardware_interface::CallbackReturn::SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // Export state interfaces: Position, Velocity, Effort per joint
  // ---------------------------------------------------------------------------
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override
  {
    std::vector<hardware_interface::StateInterface> state_interfaces;
    for (size_t i = 0; i < info_.joints.size(); ++i) {
      state_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_[i]);
      state_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_vel_states_[i]);
      state_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT,   &hw_eff_states_[i]);
    }
    return state_interfaces;
  }

  // ---------------------------------------------------------------------------
  // Export command interfaces: Position, Velocity, Effort per joint
  // (Acceleration is internal — not a ros2_control command interface)
  // ---------------------------------------------------------------------------
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override
  {
    std::vector<hardware_interface::CommandInterface> command_interfaces;
    for (size_t i = 0; i < info_.joints.size(); ++i) {
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_vel_cmds_[i]);
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT,   &hw_eff_cmds_[i]);
    }
    return command_interfaces;
  }

  // ---------------------------------------------------------------------------
  // Read: feedback — states mirror commands so MoveIt sees instant motion
  // ---------------------------------------------------------------------------
  hardware_interface::return_type read(
    const rclcpp::Time & /*time*/,
    const rclcpp::Duration & /*period*/) override
  {
    // Spin the node once to process incoming feedback messages
    rclcpp::spin_some(node_);

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      for (size_t i = 0; i < hw_states_.size(); ++i) {
        hw_states_[i]     = latest_hw_states_[i];
        hw_vel_states_[i] = latest_hw_vel_states_[i];
      }
    }

    return hardware_interface::return_type::OK;
  }

  // ---------------------------------------------------------------------------
  // Write: publish standard JointState with position, velocity, effort
  // (effort is interpreted as acceleration)
  // ---------------------------------------------------------------------------
  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & /*period*/) override
  {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = time;

    for (size_t i = 0; i < info_.joints.size(); ++i) {
      // Check if the current joint name exists in our target list
      bool is_target = std::find(target_joints.begin(), target_joints.end(), info_.joints[i].name) != target_joints.end();
      if (is_target) {
        // ROS [-π, π]  →  Hardware [0, 2π]
        double pos_cmd = ros_to_hw(hw_commands_[i]);

        msg.name.push_back(info_.joints[i].name);
        msg.position.push_back(pos_cmd);
        msg.velocity.push_back(hw_vel_cmds_[i]);
        msg.effort.push_back(hw_eff_cmds_[i]);
      }
    }

    if (!msg.name.empty()) {
      command_pub_->publish(msg);
    }

    return hardware_interface::return_type::OK;
  }
};

}  // namespace aesir_plugin

PLUGINLIB_EXPORT_CLASS(aesir_plugin::TopicBridgeHardware, hardware_interface::SystemInterface)