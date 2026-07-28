#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class LoopbackSimulator(Node):
    def __init__(self):
        super().__init__('hardware_loopback_node')
        
        # Publisher: Send the "fake" physical encoder data back to the C++ plugin
        self.feedback_pub = self.create_publisher(JointState, '/hardware_node/joint_states', 10)
        
        # Subscriber: Listen to the commands your C++ plugin is sending out
        self.command_sub = self.create_subscription(
            JointState,
            '/commands_hardware',
            self.command_callback,
            10
        )
        self.get_logger().info('Loopback active: Mirroring /commands_hardware -> /hardware_node/joint_states')

    def command_callback(self, cmd_msg):
        # Create a standard ROS 2 JointState message
        state_msg = JointState()
        
        # Give it a fresh timestamp so MoveIt knows it's current
        state_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Mirror the standard JointState payload back as simulated feedback.
        state_msg.name = cmd_msg.name
        state_msg.position = cmd_msg.position
        state_msg.velocity = cmd_msg.velocity
        state_msg.effort = cmd_msg.effort
        
        # Publish the simulated physical hardware feedback!
        self.feedback_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LoopbackSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()