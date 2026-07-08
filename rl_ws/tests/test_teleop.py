#!/usr/bin/env python3
"""
aesir_teleop.py - Keyboard teleoperation for tracks and flippers.
"""
import sys
import select
import termios
import tty
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from hardware.msg import JointControl

# Keybindings
MOVE_BINDINGS = {
    'w': (1.0, 0.0),   # Forward
    's': (-1.0, 0.0),  # Backward
    'a': (0.0, 1.0),   # Turn Left
    'd': (0.0, -1.0),  # Turn Right
}

FLIPPER_BINDINGS = {
    't': (1, 1),   # Front flippers UP
    'g': (1, -1),  # Front flippers DOWN
    'y': (2, 1),   # Rear flippers UP
    'h': (2, -1),  # Rear flippers DOWN
}

msg = """
AESIR Robot Keyboard Control
---------------------------
Moving the base:
        w
   a    s    d

Controlling the flippers:
   t : Front Flippers UP
   g : Front Flippers DOWN
   y : Rear Flippers UP
   h : Rear Flippers DOWN

Spacebar : Force stop / Reset Twist
CTRL-C to quit
"""

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class AesirTeleop(Node):
    def __init__(self):
        super().__init__('aesir_teleop')
        self.cmd_vel_pub = self.create_publisher(Twist, '/hardware_node/cmd_vel', 10)
        self.joint_pub = self.create_publisher(JointControl, '/commands_hardware', 10)

        # Robot parameters
        self.speed = 1.5 # m/s
        self.turn = 3.14159  # rad/s
        self.flipper_step = 0.1 # rad per keypress

        # Track the angle offset from the neutral position (0.0 = horizontal)
        self.front_angle = 0.0
        self.rear_angle = 0.0

    def publish_twist(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear * self.speed
        twist.angular.z = angular * self.turn
        self.cmd_vel_pub.publish(twist)

    def publish_flippers(self):
        jc = JointControl()
        
        # Your custom group order (2 & 3 are front, 1 & 4 are rear)
        jc.joint_names = ['flipper_2_joint', 'flipper_3_joint', 'flipper_1_joint', 'flipper_4_joint']
        
        # Calculate the hardware position (neutral is math.pi)
        # We invert (-) the angle for one side to fix the mirroring issue!
        front_left  = math.pi + self.front_angle
        front_right = math.pi - self.front_angle 
        
        rear_left   = math.pi + self.rear_angle
        rear_right  = math.pi - self.rear_angle

        # Map them to the joint names list above
        jc.position = [
            front_left,   # flipper_2_joint
            front_right,  # flipper_3_joint
            rear_left,    # flipper_1_joint
            rear_right    # flipper_4_joint
        ]
        
        self.joint_pub.publish(jc)

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = AesirTeleop()

    print(msg)

    x = 0.0
    th = 0.0

    try:
        while rclpy.ok():
            key = get_key(settings)

            if key in MOVE_BINDINGS.keys():
                x = MOVE_BINDINGS[key][0]
                th = MOVE_BINDINGS[key][1]
                node.publish_twist(x, th)

            elif key in FLIPPER_BINDINGS.keys():
                flipper_group, direction = FLIPPER_BINDINGS[key]
                if flipper_group == 1:
                    node.front_angle += direction * node.flipper_step
                elif flipper_group == 2:
                    node.rear_angle += direction * node.flipper_step
                node.publish_flippers()

            elif key == ' ':
                x = 0.0
                th = 0.0
                node.publish_twist(x, th)
                
            elif key == '\x03': # CTRL-C
                break
            
            else:
                # Stop base if no movement key is held
                if x != 0.0 or th != 0.0:
                    x = 0.0
                    th = 0.0
                    node.publish_twist(x, th)

    except Exception as e:
        print(e)
    finally:
        node.publish_twist(0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()