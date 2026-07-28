#!/usr/bin/env python3
"""
aesir_teleop.py - Keyboard teleoperation for tracks and flippers.
"""
import os
import sys
import select
import termios
import tty
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# Raiz del proyecto al path para leer los MISMOS limites que usa el entrenamiento
# (tests/ -> rl_ws -> aesir_rl). Asi 'w'/'a'/'d' comandan exactamente V_MAX/W_MAX
# -> el teleop refleja el tope real que vera la politica, no valores hardcodeados.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from rl_ws.base_training.config import V_MAX_MPS, W_MAX_RADPS, FLIPPER_MAX_ACCEL

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
        self.joint_pub = self.create_publisher(JointState, '/commands_hardware', 10)

        # Robot parameters — MISMOS limites que el entrenamiento (config.py), asi
        # 'w'/'a'/'d' mandan justo el tope que comandaria la politica (accion=1).
        self.speed = V_MAX_MPS   # m/s   (tope lineal calibrado)
        self.turn  = W_MAX_RADPS # rad/s (tope de giro calibrado)
        self.flipper_step = 0.1  # rad per keypress
        print(f"[teleop] V_MAX={self.speed} m/s  W_MAX={self.turn} rad/s  (de config.py)")

        # Track the angle offset from the neutral position (0.0 = horizontal)
        self.front_angle = 0.0
        self.rear_angle = 0.0

    def publish_twist(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear * self.speed
        twist.angular.z = angular * self.turn
        self.cmd_vel_pub.publish(twist)

    def publish_flippers(self):
        jc = JointState()

        # Numeracion del modelo: flipper_1,2 = FRENTE ; flipper_3,4 = ATRAS.
        # Los dos flippers de cada par tienen el mismo eje (delanteros (0,1,0),
        # traseros (0,-1,0)), asi que con el mismo valor ya se mueven juntos ->
        # NO hay que invertir ningun lado (antes se invertia por el modelo viejo).
        # Neutro = math.pi (convencion "hardware"); + front_angle mueve el par.
        jc.name = ['flipper_1_joint', 'flipper_2_joint',
                   'flipper_3_joint', 'flipper_4_joint']
        jc.position = [
            math.pi + self.front_angle,   # flipper_1 (frente-izq)
            math.pi + self.front_angle,   # flipper_2 (frente-der)
            math.pi + self.rear_angle,    # flipper_3 (atras-izq)
            math.pi + self.rear_angle,    # flipper_4 (atras-der)
        ]
        jc.effort = [
            float(np.sign(pos - math.pi) * FLIPPER_MAX_ACCEL)
            for pos in jc.position
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