#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
import sys
import select
import termios
import tty

# Mapeo de teclas: (linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
KEY_BINDINGS = {
    'w': ( 1.5,  0.0,  0.0,  0.0,  0.0,  0.0), # Adelante X
    's': (-1.5,  0.0,  0.0,  0.0,  0.0,  0.0), # Atrás X
    'a': ( 0.0,  1.5,  0.0,  0.0,  0.0,  0.0), # Izquierda Y
    'd': ( 0.0, -1.5,  0.0,  0.0,  0.0,  0.0), # Derecha Y
    'q': ( 0.0,  0.0,  1.5,  0.0,  0.0,  0.0), # Arriba Z
    'e': ( 0.0,  0.0, -1.5,  0.0,  0.0,  0.0), # Abajo Z
    'u': ( 0.0,  0.0,  0.0,  3.0,  0.0,  0.0), # Roll +
    'o': ( 0.0,  0.0,  0.0, -3.0,  0.0,  0.0), # Roll -
    'i': ( 0.0,  0.0,  0.0,  0.0,  3.0,  0.0), # Pitch +
    'k': ( 0.0,  0.0,  0.0,  0.0, -3.0,  0.0), # Pitch -
    'j': ( 0.0,  0.0,  0.0,  0.0,  0.0,  3.0), # Yaw +
    'l': ( 0.0,  0.0,  0.0,  0.0,  0.0, -3.0), # Yaw -
}

msg_info = """
---------------------------------------------
¡Controlador de Teclado Optimizado para Servo!
---------------------------------------------
Moviendo el Efector Final (Traslación):
   w : Adelante (+X)    s : Atrás (-X)
   a : Izquierda (+Y)   d : Derecha (-Y)
   q : Arriba (+Z)      e : Abajo (-Z)

Girando la Muñeca (Rotación):
   u / o : Roll (+/-)
   i / k : Pitch (+/-)
   j / l : Yaw (+/-)

Frenado automático al soltar la tecla.
CTRL-C para salir de forma segura.
---------------------------------------------
"""

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # Eliminada la declaración redundante de 'use_sim_time' para evitar crasheos en ROS 2 Humble
        
        self.cli = self.create_client(Trigger, '/servo_node/start_servo')
        self.get_logger().info('Esperando al servicio /servo_node/start_servo...')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servicio no disponible, esperando de nuevo...')
            
        self.get_logger().info('Servicio encontrado. Iniciando Servo...')
        self.req = Trigger.Request()
        
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.start_servo_callback)

        # Configuración para leer el teclado en Linux
        self.settings = termios.tcgetattr(sys.stdin)
        self.speed_multiplier = 0.3  # Velocidad moderada de servo
        self.was_moving = False       # Flag para saber si necesitamos mandar comando de freno

        self.get_logger().info('Nodo de teleoperación iniciado.')
        print(msg_info)

        # Correr el bucle de lectura a ~20Hz para un control fluido
        self.timer = self.create_timer(0.05, self.timer_callback)

    def start_servo_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Éxito al iniciar servo: {response.message}')
            else:
                self.get_logger().warn(f'Servo reportó fallo al iniciar: {response.message}')
        except Exception as e:
            self.get_logger().error(f'La llamada al servicio falló: {e}')

    def get_key(self):
        """Lee una sola tecla de la terminal sin bloquear el programa."""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_twist(self, x, y, z, th_x, th_y, th_z):
        """Genera y publica el mensaje de Twist con un sello de tiempo fresco."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'arm_base_link'
        
        msg.twist.linear.x = x * self.speed_multiplier
        msg.twist.linear.y = y * self.speed_multiplier
        msg.twist.linear.z = z * self.speed_multiplier
        msg.twist.angular.x = th_x * self.speed_multiplier
        msg.twist.angular.y = th_y * self.speed_multiplier
        msg.twist.angular.z = th_z * self.speed_multiplier
        self.publisher_.publish(msg)

    def timer_callback(self):
        key = self.get_key()
        
        if key == '\x03': # Ctrl+C
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rclpy.shutdown()
            return

        if key in KEY_BINDINGS:
            # Si hay una tecla válida presionada, se manda la velocidad correspondiente
            x, y, z, th_x, th_y, th_z = KEY_BINDINGS[key]
            self.publish_twist(x, y, z, th_x, th_y, th_z)
            self.was_moving = True
        else:
            # OPTIMIZACIÓN: Solo enviamos un comando de parada única si veníamos moviéndonos.
            # Esto evita saturar MoveIt con mensajes de velocidad cero redundantes.
            if self.was_moving:
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.was_moving = False

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    
    # Quick fix: Save the original terminal settings safely here so the 'finally' block can properly restore them
    original_settings = termios.tcgetattr(sys.stdin)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Restaurar la terminal al salir de forma segura
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()