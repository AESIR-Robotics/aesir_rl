# Cómo lanzar el stack completo con MoveIt

## Cuándo necesitas esto

Solo cuando quieras correr el agente **con ROS2 activo** — es decir, en deployment
sobre el robot real o en la simulación con ros2_control.

Para entrenamiento RL puro en MuJoCo, **no necesitas nada de esto**.
`ppo_conv_train.py` corre solo, sin ROS.

---

## 3 terminales, en orden

### Terminal 1 — Simulador + ros2_control + MoveIt base

```bash
source ~/aesir_rl/workspace/install/setup.bash
ros2 launch robot_moveit_config bringup.launch.py
```

Esto lanza (en un solo comando):
- `demo.launch.py` → robot_state_publisher, move_group, controller_manager
- `servo_teleop.launch.py` → servo_node (espera 3 segundos antes de iniciarse)
- spawner del `flipper_controller`

Espera a ver en la consola:
```
[servo_node]: Servo ready
[controller_manager]: Loaded controllers: arm_controller, joint_group_velocity_controller, diff_drive_controller, flipper_controller
```

### Terminal 2 — Publicador de lidar (si usas simulación MuJoCo con ROS)

```bash
source ~/aesir_rl/workspace/install/setup.bash
ros2 run aesir_robot_description lidar_publisher \
  --ros-args -p scene_xml:=/ruta/a/aesir_mujoco.xml
```

Esto publica `/scan` desde los rangefinders de MuJoCo.
Si tu simulación ya publica `/scan` de otra forma, omite este terminal.

### Terminal 3 — Agente RL

```bash
source ~/aesir_rl/workspace/install/setup.bash
cd rl_scripts
python3 train_ppo.py
```

---

## Qué hace cada publicación del agente

El `ros_bridge.py` corregido publica así:

| Acción del agente | Topic publicado | Recibido por |
|---|---|---|
| `v_lin`, `ω_ang` | `/diff_drive_controller/cmd_vel` | diff_drive_controller |
| `flipper_1..4` | `/flipper_controller/commands` | flipper_controller |
| `joint_1..6 vel` | `/servo_node/delta_joint_cmds` | MoveIt Servo → IK → velocity controller |
| `dedos` | `/joint_group_velocity_controller/commands` | velocity controller directo |

MoveIt Servo recibe el `JointJog` con las velocidades articulares, verifica colisiones,
y manda el resultado al `joint_group_velocity_controller`. El agente nunca ve esa capa.

---

## El cambio clave en ros_bridge.py

El bridge anterior publicaba el brazo así (incorrecto para usar MoveIt):
```python
# ❌ Saltaba MoveIt completamente
self._pub_arm_vel.publish(Float64MultiArray(...))  # → velocity controller directo
```

El bridge corregido lo hace así:
```python
# ✅ Pasa por MoveIt Servo (hace cinemática, verifica colisiones)
msg = JointJog()
msg.joint_names = ["joint_1", ..., "joint_6"]
msg.velocities  = [v1, v2, v3, v4, v5, v6]
msg.duration    = 0.1
self._pub_arm_servo.publish(msg)  # → /servo_node/delta_joint_cmds
```

MoveIt Servo luego publica al `/joint_group_velocity_controller/commands` automáticamente
(configurado en `servo_params.yaml` → `command_out_topic`).

---

## Verificar que todo está conectado

```bash
# Ver todos los topics activos
ros2 topic list

# Verificar que el servo_node está recibiendo comandos
ros2 topic echo /servo_node/delta_joint_cmds

# Verificar que el brazo se mueve (velocity controller recibe comandos de MoveIt)
ros2 topic echo /joint_group_velocity_controller/commands

# Ver estado del brazo
ros2 topic echo /joint_states

# Ver odometría de la base
ros2 topic echo /odom
```

---

## Configuración ya correcta en tus archivos

`servo_params.yaml` ya tiene todo bien:
```yaml
joint_command_in_topic: "/servo_node/delta_joint_cmds"   # donde escucha
command_out_topic: "/joint_group_velocity_controller/commands"  # donde publica
command_out_type: "std_msgs/Float64MultiArray"
publish_joint_velocities: true
check_collisions: true
```

`ros2_controllers.yaml` ya tiene los controllers correctos:
- `diff_drive_controller` para la base (wheels: drive_l_1..3, drive_r_1..3)
- `flipper_controller` para los 4 flippers
- `joint_group_velocity_controller` para el brazo (recibe de MoveIt Servo)

---

## Si falla al lanzar

**`[servo_node] Could not find planning group 'arm'`**
Verifica que el SRDF de tu robot define el grupo `arm` con los joints correctos.
Busca en `robot_moveit_config/config/` el archivo `.srdf`.

**`[controller_manager] Could not load controller: flipper_controller`**
El URDF no expone interfaces `position` para los flipper joints.
Necesitas añadir `<command_interface name="position"/>` en el URDF para cada flipper joint.

**`JointJog: ImportError`**
Instala el paquete: `sudo apt install ros-$ROS_DISTRO-control-msgs`

**El brazo no se mueve aunque el servo_node recibe comandos**
Verifica que el `joint_group_velocity_controller` está activo:
```bash
ros2 control list_controllers
```
Debe aparecer como `active`. Si está `inactive`, activarlo:
```bash
ros2 control switch_controllers --activate joint_group_velocity_controller
```
