# Arquitectura del Agente PPO y Entorno RL

Este documento explica de forma concisa cómo están estructurados los paquetes `rl_agent_env` y `rl_trainer`, además de documentar las acciones de los dos agentes concurrentes diseñados para entrenar y operar el robot Aesir.

## Visión General de los Paquetes

1. **`rl_agent_env`**: 
   Actúa como el puente o interfaz (Wrapper tipo Gym) que integra la simulación asíncrona de ROS 2 con el ciclo síncrono que requiere el algoritmo de RL (PPO). En proyectos básicos o de pruebas unitarias incluye un script `rl_env.py` (para 8 acciones unificadas) apoyado en `ros_bridge.py` para intercambiar tópicos de manera thread-safe.

2. **`rl_trainer`**: 
   Contiene el bucle de entrenamiento, el manejador de memoria (buffer) y la lógica matemática del modelo PPO (política y valor). Implementa una arquitectura **Multi-Agente (Dual Agent)** mediante la clase `HybridAesirEnv` en `train_ppo.py`. Este entorno híbrido controla la física del robot de manera síncrona en MuJoCo, mientras envía los comandos equivalentes asíncronamente a los controladores ROS 2.

## Arquitectura de Multi-Agente (Dual Agent)

Ambos agentes reciben **la misma observación multimodal**:
- Imágenes de tres cámaras (Gripper, OAK-D, Trasera).
- Lecturas de LiDAR (7 rayos).
- Estados de articulaciones (posiciones y velocidades de los 26 actuadores del robot).

Sin embargo, **dividen el control de los actuadores** de la siguiente manera:

### Agente A (Manipulador y Gripper)
Controla la sección superior del robot (el brazo).
* **Dimensión de Acción**: 8 valores continuos normalizados en `[-1, 1]`.
  * `[0:6]` controlan las 6 articulaciones del brazo `pos_joint_1` a `pos_joint_6`.
  * `[6:8]` controlan los dedos del gripper `pos_left_finger` y `pos_right_finger`.
* **Cómo acciona (Directo vs MoveIt)**:
  * **MuJoCo (Directo)**: Escribe posiciones directamente sobre las articulaciones físicas del brazo y el gripper.
  * **ROS 2**: El script calcula la velocidad de cada articulación (`(posición_objetivo - posición_actual) / dt`) y la publica al tópico `/joint_group_velocity_controller/commands`. Desde allí ros2_control y, opcionalmente, la planificación reactiva (MoveIt / Servo) toman procedencia.

### Agente B (Base y Flippers)
Controla la locomoción y las orugas secundarias.
* **Dimensión de Acción**: 14 valores continuos normalizados en `[-1, 1]`.
  * `[0]` Velocidad lineal hacia el frente (`v_lin`).
  * `[1]` Velocidad angular de la base (`omega`).
  * `[2:6]` Posiciones articulares de los 4 flippers.
  * `[6:14]` Velocidades rotacionales de las orugas (wheels) integradas en los flippers.
* **Cómo acciona (Directo vs ROS)**:
  * **MuJoCo (Directo)**: Traduce internamente `(v_lin, omega)` a comandos de tracción individual (Differential Drive) de las 6 ruedas principales. Las posiciones y velocidades de los flippers se inyectan a motores nativos de MuJoCo.
  * **ROS 2**: Escala `(v_lin, omega)` según los límites físicos del chasis (por defecto `MAX_LIN_VEL = 0.5`, `MAX_ANG_VEL = 1.0`) y publica el comando directamente por medio de un mensaje `Twist` al tópico `/diff_drive_controller/cmd_vel`. Aquí **no** hay integración de MoveIt, es envío directo para comando del DiffDrive o controlador análogo.

## Pauta para continuar adaptando el PPO

Si decides incorporar nuevas lógicas de reward, cambiar los joints disponibles o agregar sensores para el PPO:
1. **Dimensiones de matriz**: Revisa siempre constantes como `AGENT_A_ACT_DIM` y `AGENT_B_ACT_DIM` en `rl_trainer/train_ppo.py` tras tus cambios.
2. **Recompensas Compartidas vs Divididas**: Actualmente cada agente empuja funciones de reward enfocadas (penalización por obstáculos para la base, supervivencia). Considera mantener los rewards ortogonales si notas que los agentes compiten o si observas fallas en el Agente A arrastradas por los movimientos del Agente B.
3. **Escalamiento Físico**: Al agregar un "acción" nueva de RL (que siempre saldrá de la red neuronal en `[-1, 1]`), verifica implementar `_scale()` u operadores equivalentes en `_apply_actions()` para evitar velocidades inestables que vuelquen la simulación.