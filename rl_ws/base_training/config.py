#!/usr/bin/env python3
"""
config.py — TODOS los parametros ajustables del pipeline base_training, en UN
solo lugar. Si quieres tunear algo (mision, reward, limites de comando, rampas
de actuadores, red o hiperparametros de PPO), se cambia AQUI y nada mas.

Quien consume cada seccion:
  base_env.py             → mision, escalas de comando, pesos de reward,
                            geometria de flippers, lookahead, OBS/ACT
  robot_control.py        → cinematica diferencial + limites de rampa AVR446
  ../global_navigator.py  → planeacion A* + seguimiento vortex APF
  mujoco_sim_base.py      → rutas del modelo XML, decimacion, brazo, spawn
  ppo.py                  → arquitectura de la red y limites del update
  train_fast.py           → hiperparametros de entrenamiento y checkpoints
  ../base_ros_env.py      → mismas escalas/mision/reward + seccion ROS2
  ../mujoco_sim_rosbridge.py → rampas, spawn, SIM_SPEEDUP, brazo (seccion ROS2)
  ../train_base.py        → hiperparametros PPO + ROS_* (recoleccion lenta)

OJO: estos valores definen la TAREA y la DINAMICA que ve la politica. Cambiar
OBS_DIM/ACT_DIM (p.ej. via N_LOOKAHEAD) invalida checkpoints viejos; cambiar
rampas/limites de comando cambia la dinamica y abre gap con el despliegue
(el bridge ROS debe usar los mismos valores).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Rutas (absolutas, no dependen del CWD) ───────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))            # .../rl_ws/base_training
ROOT     = os.path.dirname(os.path.dirname(_HERE))               # .../aesir_rl
RL_WS    = os.path.abspath(os.path.join(_HERE, ".."))            # .../rl_ws
NAV_JSON = os.path.join(RL_WS, "obstacles.json")
_MODELS  = os.path.join(ROOT, "models")
_FULL    = os.path.join(_MODELS, "aesir_complete.xml")
_ROBOT   = os.path.join(_MODELS, "aesir_mujoco_robot.xml")
XML_PATH = _FULL if os.path.exists(_FULL) else _ROBOT
CHECKPOINT_DIR = Path(ROOT) / "checkpoints_base"

# ── Parametros Heat_Map ──────────────────────────────────────

USE_HEATMAP = True
OCTOMAP_BT_PATH = os.path.join(RL_WS, "occupied_map.bt")
OCTOMAP_RESOLUTION = 0.05
HEATMAP_RADIUS_M = 1.0
HEATMAP_PIXELS = 64
SAVE_HEATMAP_DEBUG = False

# ── Velocidad de la base: DOS pares, no confundir ────────────────────────────
# Hay dos conceptos distintos porque la oruga DESLIZA (skid-steer): lo que se
# COMANDA no es lo que el robot LOGRA.
#
#   1) V_MAX / W_MAX  = ESCALA DE COMANDO (setpoint tipo cmd_vel).
#      accion=1 -> se pide este twist. set_base_twist lo pasa a velocidad de
#      rueda con el modelo SIN deslizamiento (rueda = v/R avanzar, w*L/2/R girar)
#      y MuJoCo lo ejecuta. NO es la velocidad que el robot alcanza: por el
#      deslizamiento (sobre todo al girar) el robot logra bastante MENOS.
#      Igual que un cmd_vel real: es una peticion, no una garantia.
#      Estan calibrados para que accion=1 mueva la rueda cerca del ctrlrange
#      (6.28 rad/s del XML) -> el robot llega a ~90% de su tope fisico.
#         Lo usa: _apply_action / publish_action (el COMANDO).
#
#   2) V_REF / W_REF  = VELOCIDAD REAL alcanzable (medida a fondo en sim).
#      Es lo que el chasis DE VERDAD alcanza a fondo. Lo usa el REWARD (v_des,
#      normalizacion de accel/backward) porque el reward compara contra la
#      velocidad MEDIDA (fb["twist"]) -> tiene que estar en m/s y rad/s reales.
#         Lo usa: compute_reward (la REFERENCIA).
#
# Medido en sim (accion=1 -> velocidad real):
#   V_MAX=0.85 -> 0.43 m/s   (90% de 0.48 max real)   | V_REF=0.48
#   W_MAX=5.0  -> 0.70 rad/s (92% de 0.76 max real)   | W_REF=0.76
# (Con W_MAX=0.68 el robot giraba a 0.004 rad/s: por eso los nombres _MPS/_RADPS
#  de V_MAX/W_MAX ENGANAN -- son setpoint de comando, no velocidad lograda.)
# Si cambias la fisica (forcerange/ruedas en el XML) hay que re-medir AMBOS pares.
V_MAX_MPS   = 0.85     # COMANDO linear.x  a accion=1 (setpoint, no velocidad lograda)
W_MAX_RADPS = 5.0      # COMANDO angular.z a accion=1 (setpoint, no velocidad lograda)
V_REF_MPS   = 0.48     # REAL: velocidad lineal maxima medida  (referencia del reward)
W_REF_RADPS = 0.76     # REAL: velocidad angular maxima medida (referencia del reward)
FLIPPER_MAX = 3.1416   # rad a flip = 1
# Limite de recorrido de los flippers (por software)
FLIPPER_MIN_RAD = -1.3
FLIPPER_MAX_RAD = 3.14159

# ── Mision (frame de aesir_complete.xml == frame de obstacles.json) ──────────
START_XY: Tuple[float, float] = (-1.5, 3.5)
GOAL_XY:  Optional[Tuple[float, float]] = None   # None -> ultimo pallet del JSON
SPAWN_Z   = 0.20
FINISH_DIST       = 0.10
EPISODE_MAX_STEPS = 10000
SPAWN_XY_RANGE = 8.0
GOAL_XY_RANGE  = 9.0

PLATFORM_HALF_EXTENT = 10.0
USE_VIRTUAL_OBSTACLE = True
# plan_platform_route_with_obstacle elige PRIMERO donde va el obstaculo y
# arma la ruta alrededor (nunca al reves) -- el tamano sale de la distancia
# real entre los dos waypoints que terminan bordeando el hueco abierto.
VIRTUAL_OBSTACLE_HALF_SIZE     = 0.3    # media-arista MAXIMA (m); se achica si el hueco es mas chico
VIRTUAL_OBSTACLE_MIN_HALF_SIZE = 0.10   # media-arista MINIMA; por debajo de esto se descarta el obstaculo
# Margen (m) obligatorio entre la caja y cada waypoint vecino. Con REP_RANGE=0.1
# (corto) y el limite de giro del robot, 0.15 dejaba muy poco margen de
# reaccion y el vortex podia rozar una esquina en aproximaciones casi
# tangenciales a un lado plano (ver tests/plot_path_platform.py). Con 0.30 +
# el fix de direccion de repulsion (borde real del box, no su centro) da
# 0/596 rozamientos en bateria de prueba.
VIRTUAL_OBSTACLE_CLEARANCE     = 0.30
VIRTUAL_OBSTACLE_MAX_SKIP      = 3      # cuantos waypoints vecinos se pueden saltar agrandando el hueco
VIRTUAL_OBSTACLE_OFFSET_FRAC   = (0.2, 0.6)   # offset lateral, como fraccion del half_size del obstaculo

VIRTUAL_OBSTACLE_HEIGHT_HALF = 0.3     
PLATFORM_SURFACE_Z            = 0.12    

# ── Pesos de reward ──────────────────────────────────────────────────────────
W_DIRECTION    = 0.5     # encarar al objetivo (cos Δθ)
W_VELOCITY     = 0.5     # igualar la velocidad forward objetivo
WP_BONUS       = 200.0   # bonus al cruzar un waypoint
TIME_PENALTY   = 0.1
FALL_PENALTY   = 250.0
OBSTACLE_PENALTY = 1.0
STUCK_MAX      = 1.0
ENERGY_W       = 1e-8
FLIPPER_JERK_W = 1.0
TILT_W         = 5.0
FLIPPER_COLLISION_W = 50.0
# Castigo por aceleraciones fuertes del chasis (cuidar la integridad del robot
# en terreno dificil).
ACCEL_W        = 1.5
ACCEL_DEADZONE = 0.3
# Velocidad deseada = DISTANCIA al punto-guia del vortex (lejos -> rapido, cerca
# -> lento). Se premia alcanzarla encarando la guia; retroceder se castiga.
GUIDE_SPEED_SCALE = 1.0   # [m] distancia del guia que ya pide velocidad plena V_MAX
BACKWARD_W        = 4.0   # castigo por retroceder (x fraccion de V_MAX en reversa)

# ── Geometria de flippers (para detectar auto-colision desde los angulos) ────
FLIPPER_MOUNTS = np.array([
    [ 0.24,  0.274, 0.06],   # flipper_1  (frente-izq)
    [ 0.24, -0.274, 0.06],   # flipper_2  (frente-der)
    [-0.24,  0.274, 0.06],   # flipper_3  (atras-izq)
    [-0.24, -0.274, 0.06],   # flipper_4  (atras-der)
], dtype=np.float64)
# flipper_1,2 (frente) eje (0,1,0) -> +1 ; flipper_3,4 (atras) eje (0,-1,0) -> -1
FLIPPER_AXIS_SIGN      = np.array([1.0, 1.0, -1.0, -1.0])
FLIPPER_L              = 0.35
FLIPPER_COLLISION_DIST = 0.13

# ── Lookahead de la trayectoria (puntos futuros del vortex que ve la politica)
N_LOOKAHEAD    = 5      # nº de puntos futuros del rollout del vortex
LOOKAHEAD_STEP = 0.25   # avance (m) del muestreo de la ruta por punto

# ── Tamaños expuestos a la politica ──────────────────────────────────────────
# Estado (19) = guia_vortex(3) + waypoint_crudo(3) + twist[v_fwd,ω](2) +
#   flipper_qpos(4) + flipper_qvel(4) + gravedad_en_cuerpo(3)
# + lookahead(3*N) + heatmap(HEATMAP_PIXELS**2).
# Nota: se quito v_lat (diferencial no strafea) y upright(1) se cambio por la
# gravedad en cuerpo(3) para captar pitch/roll en pendientes.
OBS_DIM = 19 + 3*N_LOOKAHEAD + HEATMAP_PIXELS**2
ACT_DIM = 6    # v, ω, flipper×4

# ── Navegacion global: planeacion A* sobre la zona segura de pallets ─────────
ROBOT_RADIUS         = 0.30   # radio con el que se erosiona la zona transitable
GAP_BRIDGE_DISTANCE  = 0.15   # huecos entre pallets menores a esto se "puentean"
GRID_RESOLUTION      = 0.05   # celda (m) de la grilla A*
CORNER_DOT_THRESHOLD = 0.99   # giro se conserva como esquina si dot(v1,v2) < esto
MAX_WAYPOINT_DIST    = 0.50   # espaciado maximo (m) entre waypoints en rectas

# ── Navegacion global: seguimiento (vortex APF + guia) ───────────────────────
REACH_DIST     = 0.10   # distancia a la que un waypoint cuenta como alcanzado
MAX_GUIDE_DIST = 5.0    # normalizacion de la distancia en la obs (dist_norm)
ATT_GAIN     = 1.0      # grado de atraccion (magnitud maxima del pull al waypoint)
ATT_RANGE    = 0.1      # distancia sobre la que la atraccion crece de 0 a ATT_GAIN
REP_GAIN     = 1.0      # grado de repulsion (magnitud maxima del push por fuente)
REP_RANGE    = 0.1      # distancia (del cuerpo del robot) a la que empieza a repeler
ROBOT_HALF   = 0.30     # media anchura del robot (su cuerpo, no un punto)
SWIRL        = 1.0      # peso de la componente tangencial (0 = APF plano, 1 = vortex)
MIN_PROGRESS = 0.1      # fraccion de la atraccion que SIEMPRE sobrevive (garantia
                        # anti-minimo-local: la repulsion nunca frena mas del
                        # (1-MIN_PROGRESS) del avance hacia el waypoint)

# ── Cinematica diferencial de las orugas (igual que el bridge) ───────────────
TRACK_SEPARATION = 0.36
WHEEL_RADIUS     = 0.15

# ── Limites AVR446 (rampa trapezoidal, identicos al bridge de despliegue) ────
VELOCITY_MAX_ACCEL = 15.0    # rad/s^2 de los tracks (velocidad)
FLIPPER_MAX_VEL    = 3.0     # rad/s   de los flippers (posicion)
FLIPPER_MAX_ACCEL  = 10.0    # rad/s^2 de los flippers

# ── Nombres de joints/actuadores en el XML ───────────────────────────────────
FLIPPER_JOINTS = ["flipper_1_joint", "flipper_2_joint",
                  "flipper_3_joint", "flipper_4_joint"]
FLIPPER_ACTS   = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
ARM_JOINTS     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS  = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
                  "pos_joint_4", "pos_joint_5", "pos_joint_6"]
ARM_REST_POSE  = {"joint_1": 0.0, "joint_2": -2.86234, "joint_3": 2.86234,
                  "joint_4": -1.5708, "joint_5": -1.5708, "joint_6": 1.5708}

# ── Simulacion MuJoCo directa ────────────────────────────────────────────────
CONTROL_DECIMATION = 25   # 25 * timestep(0.002) = 0.05 s por paso de control (20 Hz sim)
SPAWN_SETTLE_STEPS = 50

# ── Backend ROS2 (mujoco_sim_rosbridge.py + base_ros_env.py) ─────────────────
CONTROL_HZ  = 20.0    # frecuencia del lazo de control RL sobre el bridge
PHYSICS_HZ  = 100.0   # tick del timer del bridge (debe igualar ros2_controllers.yaml)
SIM_SPEEDUP = 1.0     # Factor de aceleracion respecto a TIEMPO REAL.
SPAWN_YAW   = 0.0     # yaw inicial del chasis en el reset del bridge
# Limites de rampa AVR446 del BRAZO en el bridge (rad/s, rad/s^2 por joint).
# Los flippers usan FLIPPER_MAX_VEL / FLIPPER_MAX_ACCEL (arriba).
ARM_JOINT_LIMITS = {
    "joint_1": dict(max_vel=6.0, max_accel=10.0),
    "joint_2": dict(max_vel=3.0, max_accel=10.0),
    "joint_3": dict(max_vel=6.0, max_accel=10.0),
    "joint_4": dict(max_vel=9.0, max_accel=10.0),
    "joint_5": dict(max_vel=9.0, max_accel=10.0),
    "joint_6": dict(max_vel=9.0, max_accel=10.0),
}

# ── Red (actor-critic) ───────────────────────────────────────────────────────
HIDDEN        = 256      # neuronas por capa del trunk (2 capas Tanh)
LOG_STD_INIT  = -0.5     # log-std inicial de la gaussiana de la politica
LOG_STD_MIN   = -5.0     # clamp del log-std aprendible
LOG_STD_MAX   = 1.0
MAX_GRAD_NORM = 0.5      # clip de norma de gradiente en el update

# ── Hiperparametros PPO / entrenamiento ──────────────────────────────────────
N_ENVS        = 14       # envs MuJoCo en paralelo (threads)
STEPS_PER_ENV = 512      # pasos por env por rollout -> batch de T*N
ITERS         = 10000     # iteraciones de entrenamiento
PPO_EPOCHS    = 10       # pasadas sobre el batch por update
BATCH_SIZE    = 1024     # tamaño de minibatch
GAMMA         = 0.99     # descuento
GAE_LAMBDA    = 0.95     # lambda de GAE
CLIP          = 0.2      # clip-surrogate de PPO
VF_COEF       = 0.5      # peso del value loss
ENT_COEF      = 0.005    # bonus de entropia (exploracion)
LR            = 3e-4     # learning rate (Adam)
SAVE_EVERY    = 50       # iters entre checkpoints numerados
DEVICE        = "auto"   # "auto" -> cuda si hay, si no cpu
WANDB_PROJECT = "AIDL-PPO-AESIR-BASE-FAST"

# Entrenamiento sobre el backend ROS (train_base.py): la recoleccion es ~50x
# mas lenta (bridge en tiempo semi-real), por eso iteraciones/batches menores.
ROS_STEPS_PER_ITER = 1024
ROS_ITERS          = 500
ROS_BATCH_SIZE     = 256
ROS_SAVE_EVERY     = 25
WANDB_PROJECT_ROS  = "AIDL-PPO-AESIR-BASE"


