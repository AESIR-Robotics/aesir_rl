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
_MODELS  = os.path.join(ROOT, "models")
CHECKPOINT_DIR = Path(ROOT) / "checkpoints_base"

# ── PISTAS (registro multi-pista) ────────────────────────────────────────────
# Cada pista define: su XML completo (robot + terreno), sus octomaps (.bt, para
# el heatmap) en rl_ws/tracks/<pista>/, la altura de spawn (el terreno cambia
# de techo) y su tipo de navegacion:
#   kind="platform" -> mision procedural (spawn/meta random, ruta directa +
#                      obstaculo virtual, zona segura = cuadrado erosionado)
#   kind="pallets"  -> mision clasica del JSON (A* sobre pallets/sticks,
#                      spawn/meta fijos) — la implementacion original, intacta.
# Los .bt se regeneran con: python3 rl_ws/octomap_from_mujoco.py --track <pista>
TRACKS_DIR = os.path.join(RL_WS, "tracks")
TRACK_DEFS = {
    "flat": dict(
        kind="platform",
        xml=os.path.join(_MODELS, "aesir_complete_flat.xml"),
        bt=os.path.join(TRACKS_DIR, "flat", "occupied_map.bt"),
        spawn_z=0.20, settle_steps=50),          # techo terreno z=0.12
    "steps": dict(
        kind="platform",
        xml=os.path.join(_MODELS, "aesir_complete_steps.xml"),
        bt=os.path.join(TRACKS_DIR, "steps", "occupied_map.bt"),
        spawn_z=0.55, settle_steps=150),         # escalones hasta z=0.42
    "steps1m": dict(
        kind="platform",
        xml=os.path.join(_MODELS, "aesir_complete_steps1m.xml"),
        bt=os.path.join(TRACKS_DIR, "steps1m", "occupied_map.bt"),
        spawn_z=0.55, settle_steps=150),         # escalones de 1x1 m hasta z=0.42
    "ramps": dict(
        kind="platform",
        xml=os.path.join(_MODELS, "aesir_complete_ramps.xml"),
        bt=os.path.join(TRACKS_DIR, "ramps", "occupied_map.bt"),
        spawn_z=0.45, settle_steps=150),         # rampas hasta z=0.28
    "pallets": dict(
        kind="pallets",
        xml=os.path.join(_MODELS, "aesir_complete_pallets.xml"),
        bt=os.path.join(TRACKS_DIR, "pallets", "occupied_map.bt"),
        nav_json=os.path.join(TRACKS_DIR, "pallets", "obstacles.json"),
        spawn_z=0.20, settle_steps=50),          # pista original de pallets
    # Maze: laberinto de paredes (140 cajas "maze_collision_*" en maze.xml) sobre
    # el plano "maze_floor" a z=0. OJO con las tres claves propias:
    #   spawn_xy/spawn_yaw — la mision NO es procedural como en platform: el robot
    #     SIEMPRE nace en la entrada del laberinto (medido libre, holgura 0.40 m)
    #     mirando al corredor abierto. START_XY global no sirve: es la mision de
    #     pallets, que en el frame del maze cae en otro lado.
    #   fall_z_min — el piso caminable del maze es z=0, no una plataforma elevada
    #     (flat/steps/ramps tienen el suyo a z>=0.12). Medido en sim, el chasis
    #     REPOSA a z=0.036, o sea por DEBAJO del FALL_Z_MIN global de 0.10: con el
    #     default el robot nace "caido" y is_fallen() mata el episodio en el paso
    #     1, siempre. -0.20 queda debajo del piso caminable y encima del
    #     "fatal_floor" (z=-0.5), que sigue siendo la red de seguridad real.
    #   finish_dist — 0.10 m es inalcanzable de forma fiable para un skid-steer en
    #     pasillos de ~0.9 m; 0.25 es "recoger el waypoint" sin pedir puntería.
    "maze": dict(
        kind="maze",
        xml=os.path.join(_MODELS, "aesir_complete_maze.xml"),
        bt=os.path.join(TRACKS_DIR, "maze", "occupied_map.bt"),
        # spawn_yaw=0 (mirando a +x) NO es arbitrario: medido sobre 300 metas
        # sorteadas, el 100% de las rutas sale de la entrada hacia +x (rumbo
        # medio +3°). Con yaw=pi el robot arrancaba 177° girado en TODOS los
        # episodios, y como el reward de progreso es ciego a la orientacion le
        # salia mas barato conducir en REVERSA que girar -> aprendia al reves.
        spawn_xy=(0.5, -0.4), spawn_yaw=0.0,
        spawn_z=0.12, settle_steps=100,
        fall_z_min=-0.20, finish_dist=0.25),
}
# Pistas ACTIVAS para entrenar: una o varias — VecMujocoEnv las reparte entre
# los envs (round-robin). Ej: ["flat"], ["steps"], ["flat","steps","ramps"].
ACTIVE_TRACKS = ["ramps", "ramps", "steps1m", "flat"]

# Compatibilidad: el "mundo por default" (bridge ROS, scripts single-track) es
# la PRIMERA pista activa. Para probar otra pista por ROS, cambia el orden.
XML_PATH        = TRACK_DEFS[ACTIVE_TRACKS[0]]["xml"]
OCTOMAP_BT_PATH = TRACK_DEFS[ACTIVE_TRACKS[0]]["bt"]
NAV_JSON        = TRACK_DEFS["pallets"]["nav_json"]   # mapa de navegacion pallets

# ── Parametros Heat_Map ──────────────────────────────────────
USE_HEATMAP = True
OCTOMAP_RESOLUTION = 0.05
HEATMAP_RADIUS_M = 1.0
HEATMAP_PIXELS = 64
SAVE_HEATMAP_DEBUG = False
HEATMAP_Z_RANGE_M = 1.0


# ── Point cloud del octomap hacia MoveIt (brazo) ─────────────────────────────
OCTOMAP_POINTCLOUD_TOPIC = "/stepfield/points"
OCTOMAP_POINTCLOUD_FRAME = "map"
POINTCLOUD_RADIUS_M = 1.25
POINTCLOUD_PUBLISH_HZ = 2.0
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

# Limite de recorrido de los flippers (por software)
FLIPPER_MIN_RAD = -1.1
FLIPPER_MAX_RAD = 2.64159
CONTROL_FLIPPERS = True

# ── Mision (frame de aesir_complete.xml == frame de obstacles.json) ──────────
START_XY: Tuple[float, float] = (0.5, -0.4)
GOAL_XY:  Optional[Tuple[float, float]] = None   # None -> ultimo pallet del JSON
SPAWN_Z   = 0.40          # (feat/maze)
FINISH_DIST       = 0.20  # (base) default global; cada pista puede pisarlo con
                          # TRACK_DEFS[...]["finish_dist"] -> fb["finish_dist"]
EPISODE_MAX_STEPS = 10000
SPAWN_XY_RANGE = 8.0
GOAL_XY_RANGE  = 9.0

PLATFORM_HALF_EXTENT = 10.0

# ── Maze: mapa de ocupacion y muestreo del waypoint ──────────────────────────
# El area navegable NO se declara a mano (antes eran dos rangos XY fijos que
# caian mayormente DENTRO de paredes): global_navigator.get_maze_map() la deriva
# del propio XML — rasteriza las cajas de colision, infla por ROBOT_RADIUS y se
# queda con la componente conexa de la entrada (medido: 27.8 m2 navegables). Asi
# el waypoint SIEMPRE cae en pasillo libre y alcanzable, y si el maze cambia el
# mapa se recalcula solo.
MAZE_ROBOT_TOP_M     = 0.45   # alto del robot: una pared que EMPIEZA por encima
                              # de esto es un dintel y se pasa por debajo (68 de
                              # las 159 cajas del maze son dinteles) -> no bloquea
MAZE_GOAL_MIN_DIST_M = 2.0    # separacion minima robot->waypoint nuevo (si no,
                              # sortearia metas pegadas y el episodio seria trivial)
USE_VIRTUAL_OBSTACLE = True
VIRTUAL_OBSTACLE_HALF_SIZE     = 0.3    # media-arista MAXIMA (m); se achica si el hueco es mas chico
VIRTUAL_OBSTACLE_MIN_HALF_SIZE = 0.10   # media-arista MINIMA; por debajo de esto se descarta el obstaculo
VIRTUAL_OBSTACLE_OFFSET_FRAC   = (0.2, 0.6)   # offset lateral, como fraccion del half_size del obstaculo

VIRTUAL_OBSTACLE_HEIGHT_HALF = 0.3     
PLATFORM_SURFACE_Z            = 0.12    

# ── Pesos de reward ──────────────────────────────────────────────────────────
W_DIRECTION    = 0.7     # encarar al objetivo (cos Δθ)
W_VELOCITY     = 0.7     # igualar la velocidad forward objetivo
WP_BONUS       = 200.0   # bonus al cruzar un waypoint
GOAL_BONUS     = 1000.0
TIME_PENALTY   = 0.1
FALL_PENALTY   = 250.0
FALL_UPRIGHT_MIN = 0.20   # R[2,2]=cos(inclinacion); <0.20 -> volcado (~78°)
FALL_Z_MIN       = 0.10   # altura del chasis (m) por debajo de la cual se cayo
OBSTACLE_PENALTY = 1.0
STUCK_MAX      = 1.0
STUCK_ANGULAR_THRESH_RAD = 0.02
ENERGY_W       = 1e-8
FLIPPER_JERK_W = 0.02
FLIPPER_COLLISION_W = 10.0
FLIPPER_TERRAIN_W           = 5.0    # medido contra flipper_jerk_pen (flip_qvel real)
FLIPPER_TERRAIN_MIN_STEP_M  = 0.10   # |desnivel| bajo esto = ruido de piso, no un borde real
FLIPPER_TERRAIN_MAX_CLIMB_M = 0.45   # techo heuristico de lo "trepable" (pared arriba)
FLIPPER_TERRAIN_MAX_DESCENT_M = 0.35
FLIPPER_TERRAIN_SAMPLES     = 8      # puntos escaneados hasta FLIPPER_L buscando el borde
# Castigo por aceleraciones fuertes del chasis (cuidar la integridad del robot
# en terreno dificil).
ACCEL_W        = 3.0
ACCEL_DEADZONE = 0.3
# Velocidad deseada = DISTANCIA al punto-guia del vortex (lejos -> rapido, cerca
# -> lento). Se premia alcanzarla encarando la guia; retroceder se castiga.
GUIDE_SPEED_SCALE = 1.0   # [m] distancia del guia que ya pide velocidad plena V_MAX
BACKWARD_W        = 3.0   # castigo por retroceder (x fraccion de V_MAX en reversa)

# ── Terminacion por atasco (sin progreso hacia la meta) ──────────────────────
STUCK_TIMEOUT_STEPS = 300     # pasos sin progreso -> termina
STUCK_NO_PROGRESS_M = 0.05    # mejora minima (m) hacia la meta que reinicia el contador


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
# Accion (6): v, ω, flipper×4. Distribucion HIBRIDA (ver CNNActorCritic/
# MLPActorCritic): v,ω Gaussiana (recortada a [-1,1]); flipper×4 Beta (soporte
# YA en [0,1], sin recorte).
#   [0] v      linear.x  (escala V_MAX)
#   [1] ω      angular.z (escala W_MAX)
#   [2:6] flipper×4  muestra Beta en [0,1], reescalada afin a
#         [FLIPPER_MIN_RAD, FLIPPER_MAX_RAD] (ver base_env.flipper_targets)
#
# Los flippers se comandan SIEMPRE desde [2:6]. Hubo una 7a dimension (un gate
# Bernoulli que los mandaba a reposo ignorando [2:6]), se elimino. Ver docs/gate_flippers.md.
ACT_DIM = 6
FLIPPER_HOME_RAD = 0.0   # pose de reposo de los flippers (fase 1 / referencia del reward de terreno)

# ── Navegacion global: planeacion A* sobre la zona segura de pallets ─────────
ROBOT_RADIUS         = 0.35   # radio con el que se erosiona la zona transitable
GAP_BRIDGE_DISTANCE  = 0.15   # huecos entre pallets menores a esto se "puentean"
GRID_RESOLUTION      = 0.05   # celda (m) de la grilla A*
# Las pistas platform son 20x20 m con UN obstaculo de 60 cm: a 0.05 serian
# 401x401 celdas (4x el maze) para una precision que ahi no compra nada.
PLATFORM_GRID_RESOLUTION = 0.15
CORNER_DOT_THRESHOLD = 0.99   # giro se conserva como esquina si dot(v1,v2) < esto
MAX_WAYPOINT_DIST    = 0.50   # espaciado maximo (m) entre waypoints en rectas

# ── Huella RECTANGULAR del robot (para planear con orientacion) ──────────────
# Medidas REALES del robot: 67.4 cm (x) x 60 cm (y), con el BRAZO RECOGIDO
# (plegado dentro del perimetro: no cuenta para navegar) y los flippers plegados.
#
# Un solo radio no puede describir este chasis, porque avanzar y girar piden
# cosas distintas:
#     avanzar recto  -> holgura lateral >= ROBOT_HALF_WIDTH  (0.300 m)
#     girar en sitio -> holgura         >= hypot(hw, hl)     (0.451 m)
# Con ROBOT_RADIUS=0.35 (entre los dos) el A* daba rutas transitables pero con
# giros en celdas donde el chasis no cabe rotando; subirlo a 0.40 fragmentaba el
# laberinto y dejaba el spawn en un bolsillo de 0.9 m2. Ver MazeMap.
SEGURY_DIST = 0.0                       # margen extra de seguridad, por si acaso
ROBOT_HALF_WIDTH  = 0.290 + SEGURY_DIST # media-anchura (eje y):  60.0 cm / 2
ROBOT_HALF_LENGTH = 0.327 + SEGURY_DIST # media-longitud (eje x): 67.4 cm / 2
# 8 rumbos = 45 grados, y cada uno apunta a un vecino de la rejilla, asi que
# TODAS las primitivas de avance son de una celda (5 cm). Con 16 los 8 rumbos
# impares (22.5, 67.5...) no caen en ningun vecino y su paso minimo pasaba a ser
# (2,1) = 11 cm; con 32 seria (5,1) = 25 cm. Ese desplazamiento minimo desigual
# rompia las rutas.
#
# Y la precision extra no compra nada: MEDIDO sobre este maze, el area donde el
# robot cabe en algun rumbo es 42.07 m2 con 8, 42.16 con 16 y 42.16 con 32 --
# de 16 a 32 la ganancia es EXACTAMENTE cero. La rejilla es de 5 cm, asi que
# rotar 11.25 grados mueve el contorno del rectangulo menos de una celda y la
# mascara rasterizada sale identica: la resolucion angular esta limitada por la
# espacial. Para afinar de verdad habria que bajar GRID_RESOLUTION a 0.025 Y
# subir los rumbos a la vez (rejilla 284x480, arbol de ~2.2M estados).
#
# Solo se avanza HACIA DELANTE: una ruta que exigiera reversa estaria peleada
# con BACKWARD_W, que castiga la velocidad hacia atras.
MAZE_N_HEADINGS   = 8         # rumbos discretos del A*
MAZE_TURN_COST    = 0.5       # coste (en celdas) por cada 360/N grados de giro
MAZE_REVERSE_COST = 2.0

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
LOG_STD_MAX   = 0.0
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

# ── SAC (train_sac.py) — pipeline PARALELO a PPO, mismo env y mismo config ──
SAC_BUFFER_SIZE   = 1_000_000 
SAC_BATCH_SIZE    = 512
SAC_LR            = 3e-4
SAC_TAU           = 0.005    # soft-update de los criticos objetivo
SAC_GAMMA         = 0.99
SAC_START_STEPS   = 5_000    # pasos con accion ALEATORIA antes de usar la red
SAC_LEARN_STARTS  = 10_000   # transiciones en buffer antes del primer update
SAC_UTD           = 1.0     # updates por paso de entorno (update-to-data).
                             # 1.0 es el canonico; 0.5 por el costo de la CNN.
SAC_LOG_STD_MIN   = -5.0
SAC_LOG_STD_MAX   = 2.0

SAC_ACT_DIM = 6
SAC_REWARD_SCALE = 0.01
SAC_TARGET_ENTROPY = -2.0
WANDB_PROJECT_SAC = "AIDL-SAC-AESIR-BASE-FAST"

# Entrenamiento sobre el backend ROS (train_base.py): la recoleccion es ~50x
# mas lenta (bridge en tiempo semi-real), por eso iteraciones/batches menores.
ROS_STEPS_PER_ITER = 1024
ROS_ITERS          = 500
ROS_BATCH_SIZE     = 256
ROS_SAVE_EVERY     = 25
WANDB_PROJECT_ROS  = "AIDL-PPO-AESIR-BASE"
