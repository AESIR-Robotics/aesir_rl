"""
boton_env.py — Env MuJoCo para presionar botones con la GARRA, brazo solo.

Tarea
=====
El robot esta quieto (base congelada) frente a la TORRE de botones de
models/torre_botones.xml — la replica actuable de la torre que hay en el maze
(pares hub_N/door_N, que alli son cajas rigidas sin un solo joint).

Cada episodio se sortea un boton de `targets` (por defecto TARGETS_ALCANZABLES
= los 4 que se alcanzan con la base quieta); la politica recibe su posicion
RELATIVA a arm_base_link y tiene que llevar la punta de la garra hasta el y
hundirlo PRESS_THRESH_FRAC de su recorrido.

La torre es una CUPULA: cuatro botones a 45 grados hacia afuera-arriba (+-X,
+-Y) y uno vertical en la cuspide. Cada boton se hunde a lo largo de SU PROPIA
normal, no de una comun — por eso la observacion incluye esa normal y el
controlador la consulta con button_normal(). Las normales se LEEN del modelo
(xmat del cuerpo del boton), no se escriben a mano: si se mueve o se gira la
torre, siguen siendo correctas.

Observacion — 32 valores
========================
   0:6   qpos del brazo (joint_1..6)
   6:12  qvel del brazo
  12:14  qpos de los dedos          14:16  qvel de los dedos
  16:19  POSICION DEL BOTON relativa a arm_base_link   <- el dato de la tarea
  19:22  vector TCP -> boton, en arm_base_link
  22:25  NORMAL del boton en arm_base_link (su eje de presion)
  25     distancia TCP -> boton
  26     fraccion hundida del boton (0 suelto, 1 a fondo)
  27     condicionamiento del jacobiano / 100, saturado en 2. Es la metrica de
         singularidad de MoveIt Servo: >=1 significa que Servo pararia el brazo
         en el robot real. Va en la observacion porque se CASTIGA (ver
         W_SINGULAR): castigar algo que la politica no puede ver no le deja
         forma de evitarlo.
  (Ya NO hay one-hot del boton. Ataba la red a la torre de 5 botones: con el,
  la politica podia memorizar cinco casos en vez de resolver "ve a estas
  coordenadas y empuja en esta direccion". La posicion relativa y la normal
  describen el objetivo por completo, sea de la torre o suelto.)

Todo lo que describe al boton va en el frame arm_base_link, que es el frame en
el que MoveIt Servo recibe los comandos: la politica ve y actua en el mismo
sistema de referencia.

Accion — 7 valores en [-1,1], compatible con MoveIt Servo
=========================================================
  [0..2]  velocidad lineal  del TCP (vx,vy,vz) en el frame arm_base_link
  [3..5]  velocidad angular del TCP (wx,wy,wz) en el frame arm_base_link
  [6]     apertura de la garra (-1 = abierta, +1 = cerrada)

Es el mismo contrato que ArmServoEnv: twist cartesiano unitless en
arm_base_link, que es exactamente lo que espera /servo_node/delta_twist_cmds
con robot_link_command_frame=arm_base_link y command_in_type=unitless. Las
escalas salen de arm_env (MAX_LINEAR_VEL/MAX_ANGULAR_VEL), atadas a
scale.linear/scale.rotational de servo_params.yaml. NO se toca nada de MoveIt.

Dos correcciones frente a arm_env.ArmMuJoCoEnv
==============================================
1. PUNTO DE CONTROL. arm_env usa el cuerpo 'logitech_gripper_assembly' como
   end-effector. Ese cuerpo es la CARCASA DE LA WEBCAM: esta a 19.1 cm del punto
   medio entre las puntas de los dedos (medido). Para presionar hay que controlar
   la punta de la garra. Aqui el TCP es
       tcp = xpos[link_6] + xmat[link_6] @ (0, 0.12, 0)
   que cae a 0.00000 m del punto medio entre las puntas (medido en poses
   aleatorias) y coincide con el `tool_link` del URDF — el mismo ee_frame_name
   que usa servo_params.yaml.

2. JACOBIANO EN EL PUNTO, NO EN EL ORIGEN DEL CUERPO. arm_env llama
   mj_jacBody(link_6), que da el jacobiano del ORIGEN de link_6. MoveIt Servo
   calcula el jacobiano en ee_frame_name (tool_link). Aqui se usa mj_jac() en el
   punto TCP, que es el jacobiano de tool_link — asi la misma accion produce el
   mismo movimiento en sim y en el stack real.

Frame arm_base_link en MuJoCo
=============================
El modelo MuJoCo no tiene cuerpo 'arm_base_link' (ni 'base_link'). Del URDF
rescue_robot_v2: arm_base_link cuelga de base_link en xyz (0.12, 0, 0.06) sin
rotacion, y base_link comparte frame con 'footprint_link' de MuJoCo (verificado:
joint_1 cae en (0.12, 0, 0.2183) en ambos). Por eso:
    p_arm_base = xpos[footprint_link] + xmat[footprint_link] @ (0.12, 0, 0.06)
    R_arm_base = xmat[footprint_link]

Reward
======
El shaping de acercamiento es POTENCIAL (prev_dist - dist), no -dist. En este
repo ya se documento que un reward denso por distancia se explota oscilando
(docs/evolucion_proyecto.md, Fase 6 punto 3): avanzar y retroceder farmea
recompensa. Un termino potencial telescopa a cero en cualquier ciclo cerrado,
asi que ese exploit no existe.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco

# IK amortiguada compartida con arm_env.
from arm_env import DLS_DAMPING
from arm_env import MAX_LINEAR_VEL as SERVO_SCALE_LINEAR
from arm_env import MAX_ANGULAR_VEL as SERVO_SCALE_ANGULAR

# ──────────────────────────── Constantes ───────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
XML_PATH  = os.path.normpath(os.path.join(_HERE, "..", "models", "aesir_arm_botones.xml"))

ARM_JOINTS     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS  = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
                  "pos_joint_4", "pos_joint_5", "pos_joint_6"]
FLIPPER_JOINTS = ["flipper_1_joint", "flipper_2_joint",
                  "flipper_3_joint", "flipper_4_joint"]
FLIPPER_ACTS   = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
# Angulo de PLEGADO de los flippers. A qpos=0 los flippers apuntan hacia ARRIBA
# y sus ruedas de punta quedan a z=0.448, justo dentro del espacio de trabajo
# del brazo (que trabaja entre z=0.256 y 0.360): el brazo chocaba contra ellos
# y, con la autocolision como condicion de corte, mataba el episodio. Medido:
# era el 14 de 19 de las autocolisiones del controlador de referencia.
# A pi/2 quedan horizontales, con las puntas a z=0.083 -- fuera del camino -- y
# con CERO contactos contra el chasis o el suelo. Mas alla de 2.0 rad tocan el
# suelo y levantan el robot. Dentro de FLIPPER_MIN/MAX_RAD de config.py.
# En esta tarea la base esta quieta y los flippers no se controlan: plegarlos es
# lo que haria el robot real al pararse a manipular.
FLIPPER_STOW_RAD = 1.5708

FINGER_JOINTS  = ["left_finger_joint", "right_finger_joint"]
FINGER_ACTS    = ["pos_left_finger", "pos_right_finger"]
FINGER_CLOSED  = 0.03    # ctrlrange del XML: 0 = abierta, 0.03 = cerrada

N_BOTONES      = 5
BOTON_JOINTS   = [f"boton_{i}_slide" for i in range(N_BOTONES)]
BOTON_SITES    = [f"boton_{i}_target" for i in range(N_BOTONES)]
BOTON_BODIES   = [f"boton_{i}" for i in range(N_BOTONES)]
PRESS_TRAVEL   = 0.018   # == range del joint en torre_botones.xml

# Correspondencia con la torre del maze (models/maze.xml, torre B):
#   0 -> hub_8/door_8   (-X, 45 grados)  el que mira al robot
#   1 -> hub_11/door_11 (+Y, 45 grados)
#   2 -> hub_9/door_9   (-Y, 45 grados)
#   3 -> hub_7/door_7   (+Z, vertical)   la cuspide
#   4 -> hub_10/door_10 (+X, 45 grados)  el lado opuesto al robot
BOTON_MAZE_REF = ["door_8", "door_11", "door_9", "door_7", "door_10"]

# ── Botones SUELTOS (models/botones_sueltos.xml) ────────────────────────────
N_LIBRES      = 4
LIBRE_BODIES  = [f"libre_{i}" for i in range(N_LIBRES)]
LIBRE_JOINTS  = [f"libre_{i}_slide" for i in range(N_LIBRES)]
LIBRE_SITES   = [f"libre_{i}_target" for i in range(N_LIBRES)]
LIBRE_HEADS   = [f"libre_{i}_head" for i in range(N_LIBRES)]
# Aparcadero de los botones que no se usan en el episodio.
#
# ARRIBA, no abajo. Estaba en z=-5, pero el suelo es un PLANO, o sea un
# semiespacio infinito: todo lo que queda debajo esta DENTRO de el. El solver
# empujaba las cabezas hacia arriba por sus joints deslizantes, violando el
# limite de 18 mm hasta dejarlas colgando a z~-0.35 y a 3.7 m del origen (viajan
# en diagonal porque los ejes de los laterales van a 45 grados). Efectos:
#   - 24 de los 47 contactos de la simulacion eran basura de cuerpos aparcados
#   - violacion permanente de restricciones en cada paso
#   - los puntos de contacto se ven como circulos repartidos por el suelo
# Ademas se les quita la colision (ver _aparcar), que es lo que de verdad los
# saca de la fisica; la altura es solo para que no se vean.
PARKING       = np.array([0.0, 0.0, 30.0])

# Volumen de aparicion de un boton suelto, en coordenadas del HOMBRO (joint_1).
# Los limites salen del alcance medido del brazo (0.028-1.079 m desde el hombro)
# recortado a lo que de verdad es utilizable con la garra encarando el boton:
# la torre a 0.55 m funciona y a 0.65 m los laterales ya no se alcanzan.
# Radio MINIMO 0.46, no 0.34. Medido barriendo el volumen de aparicion: a
# 0.34-0.40 m del hombro la viabilidad es del 0% (n=8) -- el brazo no puede
# plegarse tanto manteniendo la garra encarando el boton. De 0.50 a 0.58 sube al
# 50-55%. Poner botones ahi no ensena nada: son episodios imposibles, y ya vimos
# con el boton 2 lo que cuesta detectar una tarea que no se puede completar.
LIBRE_R_MIN, LIBRE_R_MAX     = 0.46, 0.58    # m desde el hombro
LIBRE_AZIM_MAX               = 0.80          # rad (~46 grados) a cada lado de +X
# Elevacion minima por encima de la horizontal del hombro: la viabilidad sube
# con la altura (0-15 grados -> 25%, 30-43 -> 50%), y por debajo el boton acaba
# a la altura de las orugas.
LIBRE_ELEV_MIN, LIBRE_ELEV_MAX = 0.10, 0.75  # rad sobre la horizontal del hombro
# Cuanto puede desviarse la normal del boton de "mirar al hombro". 0 = de frente.
LIBRE_TILT_MAX               = 0.45          # rad (~26 grados)
# Separacion minima entre botones sueltos de un mismo episodio, para que los
# distractores no se solapen con el objetivo.
LIBRE_SEP_MIN                = 0.16          # m
# Suelo absoluto de altura. El muestreo en esfericas alrededor del hombro puede
# bajar hasta z=0.117 con elevacion negativa y radio grande, que es la altura de
# las orugas: ahi el boton nace DENTRO del robot. Medido en la auditoria: hasta
# 16 contactos robot<->boton nada mas colocarlo, lo que mataria el episodio en
# el primer paso.
LIBRE_Z_MIN                  = 0.30          # m

# Reparto de MODOS por episodio. "torre" mantiene la tarea original (la replica
# del maze); "sueltos" pone de 1 a N_LIBRES botones aleatorios delante del robot.
# Entrenar los dos a la vez evita que la politica se especialice en uno.
P_MODO_TORRE                 = 0.35

# Pose de reposo del brazo. MISMA tabla que config.ARM_REST_POSE /
# initial_positions.yaml / el reset del bridge — no se inventa otra
# (arm_env.REST_ANGLES es una tercera tabla distinta; ver el informe).
ARM_REST_POSE = {"joint_1": 0.0, "joint_2": -2.86234, "joint_3": 2.86234,
                 "joint_4": -1.5708, "joint_5": -1.5708, "joint_6": 1.5708}

# arm_base_link respecto a base_link/footprint_link (URDF rescue_robot_v2)
ARM_BASE_OFFSET = np.array([0.12, 0.0, 0.06])
# arm_base_link -> joint_1 (el HOMBRO). Es el centro de la envolvente de alcance
# del brazo, asi que es el marco correcto para sortear donde puede aparecer un
# boton. H_BASE del URDF rescue_robot_v2.
SHOULDER_OFF   = np.array([0.0, 0.0, 0.1583])
# tool_link respecto a link_6 (URDF rescue_robot_v2: tool_joint)
TCP_OFFSET_L6   = np.array([0.0, 0.12, 0.0])

CONTROL_DECIMATION = 25      # 25 * 0.002 s = 0.05 s -> 20 Hz, igual que config.py
SETTLE_STEPS       = 200

# Presupuesto de tiempo POR BOTON. Si se agota sin hundirlo, el episodio termina
# como FALLO y el reset sortea otro boton (ver reset(): con fixed_target=None
# cada episodio nuevo elige objetivo de `targets`).
# Es tiempo SIMULADO, no de reloj: la sim corre mucho mas rapido que el tiempo
# real, asi que 35 s aqui cuestan una fraccion de eso en pared.
# Referencia: el controlador scripted resuelve en ~123 pasos = 6.2 s, o sea que
# 35 s es un techo holgado (~5.6x). Subirlo mas no hace la tarea mas facil, solo
# alarga los episodios FALLIDOS, que es donde se va el tiempo de entrenamiento.
EPISODE_TIME_S     = 35.0
EPISODE_MAX_STEPS  = int(round(EPISODE_TIME_S / (0.002 * CONTROL_DECIMATION)))  # = 700

# Fraccion del recorrido que cuenta como pulsacion. 0.98, no 0.60: se pide que
# el boton quede hundido DEL TODO. No se exige 1.000 exacto porque el tope del
# joint lo hace el solver de restricciones, que siempre deja una decima de mm de
# blandura; 0.98 es "a fondo" sin depender de esa tolerancia.
# Con el umbral en 0.60 la politica dejaba de empujar en cuanto cobraba el bonus:
# medido, solo 5 de 9 episodios llegaban al 100%.
PRESS_THRESH_FRAC  = 0.98

# SOSTENER el boton. No basta con hundirlo: hay que MANTENERLO hundido
# HOLD_TIME_S segundos seguidos, con la garra, para que cuente como pulsado.
# Un toque instantaneo puntua, pero mucho menos (ver los pesos del reward).
# HISTERESIS del sostenido. Se ENTRA en la racha al llegar a PRESS_THRESH_FRAC
# (0.98, el boton a fondo) pero se MANTIENE mientras no baje de HOLD_KEEP_FRAC.
# Sin histeresis la racha se rompia sola: el boton se queda hundido pero cede
# decimas de milimetro por la blandura del contacto y la relajacion del brazo, y
# con un unico umbral al 98% eso reiniciaba el contador. Medido: el brazo
# congelado a proposito sobre el boton sostenia 0.15 s y 0.65 s en dos de los
# tres botones pese a tener el boton visiblemente hundido al tope.
# 0.90 de 18 mm = 16.2 mm: sigue siendo "hundido a fondo" a ojo y por contacto.
HOLD_KEEP_FRAC     = 0.90
HOLD_TIME_S        = 2.0
HOLD_STEPS         = int(round(HOLD_TIME_S / (0.002 * CONTROL_DECIMATION)))  # = 40

# Aleatorizacion de la torre. Con el modo "sueltos" en juego, la torre ya no es
# la unica tarea, asi que se le puede dar mas recorrido: mover la torre entera
# es otra forma de que el mismo boton aparezca en sitios distintos.
# La torre se apoya en el suelo, asi que solo se mueve en XY; el giro sobre su
# eje cambia que boton encara al robot, que es variedad util y gratis.
# Envolvente del jitter, acotada por DOS medidas opuestas:
#   - por abajo: a x<=0.52 la garra en pose de reposo ya toca el poste.
#   - por arriba: el boton de la cuspide esta al borde del espacio de trabajo.
#     Error del IK offline al alejar la torre:  x=0.55 -> 4.0 mm (bien),
#     x=0.57 -> 9.0 mm (justo), x=0.59 -> 14.6 mm (fuera). Con un jitter de
#     +-0.04 en X ese boton se salia y fallaba 4 de cada 5 episodios.
# Por eso X apenas se mueve; la variedad la ponen Y y sobre todo el GIRO, que
# desplaza los cuatro laterales bastante mas que cualquier traslacion.
TORRE_JITTER_LOW   = np.array([0.00, -0.06, 0.0])
TORRE_JITTER_HIGH  = np.array([0.01,  0.06, 0.0])
TORRE_YAW_JITTER   = 0.35    # rad (~20 grados) sobre el eje de la torre

# Botones ENTRENABLES con la base quieta. Criterio de tres filtros -- alcance,
# ausencia de colision Y condicionamiento por debajo del hard-stop de Servo --
# medido con 150 semillas de IK (`python3 tests/test_boton.py --reach`):
#
#   boton              poses que    sin        cond(J)      por debajo
#                      alcanzan     colision   min/mediana  de 100
#   0 (≙ door_8,  -X)      53          53        33 /  39    53/53   VIABLE
#   1 (≙ door_11, +Y)      59          52        21 /  22    43/52   VIABLE
#   2 (≙ door_9,  -Y)      74          72        22 /  29    41/72   VIABLE
#   3 (≙ door_7,  +Z)      53          53       132 / 143     0/53   <- excluido
#   4 (≙ door_10, +X)       0           -          -           -     <- excluido
#
# El 4 esta en la cara OPUESTA de la cupula: ninguna pose lo alcanza (error de
# IK ~105 mm frente a ~3.5 mm de los demas). Para ese hay que mover la base, que
# es lo que esta tarea no entrena.
#
# El 3 SI se alcanza y sin colision, pero practicamente todas sus poses tienen
# cond(J) por ENCIMA de 100, que es hard_stop_singularity_threshold en
# servo_params.yaml: MoveIt Servo pararia el brazo en seco en el robot real.
# Dos muestreos independientes de IK dieron 0/53 y 1/51 poses ejecutables, o sea
# del orden del 0-2%. Se deja fuera del set por defecto porque una tarea que casi
# nunca se completa no puede arrancar el aprendizaje -- es la leccion de
# docs/sigma_estado.md §7 sobre `pallets`.
#
# Para incluirlo de todos modos: BotonArmEnv(targets=[0, 1, 2, 3]).
TARGETS_ALCANZABLES = [0, 1, 2]

# ── Envolvente SEGURA de movimiento ─────────────────────────────────────────
# No son numeros inventados: salen de los limites que MoveIt hara cumplir en
# despliegue, para que lo que la politica aprende aqui siga siendo ejecutable
# alli. Si la politica vive fuera de esta envolvente, en el robot real MoveIt
# Servo la frena o la para en seco, y lo aprendido no sirve.
#
#   magnitud              tope duro     de donde sale                zona segura
#   --------------------  ------------  --------------------------  -----------
#   vel. articular        3.14 rad/s    joint_limits.yaml           <= 40%
#   acel. articular       10 rad/s^2    config.ARM_JOINT_LIMITS     <= 50%
#   recorrido articular   +-3.14 rad    URDF / XML MuJoCo           margen 0.15
#   cond. del jacobiano   100           servo_params.yaml           <= 80
#   vel. lineal del TCP   1.0 m/s       scale.linear + Pilz         (ya en tope)
#   vel. angular del TCP  1.57 rad/s    Pilz max_rot_vel            (ya en tope)
#
# Las dos ultimas no necesitan castigo: la accion esta acotada a [-1,1] y se
# escala por MAX_LINEAR_VEL=1.0 / MAX_ANGULAR_VEL=1.0, asi que el twist
# comandado ya nace dentro del limite cartesiano de Pilz.
#
# OJO — contradiccion del repo: config.ARM_JOINT_LIMITS declara max_vel 9.0
# rad/s para joint_4/5/6, pero joint_limits.yaml declara 3.14 para los seis.
# Manda el de MoveIt, que es el que se aplica en el robot: 3.14.
JOINT_VEL_MAX      = 3.14   # rad/s, joint_limits.yaml
JOINT_VEL_SAFE     = 0.40   # fraccion del tope sin castigo (zona muerta)
JOINT_ACC_MAX      = 10.0   # rad/s^2, config.ARM_JOINT_LIMITS (rampas del bridge)
JOINT_ACC_SAFE     = 0.50
# LIMITE POR SOFTWARE del recorrido articular. El brazo NUNCA comanda una
# posicion a menos de ARM_SOFT_MARGIN_RAD del tope fisico: en el robot real
# llevar un servo contra su tope lo hace forzar y "tronar", y es una forma
# conocida de romperlos. Mismo patron que FLIPPER_MIN_RAD/FLIPPER_MAX_RAD en
# config.py para los flippers.
#
# 0.15 rad (8.6 grados) no es arbitrario: es lo que CABE. La pose de reposo
# canonica (ARM_REST_POSE, la misma de initial_positions.yaml) tiene joint_2 y
# joint_3 en +-2.86234, o sea a 0.2777 rad del tope. Un margen mayor dejaria la
# propia pose de reposo fuera de rango.
ARM_SOFT_MARGIN_RAD = 0.15
JOINT_LIMIT_MARGIN  = 0.10   # rad ANTES del limite blando donde empieza el castigo
# Umbrales de singularidad DE servo_params.yaml. Servo mide el condicionamiento
# del jacobiano: a 80 empieza a frenar, a 100 PARA EN SECO. Una politica que
# opere por encima de 100 sencillamente no se movera en el robot real.
SING_COND_WARN     = 80.0
SING_COND_STOP     = 100.0

# Tope de velocidad LINEAL del punto de control. Con las escalas actuales no
# llega a morder: la accion esta acotada a [-1,1] y se multiplica por
# MAX_LINEAR_VEL=1.0, asi que el TCP no pasa de ~1.3 m/s (medido sobre una
# politica entrenada: media 0.50, p95 1.20, max 1.32; 0 de 180 pasos por encima
# de 3). Se deja escrito y vigilado igualmente: si alguien sube scale.linear en
# servo_params.yaml y con el MAX_LINEAR_VEL, este limite pasa a ser el que manda
# en vez de quedar el brazo suelto sin tope.
TCP_SPEED_MAX      = 3.0    # m/s

# ── PRECISION ANTES QUE VELOCIDAD ───────────────────────────────────────────
# Escala del twist comandado. arm_env/servo_params.yaml usan 1.0 m/s y 1.0
# rad/s; aqui se baja porque la tarea es de PRECISION, no de rapidez, y con el
# valor alto la politica arrancaba a fondo: medido sobre la politica entrenada,
# |accion lineal| = 0.85 de 1.0 en el paso 0 del episodio, cuando su media a lo
# largo del episodio es 0.10. O sea, un lunge inicial seguido de ajuste fino.
#
# OJO PARA DESPLIEGUE: si se baja aqui hay que bajar scale.linear y
# scale.rotational en servo_params.yaml EXACTAMENTE IGUAL, o el robot real se
# movera 2.5x mas rapido que lo entrenado. No se toca ese archivo desde aqui.
MAX_LINEAR_VEL     = 0.40   # m/s   (servo_params.yaml scale.linear seria 0.40)
MAX_ANGULAR_VEL    = 0.60   # rad/s (servo_params.yaml scale.rotational = 0.60)

# Topes POR ARTICULACION, no globales. El tope unico de 3.14 rad/s no distingue
# entre el hombro y la muñeca, y son muy distintos: joint_4/5/6 tienen
# actuatorfrcrange de +-16 N.m en el XML frente a los +-100/+-200 de los
# eslabones grandes. Son las articulaciones DEBILES, y son las que mueven la
# garra entera: forzarlas es lo que la dobla.
#
# Medido sobre la politica entrenada (max por joint, rad/s):
#     j1=2.76  j2=2.66  j3=2.22  j4=1.61  j5=1.81  j6=0.60
# y en aceleracion (max, rad/s^2):
#     j1=27.4  j2=34.1  j3=18.3  j4=22.0  j5=18.9  j6=10.3
# o sea hasta 3.4x la referencia de 10 rad/s^2 de config.ARM_JOINT_LIMITS.
ARM_VEL_MAX = np.array([1.50, 1.50, 1.50, 1.00, 1.00, 0.80])   # rad/s
ARM_ACC_MAX = np.array([8.00, 8.00, 8.00, 5.00, 5.00, 4.00])   # rad/s^2

# ── ANTI-WINDUP: que el hombro no reviente la muñeca ────────────────────────
# La accion se integra en una posicion comandada (_joint_pos) que va a unos
# actuadores <position kp=200>. Cuando la garra topa con el boton, el brazo deja
# de avanzar pero el integrador SIGUE, asi que el comando se separa de la
# posicion real y el par crece sin freno: hombro y codo empujan contra un
# obstaculo y la muñeca, que es la pieza debil, se lleva la reaccion y se dobla.
#
# Medido sobre la politica entrenada, par del actuador contra su tope del XML:
#     joint_3: 112.5 N.m de 100     joint_4: 26.5 de 16
#     joint_5:  68.8 N.m de  16  <- 4.3 veces su nominal
#
# El limite es geometrico: con un servo de posicion, par = kp * desfase. Luego
# acotar el desfase a (tope_de_par / kp) hace que el actuador NO PUEDA superar
# su par nominal, pase lo que pase. No es un numero elegido: sale de
# actuatorfrcrange del XML dividido por kp, y se recalcula del modelo en el
# __init__ por si el XML cambia.
#
# Es tambien lo que hace un variador real (anti-windup del lazo de posicion), y
# lo que evita que en el robot de verdad se rompa la muñeca empujando.
ANTIWINDUP = True
# Margen de seguridad sobre el desfase maximo. El recorte se aplica entre
# tramos de subpasos, no en cada subpaso, asi que el desfase real puede
# rebasar un poco el nominal antes de que le toque el recorte. Con 0.85 el
# rebase cabe dentro del par nominal en vez de superarlo.
LAG_SAFETY = 0.85

# El paso de control (CONTROL_DECIMATION subpasos) se parte en tramos, y entre
# tramos se vuelve a aplicar el anti-windup y la compensacion de gravedad.
#
# Por que no en cada subpaso: mj_step con nstep hace el bucle en C y suelta el
# GIL UNA vez; llamarlo 25 veces sueltas mata el paralelismo por hilos (medido:
# 1.27x en vez de 2.3x con 8 envs). Por que no una sola vez: recortando solo al
# principio, el contacto empuja el brazo durante los 25 subpasos y el desfase
# crece sin freno -- medido, 0.132 rad de desfase con un tope de 0.080 y 26.3
# N.m de par en joint_4 sobre un nominal de 16.
# 5 tramos de 5 subpasos es el punto medio: 5 sueltas de GIL en vez de 25.
SUBPASOS_POR_TRAMO = 5

# ── Pesos del reward ────────────────────────────────────────────────────────
# CALIBRACION. Lo que importa no es cada peso por separado sino la distancia
# entre conductas. Medido sobre episodios completos de 300 pasos:
#     no hacer nada (congelarse) .........  -3
#     acercarse sin chocar y fallar ......  +7
#     chocar de vez en cuando ............ -40
#     empotrado todo el episodio ......... -180
#     presionar el boton ................. +115
# La primera version cobraba -600 por empotrarse y -3 por quedarse quieto: con
# esa diferencia la politica aprende a NO MOVERSE, que es optimo y inutil.
# Es el mismo equilibrio que usa la base: bonus terminal grande (GOAL_BONUS
# 1000) contra castigo por paso pequeño (OBSTACLE_PENALTY 1.0).
# Los cuatro terminos de la tarea, en orden de lo que se quiere conseguir:
#   tocar el boton con la garra  ->  hundirlo  ->  SOSTENERLO  ->  2 s completos
# Los tres primeros son ESCALONES hacia el cuarto, y los tres estan acotados por
# episodio: se cobran una sola vez (touch) o como incremento sobre el mejor
# valor alcanzado (press, hold). Esto no es estetica, es lo unico que evita el
# farmeo -- ver la nota larga en _reward().
W_PROGRESS   = 50.0    # shaping potencial (prev_dist - dist); ~+15 por el trayecto
W_TOUCH      = 5.0     # UNA vez, la primera que la garra toca la cara del boton
W_PRESS      = 20.0    # potencial sobre el hundimiento maximo logrado con la garra
W_HOLD       = 40.0    # potencial sobre la mejor RACHA de sostenido lograda
W_ACTION     = 1e-3
# 250, no 100. Medido en la corrida anterior: la politica llegaba a avg_ret=102
# con 0% de exito, o sea que el SHAPING solo (touch 5 + press 20 + hold 40 +
# progress ~40) ya paga casi tanto como completar. Con esa diferencia la
# politica escala el shaping y no tiene prisa por cerrar la tarea. Es el mismo
# equilibrio que la base: GOAL_BONUS 1000 contra WP_BONUS 200.
PRESS_BONUS  = 250.0   # al completar los HOLD_TIME_S seguidos -> y fin de episodio
ALIVE        = -0.01   # coste por paso: empuja a terminar rapido

# Colisiones. Un paso con colision NO cobra los terminos positivos (mismo patron
# que OBSTACLE_PENALTY en base_env.compute_reward): si no, la politica aprende a
# empotrarse contra la torre mientras sigue cobrando el progreso.
# TRES condiciones que CORTAN el episodio en vez de castigarlo por paso. Todas
# comparten el mismo razonamiento: no son roces caros que se puedan compensar
# siguiendo, son maneras de haber estropeado el intento (o el robot).
#   - tocar el boton OBJETIVO con algo que no sea la garra
#   - tocar un boton que NO es el objetivo (con lo que sea)
#   - el brazo contra el propio robot
# Rozar la ESTRUCTURA de la torre (poste, nucleo) no entra: no es un boton, y
# hacerlo terminal dejaba la tarea sin resolver ni para el controlador de
# referencia.
# Un castigo por paso deja abierto que la politica lo pague y siga; terminar no.
P_ROCE_TORRE     = 0.30  # castigo (NO terminal): rozar el poste/nucleo de la torre
P_AUTOCOLISION   = 25.0  # terminal: el brazo contra el propio robot
P_BRAZO_SUELO    = 0.30  # el brazo contra el suelo -- esto SI sigue siendo castigo

# Tocar un boton que NO es el objetivo no se castiga: CORTA el episodio. En la
# pista real pulsar el boton equivocado no es un roce caro, es haber fallado la
# tarea -- no hay forma de "compensarlo" siguiendo. Un castigo por paso deja
# abierto que la politica lo pague y siga; terminar no.
P_BOTON_AJENO    = 25.0  # terminal: boton equivocado

# Movimiento anormal (se aplican SIEMPRE, tambien en pasos con colision).
# Cada uno lleva TECHO. Sin el, el termino de aceleracion se disparaba a -10.8
# en un impacto (medido) -- crece al cuadrado y sin acotar -- y un solo pico
# tapaba al resto del reward, incluido el bonus de haber presionado. Con techo,
# ningun castigo suave supera en magnitud a un choque (1.0-2.0), y el bonus de
# exito (50) sigue mandando en un episodio bueno.
# Claves del desglose del reward. sum(info[k] for k in REWARD_TERMS) == reward
# devuelto por step(), misma convencion que base_env.compute_reward -> una tabla
# de desglose cuadra sin trucos.
REWARD_TERMS = ("progress", "touch", "press", "hold", "bonus", "action", "alive",
                "col_torre", "autocolision", "brazo_suelo", "boton_ajeno",
                "tcp_speed", "joint_vel", "joint_acc", "joint_lim", "singular")

W_TCP_SPEED,  P_TCP_MAX  = 2.0, 0.50
W_JOINT_VEL,  P_VEL_MAX  = 0.5, 0.30
W_JOINT_ACC,  P_ACC_MAX  = 0.3, 0.30
W_JOINT_LIM,  P_LIM_MAX  = 1.0, 0.50
W_SINGULAR,   P_SING_MAX = 1.0, 0.30


def _dls_joint_vel(model, data, point: np.ndarray, body_id: int,
                   arm_dof_adr: np.ndarray, R_cmd: np.ndarray,
                   twist_cmd: np.ndarray, damping: float = DLS_DAMPING) -> np.ndarray:
    """Twist cartesiano en el frame de comando -> velocidades articulares.

    Igual que arm_env.cartesian_twist_to_joint_vel (jacobiano + pseudo-inversa
    amortiguada, el calculo de MoveIt Servo) salvo que el jacobiano se toma EN EL
    PUNTO `point` (el TCP/tool_link) y no en el origen del cuerpo. Ver el
    docstring del modulo.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    J = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])

    twist_world = np.concatenate([R_cmd @ twist_cmd[:3], R_cmd @ twist_cmd[3:]])
    J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6))
    return J_pinv @ twist_world


class BotonArmEnv:
    """Brazo solo (base congelada) presionando botones del panel."""

    def __init__(self,
                 xml_path: str = XML_PATH,
                 max_steps: int = EPISODE_MAX_STEPS,
                 control_decimation: int = CONTROL_DECIMATION,
                 freeze_base: bool = True,
                 grav_comp: bool = True,
                 randomize_torre: bool = True,
                 fixed_target: Optional[int] = None,
                 targets: Optional[List[int]] = None,
                 render: bool = False,
                 seed: Optional[int] = None,
                 model: Optional["mujoco.MjModel"] = None,
                 disable_sensors: bool = True,
                 cortes: Optional[set] = None):
        """model: MjModel ya compilado (una copia POR env). Compilar el XML es
        caro; el vec env lo hace una vez y reparte copias, igual que
        mujoco_sim_base.VecMujocoEnv."""

        self.model = model if model is not None else mujoco.MjModel.from_xml_path(xml_path)

        # El modelo del robot trae 64 rangefinders (el lidar) que raycastean en
        # CADA mj_step. Esta tarea no lee ni uno: press_frac() saca la posicion
        # del boton de qpos, no de sensordata, y no hay camaras ni lidar en la
        # observacion. Apagarlos vale 1.7x de velocidad de simulacion
        # (2156 -> 3651 mj_step/s, medido). No toca la dinamica: los sensores
        # solo LEEN el estado.
        # NO se bajan las iteraciones del solver: eso si cambiaria la fisica
        # respecto a la que ya esta validada.
        if disable_sensors:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
        self.data  = mujoco.MjData(self.model)

        self.max_steps          = max_steps
        self.control_decimation = control_decimation
        self.freeze_base        = freeze_base
        self.grav_comp          = grav_comp
        # Condiciones que CORTAN el episodio. Se pueden desactivar una a una
        # para aislar cual bloquea la tarea (ver tests/test_boton.py --ablacion).
        #   "ajeno"    tocar un boton que no es el objetivo
        #   "no_garra" tocar el boton objetivo con algo que no es la garra
        #   "auto"     autocolision del brazo
        self.cortes = ({"ajeno", "no_garra", "auto"} if cortes is None
                       else set(cortes))
        self.randomize_torre    = randomize_torre
        self.fixed_target       = fixed_target
        # Subconjunto de botones sobre el que entrenar. La torre es una cupula:
        # desde una sola posicion del robot no todos son alcanzables (ver
        # TARGETS_ALCANZABLES y `tests/test_boton.py --reach`).
        self.targets            = list(TARGETS_ALCANZABLES) if targets is None else list(targets)
        self.rng                = np.random.default_rng(seed)
        # CURRICULO sobre los distractores. 0 -> un solo boton (el objetivo);
        # 1 -> hasta N_LIBRES. Con distractores desde el paso 1, el 60% de los
        # episodios moria por tocar un boton vecino antes de que la politica
        # tuviera precision para evitarlos: no llegaba a aprender la tarea base.
        # El entrenador lo sube segun la tasa de exito (ver train_boton.py).
        self.dificultad         = 1.0
        self._dt                = self.model.opt.timestep * control_decimation

        m = self.model
        aid = lambda n: self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        jid = lambda n: self._id(mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: self._id(mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: self._id(mujoco.mjtObj.mjOBJ_SITE, n)
        gid = lambda n: self._id(mujoco.mjtObj.mjOBJ_GEOM, n)

        # ── brazo ──────────────────────────────────────────────────────────
        self._a_arm    = np.array([aid(n) for n in ARM_ACTUATORS], dtype=np.int32)
        self._j_arm    = [jid(n) for n in ARM_JOINTS]
        self._q_arm    = np.array([m.jnt_qposadr[j] for j in self._j_arm], dtype=np.int32)
        self._v_arm    = np.array([m.jnt_dofadr[j]  for j in self._j_arm], dtype=np.int32)
        # Topes FISICOS del XML y topes BLANDOS que es lo que se comanda.
        self._arm_lo_hw = m.jnt_range[self._j_arm, 0].copy()
        self._arm_hi_hw = m.jnt_range[self._j_arm, 1].copy()
        self._arm_lo = self._arm_lo_hw + ARM_SOFT_MARGIN_RAD
        self._arm_hi = self._arm_hi_hw - ARM_SOFT_MARGIN_RAD

        # Desfase maximo comando-real por articulacion = par nominal / kp.
        # Con eso el actuador de posicion no puede pasar de su par del XML.
        kp = np.array([float(m.actuator_gainprm[a, 0]) for a in self._a_arm])
        par_max = np.array([float(np.abs(m.jnt_actfrcrange[j]).max())
                            for j in self._j_arm])
        self._lag_max = par_max / np.maximum(kp, 1e-6)

        # ── dedos ──────────────────────────────────────────────────────────
        self._a_flip = np.array([aid(n) for n in FLIPPER_ACTS], dtype=np.int32)
        self._q_flip = np.array([m.jnt_qposadr[jid(n)] for n in FLIPPER_JOINTS],
                                dtype=np.int32)
        self._a_fing = np.array([aid(n) for n in FINGER_ACTS], dtype=np.int32)
        self._j_fing = [jid(n) for n in FINGER_JOINTS]
        self._q_fing = np.array([m.jnt_qposadr[j] for j in self._j_fing], dtype=np.int32)
        self._v_fing = np.array([m.jnt_dofadr[j]  for j in self._j_fing], dtype=np.int32)

        # ── cuerpos de referencia ──────────────────────────────────────────
        self._b_l6 = bid("link_6")
        self._b_fp = bid("footprint_link")
        self._b_torre = bid("torre_botones")
        self._torre_pos0  = m.body_pos[self._b_torre].copy()
        self._torre_quat0 = m.body_quat[self._b_torre].copy()

        # ── botones ────────────────────────────────────────────────────────
        self._j_bot = [jid(n) for n in BOTON_JOINTS]
        self._q_bot = np.array([m.jnt_qposadr[j] for j in self._j_bot], dtype=np.int32)
        self._v_bot = np.array([m.jnt_dofadr[j]  for j in self._j_bot], dtype=np.int32)
        self._s_bot = [sid(n) for n in BOTON_SITES]
        self._b_bot = [bid(n) for n in BOTON_BODIES]

        # geoms de la torre (para detectar golpes con lo que no son los dedos).
        # Se recorre el SUBARBOL del cuerpo de la torre. Filtrar por body_rootid
        # no vale: la torre esta anclada al mundo, asi que su rootid es 0 y el
        # suelo entraria en el conjunto.
        torre_sub = self._subtree(self._b_torre)
        self._g_torre = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in torre_sub)

        # Cabeza de cada boton: es la UNICA superficie de la torre que se puede
        # tocar legitimamente, y solo con los dedos.
        self._g_head = [gid(f"boton_{i}_head") for i in range(N_BOTONES)]
        # Geoms de CADA boton completo (montura + vastago + cabeza), para poder
        # distinguir "toque un boton ajeno" de "roce la estructura de la torre".
        self._b_libre = [bid(n) for n in LIBRE_BODIES]
        self._q_libre = np.array([m.jnt_qposadr[jid(n)] for n in LIBRE_JOINTS], dtype=np.int32)
        self._v_libre = np.array([m.jnt_dofadr[jid(n)]  for n in LIBRE_JOINTS], dtype=np.int32)
        self._s_libre = [sid(n) for n in LIBRE_SITES]
        self._g_libre_head = [gid(n) for n in LIBRE_HEADS]
        self._g_libre = [set(g for g in range(m.ngeom)
                             if m.geom_bodyid[g] in self._subtree(bid(n)))
                         for n in LIBRE_BODIES]
        self._g_boton = [set(g for g in range(m.ngeom)
                             if m.geom_bodyid[g] in self._subtree(bid(f"boton_{i}")))
                         for i in range(N_BOTONES)]

        # Geoms del robot (subarbol del chasis) y, dentro, los del brazo.
        robot_sub = self._subtree(self._b_fp)
        self._g_robot = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_sub)
        self._b_dedos = frozenset((bid("left_finger_link"), bid("right_finger_link")))
        self._pares_benignos = set()
        arm_bodies = {bid(n) for n in
                      ("link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
                       "logitech_gripper_assembly", "left_finger_link", "right_finger_link")}
        self._g_arm = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies)

        # Geoms del cuerpo world = el suelo.
        self._g_world = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == 0)
        # geoms de los dedos = los unicos que "deben" tocar la torre
        self._g_finger = set(g for g in range(m.ngeom)
                             if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                             in ("left_finger_link", "right_finger_link"))

        # ── base (freejoint del chasis) ────────────────────────────────────
        self._base_qadr = m.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")]
        self._base_vadr = m.jnt_dofadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")]
        self._base_qpos_frozen = None
        # Restriccion de igualdad que suelda el chasis al mundo (ver el bloque
        # <equality> de aesir_arm_botones.xml). Si el mundo no la trae, se cae
        # al congelado por Python en cada substep, que es exacto pero mata el
        # paralelismo por hilos.
        self._eq_freeze = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "freeze_base")

        # ── lidar: se para; no se usa en esta tarea ────────────────────────
        self._a_lidar = aid("vel_lidar_spin")

        # ── tamaños expuestos ──────────────────────────────────────────────
        self.act_len = 7
        self.obs_len = 28

        self._joint_pos  = np.zeros(6)      # integrador de la IK
        self._qvel_cmd_prev = np.zeros(6)   # velocidad comandada anterior (rampa)
        self._prev_frac_dedo = 0.0
        self._hold = self._hold_best = self._prev_hold_best = 0
        self._tocado = False
        self._cobro_touch = False
        self._prev_qvel  = np.zeros(6)      # para la aceleracion articular
        self._prev_tcp   = None             # para la velocidad del TCP
        self._v_tcp      = 0.0
        self._cond_cache = None             # cond(J) del paso actual
        self._fin_por    = ""               # motivo de corte del episodio
        self._step_count = 0
        self._frac_con_dedo = 0.0
        self.target      = 0
        self._prev_dist  = 0.0
        self._pressed    = False

        self.viewer = None
        if render:
            from mujoco import viewer as _mj_viewer   # local: no sombrear `mujoco`
            self.viewer = _mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 2.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth   = 150
            self.viewer.cam.lookat[:] = [0.35, 0.0, 0.45]

    # ── utilidades ─────────────────────────────────────────────────────────
    def _subtree(self, root: int) -> set:
        """ids de `root` y de todos sus descendientes en el arbol de cuerpos."""
        out = {root}
        for b in range(self.model.nbody):
            p = b
            while p != 0:
                if p == root:
                    out.add(b)
                    break
                p = int(self.model.body_parentid[p])
        return out

    def _id(self, objtype, name: str) -> int:
        i = mujoco.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise ValueError(f"No encontrado en el modelo: {objtype!s} '{name}'")
        return i

    def arm_base_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """(posicion, rotacion) del frame arm_base_link en el mundo."""
        R = self.data.xmat[self._b_fp].reshape(3, 3)
        p = self.data.xpos[self._b_fp] + R @ ARM_BASE_OFFSET
        return p, R

    def tcp(self) -> np.ndarray:
        """Punto de control = tool_link = punto medio entre las puntas de los dedos."""
        R = self.data.xmat[self._b_l6].reshape(3, 3)
        return self.data.xpos[self._b_l6] + R @ TCP_OFFSET_L6

    def target_pos(self) -> np.ndarray:
        """Centro de la cara exterior del boton objetivo, en el mundo."""
        return self.data.site_xpos[self._act_sites[self.target]].copy()

    def button_normal(self, i: Optional[int] = None) -> np.ndarray:
        """Normal saliente del boton en el mundo = su eje de presion.

        Se LEE del modelo (el +Z local del cuerpo del boton), asi que sigue
        siendo correcta si la torre se mueve o se gira. Hundir el boton es
        empujar en -normal.
        """
        i = self.target if i is None else i
        return self.data.xmat[self._act_bodies[i]].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def press_frac(self, i: Optional[int] = None) -> float:
        """Fraccion del recorrido hundida (0 = suelto, 1 = a fondo)."""
        i = self.target if i is None else i
        return float(np.clip(self.data.qpos[self._act_qadr[i]] / PRESS_TRAVEL, 0.0, 1.0))

    # ── observacion ────────────────────────────────────────────────────────
    def _observation(self) -> np.ndarray:
        p_ab, R_ab = self.arm_base_frame()
        tcp   = self.tcp()
        tgt   = self.target_pos()

        # Lo que pidio la tarea: posicion del boton RELATIVA a arm_base_link,
        # que es el frame en el que MoveIt Servo recibe comandos.
        tgt_rel = R_ab.T @ (tgt - p_ab)
        tcp_rel = R_ab.T @ (tcp - p_ab)
        err_rel = tgt_rel - tcp_rel

        # Normal del boton en el frame arm_base_link. Imprescindible aqui: en la
        # cupula cada boton se hunde en una direccion distinta, asi que la
        # politica no puede deducir hacia donde empujar solo de la posicion.
        n_rel = R_ab.T @ self.button_normal()

        return np.concatenate([
            self.data.qpos[self._q_arm],            # 6
            self.data.qvel[self._v_arm],            # 6
            self.data.qpos[self._q_fing],           # 2
            self.data.qvel[self._v_fing],           # 2
            tgt_rel,                                # 3  <- posicion relativa del boton
            err_rel,                                # 3  <- vector TCP -> boton
            n_rel,                                  # 3  <- normal (eje de presion)
            [np.linalg.norm(err_rel)],              # 1
            [self.press_frac()],                    # 1
            [min(self.jac_cond() / SING_COND_STOP, 2.0)],   # 1 <- singularidad
        ]).astype(np.float32)                       # = 28

    # ── accion ─────────────────────────────────────────────────────────────
    def _apply_action(self, action: np.ndarray) -> None:
        a = np.clip(action, -1.0, 1.0)

        twist = a[:6] * np.array([MAX_LINEAR_VEL] * 3 + [MAX_ANGULAR_VEL] * 3)
        _, R_ab = self.arm_base_frame()
        qvel_arm = _dls_joint_vel(self.model, self.data, self.tcp(), self._b_l6,
                                  self._v_arm, R_ab, twist)

        # TOPES DUROS, no castigos. Un castigo con techo (0.30) contra un bonus
        # de 250 sale rentable pagarlo: la politica aprende a superarlo y lo
        # aprendido no transfiere, porque el robot real si tiene los topes.
        #
        # 1) VELOCIDAD por articulacion. Se escala el VECTOR entero por el factor
        #    mas restrictivo, no componente a componente: eso conserva la
        #    direccion del movimiento (que es lo que hace MoveIt Servo) en vez de
        #    torcerla, cosa que en una tarea de precision importa.
        exceso = np.abs(qvel_arm) / ARM_VEL_MAX
        if exceso.max() > 1.0:
            qvel_arm = qvel_arm / exceso.max()

        # 2) ACELERACION por articulacion: rampa. Limita cuanto puede CAMBIAR la
        #    velocidad comandada entre pasos de control. Es la misma idea que las
        #    rampas AVR446 que base_training/robot_control.py aplica a orugas y
        #    flippers, y que el bridge de despliegue reproduce. Sin esto el
        #    movimiento sale a tirones: medido, hasta 34 rad/s^2 en joint_2.
        dv = qvel_arm - self._qvel_cmd_prev
        lim = ARM_ACC_MAX * self._dt
        exc_a = np.abs(dv) / lim
        if exc_a.max() > 1.0:
            dv = dv / exc_a.max()
        qvel_arm = self._qvel_cmd_prev + dv
        self._qvel_cmd_prev = qvel_arm.copy()

        self._joint_pos = np.clip(self._joint_pos + qvel_arm * self._dt,
                                  self._arm_lo, self._arm_hi)

        self._aplicar_antiwindup()

        # un solo valor para los dos dedos: -1 abierta, +1 cerrada
        grip = float((a[6] + 1.0) / 2.0 * FINGER_CLOSED)
        self.data.ctrl[self._a_fing] = grip

        self.data.ctrl[self._a_lidar] = 0.0

    def _aplicar_antiwindup(self) -> None:
        """ANTI-WINDUP. El comando no puede separarse de la posicion REAL mas de
        lo que da el par nominal de cada articulacion (par = kp * desfase). Si la
        garra esta bloqueada contra el boton, el integrador deja de correr por
        delante y el hombro no puede seguir empujando contra la muñeca.

        Se llama al comandar la accion Y entre tramos de subpasos, porque el
        contacto puede empujar el brazo mientras la fisica avanza: recortando
        solo al principio del paso de control, el desfase llegaba a 0.132 rad
        con un tope de 0.080 (medido)."""
        if not ANTIWINDUP:
            return
        q_real = self.data.qpos[self._q_arm]
        lag = self._lag_max * LAG_SAFETY
        np.clip(self._joint_pos, q_real - lag, q_real + lag, out=self._joint_pos)
        self.data.ctrl[self._a_arm] = self._joint_pos

    # ── colocacion de botones ──────────────────────────────────────────────
    def _pose_libre(self, p_sh: np.ndarray, R_ab: np.ndarray):
        """Sortea (posicion, quat) de un boton suelto DE FRENTE al robot.

        Se muestrea en esfericas alrededor del HOMBRO (no del origen del mundo):
        el alcance del brazo es una envolvente centrada ahi, asi que sortear en
        ese marco garantiza que lo que sale esta dentro de rango sin tener que
        rechazar muestras.

        La NORMAL del boton apunta de vuelta hacia el hombro -- el boton "mira"
        al robot, que es lo pedido -- con una desviacion aleatoria de hasta
        LIBRE_TILT_MAX para que no sea siempre perfectamente de frente y la
        politica no aprenda a asumir una orientacion fija.
        """
        rng = self.rng
        r = rng.uniform(LIBRE_R_MIN, LIBRE_R_MAX)
        az = rng.uniform(-LIBRE_AZIM_MAX, LIBRE_AZIM_MAX)
        el = rng.uniform(LIBRE_ELEV_MIN, LIBRE_ELEV_MAX)
        d_local = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = p_sh + R_ab @ (d_local * r)

        # normal deseada: del boton HACIA el hombro, con desvio aleatorio
        n = -(R_ab @ d_local)
        eje = rng.normal(size=3)
        eje -= eje.dot(n) * n
        nn = np.linalg.norm(eje)
        if nn > 1e-9:
            ang = rng.uniform(0.0, LIBRE_TILT_MAX)
            eje /= nn
            n = n * np.cos(ang) + np.cross(eje, n) * np.sin(ang)
            n /= np.linalg.norm(n)

        # quat que lleva +Z local a n (el +Z del cuerpo es la normal saliente)
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, n) if hasattr(mujoco, "mju_quatZ2Vec") else None
        if not np.any(q):
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, n); c = float(np.dot(z, n))
            if np.linalg.norm(v) < 1e-9:
                q[:] = [1.0, 0, 0, 0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
            else:
                v /= np.linalg.norm(v)
                ang = np.arccos(np.clip(c, -1, 1))
                q[0] = np.cos(ang / 2); q[1:] = v * np.sin(ang / 2)
        return pos, q

    def _alcanzable(self, cara: np.ndarray, normal: np.ndarray,
                    iters: int = 35) -> bool:
        """IK rapida desde la pose de reposo: ¿puede el brazo poner el TCP en la
        cara del boton con la garra encarandola?

        Por que existe: barriendo el volumen de aparicion solo el 53% de las
        colocaciones salian viables. Estrechar el volumen hasta el 70% se puede,
        pero cada recorte estrecha tambien lo que la politica aprende a hacer.
        Filtrar AQUI conserva el volumen ancho y garantiza que el episodio se
        puede ganar, que es lo que de verdad hace falta: un episodio imposible
        no ensena nada y ademas gasta los 35 s enteros.

        UNA semilla (la pose de reposo) y pocas iteraciones a proposito: no
        busca la mejor solucion, solo responde si hay uso. Cuesta ~20 ms, contra
        los ~300 ms de la busqueda multi-semilla del chequeo offline.
        Sesga hacia lo alcanzable DESDE REPOSO, que es justo desde donde el
        episodio va a empezar."""
        m, d = self.model, self.data
        q = self._joint_pos.copy()
        q_save = d.qpos[self._q_arm].copy()
        want = -normal
        ok = False
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self._q_arm] = q
            mujoco.mj_forward(m, d)
            R6 = d.xmat[self._b_l6].reshape(3, 3)
            tcp = d.xpos[self._b_l6] + R6 @ TCP_OFFSET_L6
            e_pos = cara - tcp
            n_cur = R6 @ np.array([0.0, 1.0, 0.0])
            ax = np.cross(n_cur, want); sa = np.linalg.norm(ax)
            e_ori = (np.arctan2(sa, float(np.dot(n_cur, want))) * (ax / sa)
                     if sa > 1e-9 else np.zeros(3))
            if np.linalg.norm(e_pos) < 0.012 and np.linalg.norm(e_ori) < 0.35:
                ok = True
                break
            mujoco.mj_jac(m, d, jp, jr, tcp, self._b_l6)
            J = np.vstack([jp[:, self._v_arm], jr[:, self._v_arm] * 0.25])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6),
                                       np.concatenate([e_pos, e_ori * 0.25]))
            q = np.clip(q + 0.5 * dq, self._arm_lo, self._arm_hi)
        d.qpos[self._q_arm] = q_save
        mujoco.mj_forward(m, d)
        return ok

    def _aparcar(self, body_id: int, geoms: set, activo: bool) -> None:
        """Enciende o apaga un boton entero.

        Mover el cuerpo lejos NO basta: sus geoms siguen en el arbol de colision
        y generan contactos contra el suelo o entre ellos. Apagando
        contype/conaffinity el boton desaparece de la fisica de verdad, que es
        lo que se quiere de un objeto que no participa en el episodio."""
        val = 1 if activo else 0
        for g in geoms:
            self.model.geom_contype[g] = val
            self.model.geom_conaffinity[g] = val
        if not activo:
            self.model.body_pos[body_id] = PARKING

    def _colocar_botones(self) -> None:
        """Decide el MODO del episodio y deja preparados los indices activos.

        modo "torre"   -> los 5 botones de la replica del maze, la torre movida
                          dentro de su jitter; los sueltos aparcados.
        modo "sueltos" -> de 1 a N_LIBRES botones repartidos delante del robot;
                          la torre aparcada fuera de juego.

        `self._act_*` son las listas que usa TODO lo demas (target_pos,
        button_normal, press_frac, clasificacion de contactos), asi que el resto
        del env no sabe ni le importa en que modo esta.
        """
        m, d = self.model, self.data
        p_ab, R_ab = self.arm_base_frame()
        p_sh = p_ab + R_ab @ SHOULDER_OFF

        self.modo = "torre" if self.rng.random() < P_MODO_TORRE else "sueltos"

        if self.modo == "torre":
            for k, b in enumerate(self._b_libre):
                self._aparcar(b, self._g_libre[k], False)
            for g in self._g_torre:
                m.geom_contype[g] = 1
                m.geom_conaffinity[g] = 1
            if self.randomize_torre:
                m.body_pos[self._b_torre] = self._torre_pos0 + self.rng.uniform(
                    TORRE_JITTER_LOW, TORRE_JITTER_HIGH)
                yaw = float(self.rng.uniform(-TORRE_YAW_JITTER, TORRE_YAW_JITTER))
                m.body_quat[self._b_torre] = [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
            else:
                m.body_pos[self._b_torre] = self._torre_pos0
                m.body_quat[self._b_torre] = self._torre_quat0
            self._act_bodies = list(self._b_bot)
            self._act_sites = list(self._s_bot)
            self._act_qadr = list(self._q_bot)
            self._act_heads = list(self._g_head)
            self._act_geoms = list(self._g_boton)
            self.target = (self.fixed_target if self.fixed_target is not None
                           else int(self.rng.choice(self.targets)))
        else:
            self._aparcar(self._b_torre, self._g_torre, False)
            for k, b in enumerate(self._b_libre):       # todos fuera de juego
                self._aparcar(b, self._g_libre[k], False)
            mujoco.mj_forward(m, d)
            n_max = 1 + int(round(float(np.clip(self.dificultad, 0.0, 1.0)) * (N_LIBRES - 1)))
            n_bot = int(self.rng.integers(1, n_max + 1))
            puestos = []        # caras ya colocadas, para la separacion minima
            idx = []            # indices REALMENTE colocados
            for k in range(N_LIBRES):
                if len(idx) >= n_bot:
                    break
                # RECHAZO de verdad: se prueba una pose, se coloca, se recalcula
                # la cinematica y se comprueba que no toque al robot. Solo se
                # acepta si pasa los tres filtros; si en 40 intentos no sale,
                # este boton se queda aparcado (mejor uno menos que uno malo).
                # Presupuesto de intentos: el OBJETIVO (el primero) lleva ademas
                # el filtro de alcanzabilidad, que cuesta, asi que se le dan
                # menos intentos. Con 40 intentos x IK el reset costaba 2.6 s.
                intentos = 10 if len(idx) == 0 else 20
                for _ in range(intentos):
                    pos, q = self._pose_libre(p_sh, R_ab)
                    if pos[2] < LIBRE_Z_MIN:
                        continue
                    for g in self._g_libre[k]:
                        m.geom_contype[g] = 1
                        m.geom_conaffinity[g] = 1
                    m.body_pos[self._b_libre[k]] = pos
                    m.body_quat[self._b_libre[k]] = q
                    mujoco.mj_forward(m, d)
                    choca = False
                    for c in range(d.ncon):
                        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
                        a1, a2 = g1 in self._g_libre[k], g2 in self._g_libre[k]
                        # cuenta cualquier contacto del boton: con el robot, con
                        # el suelo, o con otro boton ya colocado.
                        if a1 != a2:
                            choca = True
                            break
                    if not choca:
                        # El OBJETIVO tiene que ser alcanzable; los distractores
                        # no hace falta (solo estan para que la politica aprenda
                        # a no tocar lo que no toca).
                        cara_k = d.site_xpos[self._s_libre[k]].copy()
                        nrm_k = (d.xmat[self._b_libre[k]].reshape(3, 3)
                                 @ np.array([0.0, 0.0, 1.0]))
                        if len(idx) == 0 and not self._alcanzable(cara_k, nrm_k):
                            self._aparcar(self._b_libre[k], self._g_libre[k], False)
                            mujoco.mj_forward(m, d)
                            continue
                        # separacion medida entre CARAS, no entre origenes: dos
                        # botones con origenes a 0.16 pueden tener las caras mas
                        # cerca si miran en direcciones distintas (la cara esta a
                        # 76 mm del origen a lo largo de la normal).
                        cara = d.site_xpos[self._s_libre[k]].copy()
                        if any(np.linalg.norm(cara - o) < LIBRE_SEP_MIN for o in puestos):
                            self._aparcar(self._b_libre[k], self._g_libre[k], False)
                            mujoco.mj_forward(m, d)
                            continue
                        puestos.append(cara)
                        idx.append(k)
                        break
                    self._aparcar(self._b_libre[k], self._g_libre[k], False)
                    mujoco.mj_forward(m, d)
                else:
                    self._aparcar(self._b_libre[k], self._g_libre[k], False)
                    mujoco.mj_forward(m, d)
            if not idx:                 # nunca deberia pasar; red de seguridad
                idx = [0]
                m.body_pos[self._b_libre[0]], m.body_quat[self._b_libre[0]] = \
                    self._pose_libre(p_sh, R_ab)
                mujoco.mj_forward(m, d)
            # SOLO los realmente colocados. Antes esto era range(n_bot), asi que
            # si alguno se rechazaba entraba en la lista APARCADO a z=-5, debajo
            # del plano del suelo: el "boton objetivo" podia estar enterrado y en
            # contacto permanente con el suelo.
            self._act_bodies = [self._b_libre[k] for k in idx]
            self._act_sites = [self._s_libre[k] for k in idx]
            self._act_qadr = [self._q_libre[k] for k in idx]
            self._act_heads = [self._g_libre_head[k] for k in idx]
            self._act_geoms = [self._g_libre[k] for k in idx]
            # El objetivo es SIEMPRE el primero colocado, que es el unico al que
            # se le exigio ser alcanzable.
            self.target = 0
        self._n_activos = len(self._act_bodies)

    # ── diagnostico de seguridad ───────────────────────────────────────────
    def jac_cond(self) -> float:
        """Cacheado por paso de control. Se consultaba TRES veces por step
        (observacion, castigo de singularidad e info) y cada consulta es un
        mj_jac + un SVD: medido, era el mayor coste del step despues de la
        propia fisica. El cache se invalida en _invalidate_cache(), justo
        despues de avanzar la fisica.

        Condicionamiento del jacobiano del TCP = la MISMA metrica de
        singularidad que usa MoveIt Servo (servo_params.yaml:
        lower_singularity_threshold=80 -> frena, hard_stop=100 -> para en seco).

        Se mide aqui para que la politica pague en entrenamiento el precio que
        pagaria en despliegue: una pose con cond>100 no se mueve en el robot
        real, por muy buena que parezca en MuJoCo.
        """
        if self._cond_cache is not None:
            return self._cond_cache
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.tcp(), self._b_l6)
        J = np.vstack([jacp[:, self._v_arm], jacr[:, self._v_arm]])
        sv = np.linalg.svd(J, compute_uv=False)
        self._cond_cache = float(sv[0] / max(sv[-1], 1e-12))
        return self._cond_cache

    def _invalidate_cache(self) -> None:
        self._cond_cache = None

    def _clasificar_contactos(self) -> Tuple[int, int, int, bool, int, int]:
        """Recorre los contactos del paso y devuelve:

          (torre_malo, autocolision, brazo_suelo, dedo_en_objetivo, boton_ajeno,
           no_garra_en_objetivo)

        LEGITIMO — y solo esto: un geom de los DEDOS contra la CABEZA del boton
        objetivo. Cualquier otro toque a la torre (montura, poste, nucleo, otro
        boton, o la cabeza correcta pero con el codo en vez de los dedos) cuenta
        como choque.

        Autocolision: contacto del propio robot en el que interviene EL BRAZO
        -- que es lo pedido: "el robot toca cualquier parte de si mismo con
        cualquier parte del brazo". MuJoCo ya filtra los pares padre-hijo y los
        <exclude> del XML. Encima se descartan TRES casos, los tres detectados
        mirando la simulacion, no razonando:

          1. Los dos dedos tocandose: es la garra CERRANDOSE, que es justo lo
             que hay que hacer para formar el taco. Sin esta excepcion el
             contador saltaba en cada cierre y el controlador de referencia
             pasaba de 3/3 a 0/12.
          2. Contactos SIN el brazo. Visto en el render: dos ruedas del MISMO
             flipper tocandose entre si (wheel_flipper1_1 <-> wheel_flipper1_2)
             mataban el episodio sin que el brazo interviniera.
          3. Pares que YA se tocan en la pose de reposo (self._pares_benignos).
             ARM_REST_POSE deja el brazo plegado sobre si mismo con link_1 rozando
             link_3 (0.06 mm de penetracion, medido): sin descartarlo, TODO
             episodio nace en autocolision y puede morir en el primer paso.

        Suelo: solo se cuenta para geoms del BRAZO. Las orugas y las ruedas
        apoyan en el suelo todo el rato; eso es estar de pie, no chocar.
        """
        torre_malo = auto = suelo = ajeno = no_garra = 0
        m_geom_body = self.model.geom_bodyid
        dedo_ok = False
        head_ok = self._act_heads[self.target]
        ajenos = self._g_ajenos
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            t1, t2 = g1 in self._g_tocable, g2 in self._g_tocable
            r1, r2 = g1 in self._g_robot, g2 in self._g_robot

            if t1 != t2 and (r1 or r2):                    # robot <-> torre
                g_rob = g2 if t1 else g1
                g_tor = g1 if t1 else g2
                es_garra = g_rob in self._g_finger
                if g_tor in ajenos:
                    # Un boton que no es el objetivo: da igual con que se toque.
                    ajeno += 1
                elif g_tor in self._act_geoms[self.target]:
                    # El boton objetivo. Con la GARRA vale (y si es la cara,
                    # cuenta para atribuirle el hundimiento); con cualquier otra
                    # parte del brazo, corta.
                    if es_garra:
                        if g_tor == head_ok:
                            dedo_ok = True
                    else:
                        no_garra += 1
                else:
                    # Estructura de la torre (poste, nucleo): roce caro, no un
                    # fallo de la tarea -> castigo, no corte.
                    torre_malo += 1
            elif r1 and r2:                                # robot <-> robot
                par = frozenset((int(m_geom_body[g1]), int(m_geom_body[g2])))
                if par == self._b_dedos:
                    continue          # la garra cerrandose, no una colision
                if par in self._pares_benignos:
                    continue          # ya estaban en contacto en la pose de reposo
                if not (g1 in self._g_arm or g2 in self._g_arm):
                    continue          # sin el brazo de por medio no es lo pedido
                auto += 1
            elif ((g1 in self._g_arm and g2 in self._g_world) or
                  (g2 in self._g_arm and g1 in self._g_world)):
                suelo += 1
        return torre_malo, auto, suelo, dedo_ok, ajeno, no_garra

    def _penal_movimiento(self, qvel: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Castigos de movimiento anormal. Devuelve (vel, acel, limite, singular).

        Los dos primeros usan ZONA MUERTA + exceso al cuadrado, igual que el
        castigo de aceleracion del chasis en base_env.compute_reward: moverse
        deprisa dentro de la envolvente sale gratis, solo se paga el exceso.
        """
        # velocidad LINEAL del TCP contra TCP_SPEED_MAX
        tcp = self.tcp()
        v_tcp = (float(np.linalg.norm(tcp - self._prev_tcp)) / self._dt
                 if self._prev_tcp is not None else 0.0)
        self._prev_tcp = tcp
        self._v_tcp = v_tcp
        p_tcp = min(W_TCP_SPEED * max(0.0, v_tcp / TCP_SPEED_MAX - 1.0) ** 2, P_TCP_MAX)

        # velocidad articular, normalizada al tope de joint_limits.yaml
        v = float(np.linalg.norm(qvel / JOINT_VEL_MAX) / np.sqrt(len(qvel)))
        p_vel = min(W_JOINT_VEL * max(0.0, v - JOINT_VEL_SAFE) ** 2, P_VEL_MAX)

        # aceleracion articular, normalizada al tope de rampa del bridge
        acc = (qvel - self._prev_qvel) / self._dt
        a = float(np.linalg.norm(acc / JOINT_ACC_MAX) / np.sqrt(len(acc)))
        p_acc = min(W_JOINT_ACC * max(0.0, a - JOINT_ACC_SAFE) ** 2, P_ACC_MAX)

        # cercania al tope de recorrido: lineal en la incursion en el margen
        q = self.data.qpos[self._q_arm]
        holgura = np.minimum(q - self._arm_lo, self._arm_hi - q)
        incursion = np.maximum(0.0, JOINT_LIMIT_MARGIN - holgura) / JOINT_LIMIT_MARGIN
        p_lim = min(W_JOINT_LIM * float(incursion.sum()), P_LIM_MAX)

        # singularidad: rampa lineal de WARN a STOP, saturada en 1
        cond = self.jac_cond()
        frac = np.clip((cond - SING_COND_WARN) / (SING_COND_STOP - SING_COND_WARN), 0.0, 1.0)
        p_sing = min(W_SINGULAR * float(frac), P_SING_MAX)

        return p_tcp, p_vel, p_acc, p_lim, p_sing

    # ── reward ─────────────────────────────────────────────────────────────
    def _reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Desglose FIRMADO: sum(terms de reward) == el valor devuelto, misma
        convencion que base_env.compute_reward para que una tabla de desglose
        por componente cuadre sin trucos. Las claves que no son reward
        (dist, frac, cond, n_*) van aparte, en `info`."""
        qvel = self.data.qvel[self._v_arm].copy()

        terms = {k: 0.0 for k in REWARD_TERMS}

        # 1. Movimiento anormal — se paga SIEMPRE, tambien en pasos con choque.
        p_tcp, p_vel, p_acc, p_lim, p_sing = self._penal_movimiento(qvel)
        self._prev_qvel = qvel
        terms["tcp_speed"] = -p_tcp
        terms["joint_vel"] = -p_vel
        terms["joint_acc"] = -p_acc
        terms["joint_lim"] = -p_lim
        terms["singular"]  = -p_sing
        p_mov = p_tcp + p_vel + p_acc + p_lim + p_sing

        # 2. Choques
        (n_torre, n_auto, n_suelo, dedo_ok,
         n_ajeno, n_no_garra) = self._clasificar_contactos()
        if n_ajeno and "ajeno" in self.cortes:
            self._fin_por = "toco un boton ajeno"
        elif n_no_garra and "no_garra" in self.cortes:
            self._fin_por = "toco el boton con algo que no es la garra"
        elif n_auto and "auto" in self.cortes:
            self._fin_por = "autocolision del brazo"

        # Hundimiento atribuible a LA GARRA: solo cuenta el recorrido logrado
        # mientras un dedo esta en contacto con la cara del boton objetivo.
        if dedo_ok:
            self._frac_con_dedo = max(self._frac_con_dedo, self.press_frac())
            self._tocado = True

        # Racha de SOSTENIDO: pasos SEGUIDOS con el boton hundido a fondo Y la
        # garra en contacto. Si se suelta o se pierde el contacto, vuelve a cero.
        frac_now = self.press_frac()
        umbral = HOLD_KEEP_FRAC if self._hold > 0 else PRESS_THRESH_FRAC
        if dedo_ok and frac_now >= umbral:
            self._hold += 1
        else:
            self._hold = 0
        self._hold_best = max(self._hold_best, self._hold)
        p_col = (P_ROCE_TORRE * (n_torre > 0)
                 + P_AUTOCOLISION * (n_auto > 0)
                 + P_BOTON_AJENO * (n_ajeno + n_no_garra > 0)
                 + P_BRAZO_SUELO * (n_suelo > 0))
        terms["col_torre"]    = -P_ROCE_TORRE * (n_torre > 0)
        terms["autocolision"] = -P_AUTOCOLISION * (n_auto > 0)
        terms["boton_ajeno"]  = -P_BOTON_AJENO * (n_ajeno + n_no_garra > 0)
        terms["brazo_suelo"]  = -P_BRAZO_SUELO * (n_suelo > 0)

        dist = float(np.linalg.norm(self.target_pos() - self.tcp()))
        frac = self.press_frac()
        info = dict(dist=dist, frac=frac, frac_dedo=self._frac_con_dedo,
                    cond=self.jac_cond(), v_tcp=self._v_tcp, n_col_torre=n_torre,
                    n_autocolision=n_auto, n_brazo_suelo=n_suelo,
                    n_boton_ajeno=n_ajeno, n_no_garra=n_no_garra,
                    hold_s=self._hold * self._dt,
                    hold_best_s=self._hold_best * self._dt)

        # 3. Exito. Se evalua SIEMPRE, tambien en un paso con choque: hundir el
        #    boton es la mision, y haberlo hecho sucio ya se paga aparte con el
        #    castigo de colision. Antes esto vivia despues del retorno temprano
        #    del punto 4 y habia episodios que dejaban el boton hundido al 100%
        #    marcados como FALLO, porque en cada paso con el boton hundido habia
        #    tambien un contacto del codo. El bonus es de un solo cobro, asi que
        #    pagarlo en un paso con choque no abre ningun exploit.
        # La condicion de exito NO es "el boton se hundio": es "LA GARRA lo
        # hundio". Sin esta puerta, una politica entrenada encontro el atajo de
        # empujar el boton con link_6 (la muñeca) -- medido: 8 de 10 contactos
        # eran link_6 contra la cara del boton, y cobraba el bonus entero. El
        # castigo por tocar con la parte equivocada (-0.30/paso, ~-3 por
        # episodio) no compensaba un bonus de +100, asi que salia rentable.
        # Subir el castigo seria pelearse con el sintoma; lo correcto es que
        # empujar con el codo sencillamente no cuente como pulsacion.
        if (not self._pressed) and self._hold >= HOLD_STEPS:
            self._pressed = True
            terms["bonus"] = PRESS_BONUS

        # 4. Un paso con choque no cobra el SHAPING (progress/press). Mismo patron
        #    que OBSTACLE_PENALTY en base_env: si no, la politica aprende a
        #    empotrarse contra la torre mientras sigue cobrando el progreso. Esos
        #    dos son los farmeables; el bonus no. _prev_dist se actualiza igual,
        #    o la distancia "saltada" se cobraria entera en el siguiente paso
        #    limpio, que seria justo el exploit que se quiere cerrar.
        if p_col > 0.0:
            self._prev_dist = dist
            return sum(terms.values()), {**info, **terms}

        # 5. Shaping POTENCIAL: telescopa a cero en un ciclo cerrado -> no se
        #    puede farmear oscilando (el exploit documentado en la Fase 6).
        terms["progress"] = W_PROGRESS * (self._prev_dist - dist)
        self._prev_dist = dist
        # ESCALONES hacia el sostenido, todos ACOTADOS por episodio.
        #
        # Por que no se pagan por paso: si `press` valiera W_PRESS cada paso con
        # el boton hundido, y `hold` valiera algo fijo por paso sostenido, a la
        # politica le saldria mas rentable hundir-soltar-hundir durante los 700
        # pasos que completar los 2 s y terminar. Hecha la cuenta con un pago por
        # paso plausible, nueve ciclos de 39 pasos rentaban mas que un episodio
        # perfecto. Es el mismo exploit de oscilacion que documenta la Fase 6 del
        # repo, con otra cara.
        #
        # Pagando el INCREMENTO sobre el mejor valor alcanzado, repetir no renta:
        # el total por episodio esta acotado por W_PRESS y W_HOLD pase lo que
        # pase, y solo se cobra por MEJORAR la marca. Sostener mas tiempo del que
        # nunca se sostuvo es la unica forma de seguir cobrando.
        if self._tocado and terms["touch"] == 0.0 and not self._cobro_touch:
            self._cobro_touch = True
            terms["touch"] = W_TOUCH
        terms["press"] = W_PRESS * (self._frac_con_dedo - self._prev_frac_dedo)
        self._prev_frac_dedo = self._frac_con_dedo
        terms["hold"] = W_HOLD * (self._hold_best - self._prev_hold_best) / HOLD_STEPS
        self._prev_hold_best = self._hold_best
        terms["action"] = -W_ACTION * float(np.square(action).mean())
        terms["alive"]  = ALIVE

        total = sum(terms.values())
        return total, {**info, **terms}

    # ── API publica ────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        # Modo del episodio + colocacion de los botones (torre o sueltos).
        # Tiene que ir DESPUES de que el brazo este en su sitio, porque el
        # muestreo de los sueltos se hace en el marco del HOMBRO.
        self._pendiente_colocar = True

        # brazo en la pose de reposo canonica (== config.ARM_REST_POSE)
        for name, q in ARM_REST_POSE.items():
            j = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[j]] = q
        self._joint_pos = np.array([ARM_REST_POSE[n] for n in ARM_JOINTS])
        self._qvel_cmd_prev = np.zeros(6)   # el episodio arranca desde parado
        d.ctrl[self._a_arm]  = self._joint_pos
        # Flippers plegados y fuera del camino del brazo (ver FLIPPER_STOW_RAD).
        d.qpos[self._q_flip] = FLIPPER_STOW_RAD
        d.ctrl[self._a_flip] = FLIPPER_STOW_RAD
        d.ctrl[self._a_fing] = FINGER_CLOSED
        d.ctrl[self._a_lidar] = 0.0

        # El chasis cae y se asienta con la base LIBRE (y la soldadura
        # DESACTIVADA, o lo sujetaria a la pose de referencia en vez de dejarlo
        # asentarse), y solo despues se fija.
        if self._eq_freeze >= 0:
            d.eq_active[self._eq_freeze] = 0
        d.qpos[self._base_qadr + 2] = 0.15
        self._grav_comp_arm()
        mujoco.mj_step(m, d, nstep=SETTLE_STEPS)
        self._base_qpos_frozen = d.qpos[self._base_qadr:self._base_qadr + 7].copy()
        if self.freeze_base and self._eq_freeze >= 0:
            self._anclar_weld(self._base_qpos_frozen)

        # Ya hay pose de chasis fiable -> se puede sortear donde van los botones.
        self._colocar_botones()

        # Poner a CERO el recorrido de TODOS los botones, activos y aparcados.
        #
        # Hace falta porque el asentamiento (SETTLE_STEPS pasos de fisica) corre
        # ANTES de colocarlos, con los flags de colision del episodio anterior.
        # Un boton aparcado tiene contype=0: no choca con nada, pero la gravedad
        # le sigue actuando y su cabeza cae libre por el eje del slide. El limite
        # del joint es una restriccion BLANDA, asi que no la frena: medido, el
        # joint llegaba a +0.682 m con un rango de 0..0.018 (38 veces el tope).
        #
        # El efecto era el que se veia en pantalla: press_frac recorta a [0,1],
        # asi que el boton aparecia "hundido al 100%" desde el paso 0 sin que
        # nadie lo tocara. Como no habia contacto de dedo, la racha nunca subia,
        # el episodio nunca terminaba por exito y el brazo parecia no hacer nada.
        todos_q = list(self._q_bot) + list(self._q_libre)
        todos_v = list(self._v_bot) + list(self._v_libre)
        d.qpos[todos_q] = 0.0
        d.qvel[todos_v] = 0.0
        mujoco.mj_forward(m, d)

        self._step_count = 0
        self._pressed    = False
        self._frac_con_dedo = 0.0
        self._prev_frac_dedo = 0.0
        self._hold = 0                  # pasos seguidos sosteniendo AHORA
        self._hold_best = 0             # mejor racha del episodio
        self._prev_hold_best = 0
        self._tocado = False            # la garra ha tocado la cara del boton
        self._cobro_touch = False       # ya se cobro el escalon de "tocar"
        self._fin_por    = ""
        self._cond_cache = None
        self._prev_qvel  = np.zeros(6)
        self._prev_tcp   = None
        self._v_tcp      = 0.0
        otros = [self._act_geoms[j] for j in range(self._n_activos) if j != self.target]
        self._g_ajenos = set().union(*otros) if otros else set()
        # Geoms que cuentan como "torre/boton" para clasificar contactos: los del
        # modo ACTIVO. Los aparcados a z=-5 no participan.
        self._g_tocable = (self._g_torre if self.modo == "torre"
                           else set().union(*self._act_geoms))
        # Pares robot<->robot que YA se tocan con el brazo en reposo: son parte
        # de la pose plegada, no colisiones. Se recalculan en cada reset porque
        # dependen de como haya quedado el asentamiento.
        self._pares_benignos = set()
        for k in range(d.ncon):
            g1, g2 = int(d.contact[k].geom1), int(d.contact[k].geom2)
            if g1 in self._g_robot and g2 in self._g_robot:
                self._pares_benignos.add(
                    frozenset((int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2]))))

        self._prev_dist  = float(np.linalg.norm(self.target_pos() - self.tcp()))

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        return self._observation()

    def _grav_comp_arm(self) -> None:
        """Compensa gravedad en los DOFs del brazo — mismo tratamiento que
        mujoco_sim_base._grav_comp_arm y el bridge de despliegue. Sin esto el
        brazo cede ~0.1 rad bajo su propio peso (medido) y el TCP real queda
        varios cm por detras del comando: el servo cartesiano nunca cierra."""
        if self.grav_comp:
            self.data.qfrc_applied[self._v_arm] = self.data.qfrc_bias[self._v_arm]

    def _anclar_weld(self, pose: np.ndarray) -> None:
        """Ancla la soldadura del chasis a `pose` (qpos del freejoint: xyz+quat).

        Hace falta porque el `relpose` que el XML deja por defecto es la pose de
        REFERENCIA del modelo (chasis a z=0.100), no la asentada (z=0.038): sin
        reanclar, el weld deja al robot suspendido 6.2 cm en el aire y sin tocar
        el suelo, lo que corre el brazo entero respecto a la torre.

        OJO con la convencion: `relpose` de un weld es la pose de body2 (aqui el
        MUNDO) en el frame de body1, o sea la INVERSA de la pose del chasis. Con
        la pose directa el robot se hunde y deriva 203 mm (medido)."""
        m, eid = self.model, self._eq_freeze
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, pose[3:7])
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, pose[3:7])
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = -(R.reshape(3, 3).T @ pose[0:3])
        m.eq_data[eid, 6:10] = q_inv
        m.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = 1

    def _hold_base(self) -> None:
        """Congelado del chasis por Python. Solo se usa como RESPALDO cuando el
        mundo no trae la soldadura `freeze_base`; con ella, el chasis lo sujeta
        el solver en C y aqui no hay nada que hacer."""
        if not self.freeze_base or self._base_qpos_frozen is None:
            return
        self.data.qpos[self._base_qadr:self._base_qadr + 7] = self._base_qpos_frozen
        self.data.qvel[self._base_vadr:self._base_vadr + 6] = 0.0

    def step(self, action: np.ndarray):
        self._apply_action(action)
        if self._eq_freeze >= 0 and self.freeze_base:
            # Un solo mj_step con nstep: el bucle de substeps corre en C y suelta
            # el GIL UNA vez, no 25. Con 8 envs en hilos eso es la diferencia
            # entre 1.27x y paralelismo real. El chasis lo sujeta la soldadura,
            # asi que no hace falta entrar en el bucle interno desde Python.
            # grav-comp una vez por paso de control: qfrc_applied persiste
            # (mismo tratamiento que mujoco_sim_base._grav_comp_arm).
            # Tramos de subpasos con anti-windup entre medias (ver
            # SUBPASOS_POR_TRAMO). Sin esto el desfase crece durante los 25
            # subpasos y la muñeca acaba forzada.
            restantes = self.control_decimation
            while restantes > 0:
                n = min(SUBPASOS_POR_TRAMO, restantes)
                self._grav_comp_arm()
                mujoco.mj_step(self.model, self.data, nstep=n)
                self._aplicar_antiwindup()
                restantes -= n
        else:
            for _ in range(self.control_decimation):
                self._hold_base()
                self._grav_comp_arm()
                mujoco.mj_step(self.model, self.data)
            self._hold_base()
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_cache()

        self._step_count += 1
        obs, (rew, info) = self._observation(), self._reward(np.clip(action, -1, 1))
        done = (self._pressed or bool(self._fin_por)
                or self._step_count >= self.max_steps)
        info["success"]  = self._pressed
        info["timeout"]  = (not self._pressed and not self._fin_por
                            and self._step_count >= self.max_steps)
        info["target"]   = self.target
        info["steps"]    = self._step_count
        # OJO: no llamar a esto "boton_ajeno". Ese nombre ya es una CLAVE DEL
        # REWARD, y step() mezcla {**info, **terms} y luego escribe encima: la
        # bandera booleana pisaba el castigo y el desglose reportaba True/False
        # en vez de -25.0 (visto en la tabla como un +2.0 imposible).
        info["fin_boton_ajeno"] = (self._fin_por == "toco un boton ajeno")
        info["fin_por"]  = self._fin_por
        info["hold_best_s"] = self._hold_best * self._dt
        info["modo"]        = self.modo
        info["tocado"]      = self._tocado
        info["frac_max"]    = self._frac_con_dedo
        info["n_botones"]   = self._n_activos
        info["reason"]   = (f"sostenido {HOLD_TIME_S:.0f}s" if self._pressed else
                            self._fin_por if self._fin_por else
                            f"tiempo agotado ({EPISODE_TIME_S:.0f}s)" if info["timeout"]
                            else "")

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        return obs, rew, done, info

    def close(self) -> None:
        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
