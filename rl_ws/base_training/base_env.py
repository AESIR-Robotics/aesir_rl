"""
world_base.py — Logica de "mundo" compartida por los backends del env base.

Contiene TODO lo que define la tarea y es independiente de como se simula la
fisica: constantes de mision, pesos de reward, geometria de flippers,
observacion, reward y terminacion. Son funciones PURAS que reciben un dict `fb`
(feedback cinematico) + `guidance` (del global_navigator) — no dependen de ROS
ni de MuJoCo. Asi las comparten:

  - base_env.py        (BaseRosEnv, backend ROS2/bridge — despliegue/MoveIt)
  - base_mujoco_env.py (BaseMujocoEnv, backend MuJoCo directo — entrenamiento
                        rapido, sin tiempo real, paralelizable por threads)

El dict `fb` que ambos backends deben producir:
    xy            np.array([x, y])        posicion del chasis (frame mapa)
    z             float                   altura del chasis
    yaw           float                   heading
    upright       float                   R[2,2] del chasis (1=vertical)
    twist         np.array([v_fwd, v_lat, omega_z])   velocidad en frame local
    flip_qpos     np.array(4)             angulos de flipper (rad, ROS)
    flip_qvel     np.array(4)             velocidades de flipper
    floor_contact int                     nº de contactos robot<->piso
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
NAV_JSON = os.path.join(_HERE, "..", "obstacles.json")

# ── Escalas de comando ───────────────────────────────────────────────────────
V_MAX_MPS   = 0.6      # linear.x  a v_norm = 1
W_MAX_RADPS = 1.5      # angular.z a w_norm = 1
FLIPPER_MAX = 3.1416   # rad a flip = 1
# Limite de recorrido de los flippers (por software)
FLIPPER_MIN_RAD = -1.3
FLIPPER_MAX_RAD = 3.14159

# ── Mision (frame de aesir_complete.xml == frame de obstacles.json) ──────────
START_XY: Tuple[float, float] = (-1.5, 3.5)
GOAL_XY:  Optional[Tuple[float, float]] = None   # None -> ultimo pallet del JSON
SPAWN_Z   = 0.20
FINISH_DIST       = 0.10
EPISODE_MAX_STEPS = 15000

# ── Pesos de reward ──────────────────────────────────────────────────────────
W_DIRECTION    = 0.6     # encarar al objetivo (cos Δθ)
W_VELOCITY     = 0.6     # igualar la velocidad forward objetivo
WP_BONUS       = 200.0   # bonus al cruzar un waypoint
TIME_PENALTY   = 0.1
FALL_PENALTY   = 250.0
STUCK_MAX      = 1.0
ENERGY_W       = 1e-8
FLIPPER_JERK_W = 1.0
TILT_W         = 5.0
FLIPPER_COLLISION_W = 50.0
# Castigo por aceleraciones FUERTES del chasis (cuidar la integridad del robot
# en terreno dificil).
ACCEL_W        = 1.5
ACCEL_DEADZONE = 0.3
# Velocidad deseada = DISTANCIA al punto-guia del vortex (lejos -> rapido, cerca
# -> lento). Se premia alcanzarla encarando la guia; retroceder se castiga.
GUIDE_SPEED_SCALE = 1.0   # [m] distancia del guia que ya pide velocidad plena V_MAX
BACKWARD_W        = 2.0   # castigo por retroceder (x fraccion de V_MAX en reversa)

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

# ── Tamaños expuestos a la politica ──────────────────────────────────────────
# guia(3) + lookahead(3*N) + twist_base(3) + flipper_qpos(4) + flipper_qvel(4) + upright(1)
OBS_DIM = 15 + 3 * N_LOOKAHEAD
ACT_DIM = 6    # v, ω, flipper×4


# ── Observacion ──────────────────────────────────────────────────────────────
def build_obs(guidance: dict, fb: dict) -> np.ndarray:
    return np.concatenate([
        guidance["obs"],                                    # 3   guia inmediata
        guidance["lookahead"],                              # 3*N puntos futuros
        fb["twist"],                                        # 3  [v_fwd, v_lat, omega]
        fb["flip_qpos"],                                    # 4
        fb["flip_qvel"],                                    # 4
        [fb["upright"]],                                    # 1
    ]).astype(np.float32)


# ── Estado entre pasos para el reward ────────────────────────────────────────
class RewardState:
    def __init__(self):
        self.last_xy = None
        self.last_dist_to_target = 0.0
        self.last_wp = 0
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.stuck = 0

    def reset(self, xy: np.ndarray, dist_to_target: float):
        self.last_xy = xy.copy()
        self.last_dist_to_target = dist_to_target
        self.last_wp = 0
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.last_flip = np.zeros(4, dtype=np.float32)
        self.stuck = 0


# ── Auto-colision de flippers ────────────────────────────────────────────────
def flipper_tips(flip_qpos: np.ndarray) -> np.ndarray:
    """Punta 3D de cada flipper dado su angulo (pivote sobre eje y):
        tip = mount + (axis_sign * L * sin(theta), 0, L * cos(theta))"""
    th = np.asarray(flip_qpos, dtype=np.float64)
    tips = FLIPPER_MOUNTS.copy()
    tips[:, 0] += FLIPPER_AXIS_SIGN * FLIPPER_L * np.sin(th)
    tips[:, 2] += FLIPPER_L * np.cos(th)
    return tips


def flipper_collision_penalty(flip_qpos: np.ndarray) -> float:
    """Castigo proporcional a la penetracion por cada par de puntas mas cercano
    que FLIPPER_COLLISION_DIST. Los pares de lados y opuestos nunca disparan."""
    tips = flipper_tips(flip_qpos)
    pen = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(tips[i] - tips[j]))
            if d < FLIPPER_COLLISION_DIST:
                pen += FLIPPER_COLLISION_DIST - d
    return FLIPPER_COLLISION_W * pen


# ── Reward ───────────────────────────────────────────────────────────────────
def compute_reward(fb: dict, guidance: dict, action: np.ndarray, rs: RewardState) -> float:
    """Caida/piso letal -> castigos (stuck, energia, jerk flippers, inclinacion,
    auto-colision flippers) -> bonus al cruzar waypoint (retorno temprano) ->
    progreso (delta_dist * boost exponencial de proximidad) + seguir
    velocidad/direccion de la guia del vortex (solo si avanza)."""
    xy = fb["xy"]
    dist_norm, sin_t, cos_t = guidance["obs"]
    target_xy = np.asarray(guidance["target"], dtype=np.float64)
    v_fwd = float(fb["twist"][0])

    # 1. Caida letal (chasis muy bajo) o tocar piso (salirse de los pallets)
    if fb["z"] < 0.10 or fb.get("floor_contact", 0) > 0:
        return -FALL_PENALTY

    # 2. Inactividad
    move_dist = float(np.linalg.norm(xy - rs.last_xy))
    rs.last_xy = xy.copy()
    if move_dist < 0.005:
        rs.stuck += 1
        penalty_stuck = min(STUCK_MAX, 0.01 * rs.stuck)
    else:
        rs.stuck = 0
        penalty_stuck = 0.0

    # 3. Costo de energia
    action_cost = ENERGY_W * float(np.square(action).mean())

    # 4. Movimiento erratico de flippers
    current_flipper = action[2:6].astype(np.float32)
    flipper_pen = FLIPPER_JERK_W * float(np.square(current_flipper - rs.last_flip).mean())
    rs.last_flip = current_flipper.copy()

    # 5. Inclinacion
    tilt_pen = max(0.0, 0.65 - float(fb["upright"])) * TILT_W

    # 6. Auto-colision de flippers (angulos REALES medidos)
    flipper_collision_pen = flipper_collision_penalty(fb["flip_qpos"])

    # 7. Aceleraciones FUERTES del chasis (cuidar la integridad del robot):
    #    cambio del twist base entre pasos, normalizado por [V_MAX, V_MAX, W_MAX].
    #    Zona muerta -> deja pasar la aceleracion normal (ir rapido); solo se
    #    castiga el exceso al cuadrado (jolts que dan;an el robot).
    twist = np.asarray(fb["twist"], dtype=np.float32)
    accel = (twist - rs.last_twist) / np.array([V_MAX_MPS, V_MAX_MPS, W_MAX_RADPS],
                                                dtype=np.float32)
    rs.last_twist = twist.copy()
    accel_pen = ACCEL_W * max(0.0, float(np.linalg.norm(accel)) - ACCEL_DEADZONE) ** 2

    penalties = (penalty_stuck + action_cost + flipper_pen + tilt_pen
                 + flipper_collision_pen + accel_pen)

    # 7. Waypoint cruzado -> solo bonus + penalizaciones ese paso
    if guidance["wp"] > rs.last_wp:
        rs.last_wp = guidance["wp"]
        rs.last_dist_to_target = float(np.linalg.norm(xy - target_xy))
        return WP_BONUS - penalties

    # 8. Progreso hacia el waypoint actual
    dist_to_target = float(np.linalg.norm(xy - target_xy))
    delta_dist = rs.last_dist_to_target - dist_to_target
    proximity_multiplier = float(np.exp(-dist_to_target))
    progress_reward = delta_dist * (50.0 + 100.0 * proximity_multiplier)
    rs.last_dist_to_target = dist_to_target

    # 9. Alcanzar la velocidad y orientacion que pide el vortex + castigo por
    #    retroceder. La DISTANCIA al punto-guia es la velocidad deseada (lejos ->
    #    rapido, cerca -> lento, p.ej. al llegar) y cos_t la orientacion. Se premia
    #    acercarse a esa velocidad encarando la guia; retroceder (v_fwd<0) se
    #    castiga -> evita la oscilacion.
    d_guide = float(np.linalg.norm(np.asarray(guidance["vortex"], dtype=float) - xy))
    v_des = V_MAX_MPS * min(d_guide / GUIDE_SPEED_SCALE, 1.0)
    speed_reward = W_VELOCITY * (1.0 - abs(v_fwd - v_des) / V_MAX_MPS) * max(0.0, float(cos_t))
    backward_pen = BACKWARD_W * max(0.0, -v_fwd) / V_MAX_MPS

    return progress_reward + speed_reward - backward_pen - penalties - TIME_PENALTY


# ── Terminacion ──────────────────────────────────────────────────────────────
def terminated(fb: dict, goal_xy: np.ndarray, ep_steps: int,
               max_steps: int) -> Tuple[bool, bool, Optional[str]]:
    """Devuelve (done, reached_goal, reason). 'reason' es un texto corto para
    logear (o None si no termino) — el caller decide si imprimir (util para no
    spamear con N envs en paralelo)."""
    if ep_steps >= max_steps:
        return True, False, "limite de pasos"
    if fb["upright"] < 0.20:
        return True, False, "caida (demasiado inclinado)"
    if fb["z"] < 0.10:
        return True, False, "caida (demasiado bajo)"
    if fb.get("floor_contact", 0) > 0:
        return True, False, "toco el piso"
    if float(np.linalg.norm(fb["xy"] - goal_xy)) < FINISH_DIST:
        return True, True, "META alcanzada"
    return False, False, None
