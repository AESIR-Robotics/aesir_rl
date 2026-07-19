"""
base_env.py — Logica de "mundo" compartida por los backends del env base.

Contiene TODO lo que define la tarea y es independiente de como se simula la
fisica: observacion, reward y terminacion. Son funciones PURAS que reciben un
dict `fb` (feedback cinematico) + `guidance` (del global_navigator) — no
dependen de ROS ni de MuJoCo. Todas las constantes ajustables (mision, pesos
de reward, escalas de comando, geometria) viven en config.py y aqui solo se
re-exportan por compatibilidad. Asi las comparten:

  - base_ros_env.py     (BaseRosEnv, backend ROS2/bridge — despliegue/MoveIt)
  - mujoco_sim_base.py  (BaseMujocoEnv, backend MuJoCo directo — entrenamiento
                         rapido, sin tiempo real, paralelizable por threads)

El dict `fb` que ambos backends deben producir:
    xy            np.array([x, y])        posicion del chasis (frame mapa)
    z             float                   altura del chasis
    yaw           float                   heading
    upright       float                   R[2,2] del chasis (1=vertical)
    twist         np.array([v_fwd, v_lat, omega_z])   velocidad en frame local
    grav_body     np.array(3)             gravedad unitaria en frame del cuerpo
                                          (pitch/roll/vert; para pendientes)
    flip_qpos     np.array(4)             angulos de flipper (rad, ROS)
    flip_qvel     np.array(4)             velocidades de flipper
    floor_contact int                     nº de contactos robot<->piso
    obstacle_contact int (opcional)       nº de contactos robot<->obstaculo virtual
                                          fisico (solo mujoco_sim_base.py; se lee con
                                          fb.get(..., 0), base_ros_env.py no lo produce)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Constantes de la tarea — TODAS viven en config.py; se re-exportan aqui para
# que `from base_training.base_env import START_XY, ...` siga funcionando.
from .config import (
    NAV_JSON,
    V_MAX_MPS, W_MAX_RADPS, V_REF_MPS, W_REF_RADPS,
    FLIPPER_MAX, FLIPPER_MIN_RAD, FLIPPER_MAX_RAD,
    START_XY, GOAL_XY, SPAWN_Z, FINISH_DIST, EPISODE_MAX_STEPS,
    W_DIRECTION, W_VELOCITY, WP_BONUS, TIME_PENALTY, FALL_PENALTY,
    STUCK_MAX, ENERGY_W, FLIPPER_JERK_W, TILT_W, FLIPPER_COLLISION_W,
    ACCEL_W, ACCEL_DEADZONE, GUIDE_SPEED_SCALE, BACKWARD_W,
    FLIPPER_MOUNTS, FLIPPER_AXIS_SIGN, FLIPPER_L, FLIPPER_COLLISION_DIST,
    N_LOOKAHEAD, OBS_DIM, ACT_DIM, OBSTACLE_PENALTY,
)


# ── Observacion ──────────────────────────────────────────────────────────────
def build_obs(guidance: dict, fb: dict, heatmap: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Sin heatmap: vector plano de siempre, shape (OBS_DIM,) -- 100% igual
    que antes, cero cambios de comportamiento.

    Con heatmap (minimapa por altura real, ver map_context.py): el heatmap
    se APLANA (flatten) y se concatena al final del vector de siempre.

    OJO: esto es a proposito, NO es un dict. train_fast.py (b_obs, GAE,
    ppo_update) asume en todos lados que obs es un vector plano de tamano
    fijo OBS_DIM -- si regresaramos un dict aqui, habria que reescribir
    ppo_update/compute_gae tambien (viven en ppo.py, que no tenemos a la
    vista). Aplanando, OBS_DIM simplemente crece de
    `15 + 3*N_LOOKAHEAD`  a  `15 + 3*N_LOOKAHEAD + H*W`
    y CERO codigo de train_fast.py necesita cambiar -- solo el valor de
    OBS_DIM en config.py, y la red (ver ppo_cnn_extractor.py) que
    "desdobla" la cola del vector de vuelta a imagen (H,W) adentro de
    su forward().
    """
    twist = np.asarray(fb["twist"], dtype=np.float32)
    state = np.concatenate([
        guidance["obs"],                                    # 3   guia inmediata (vortex)
        guidance["lookahead"],                              # 3*N puntos futuros de la ruta
        guidance["target_obs"],                             # 3   waypoint-objetivo crudo (relativo)
        [twist[0], twist[2]],                               # 2   [v_fwd, omega_z] (sin v_lat)
        fb["flip_qpos"],                                    # 4
        fb["flip_qvel"],                                    # 4
        fb["grav_body"],                                    # 3   gravedad en cuerpo (pitch/roll/vert)
    ]).astype(np.float32)

    if heatmap is None:
        return state

    return np.concatenate([state, heatmap.astype(np.float32).ravel()])


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
    """Caida/piso letal o choque con el obstaculo (retorno temprano, termina el
    episodio) -> castigos (stuck, energia, jerk flippers, inclinacion,
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

    if fb.get("obstacle_contact", 0) > 0:
        return -OBSTACLE_PENALTY

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
    #    cambio del twist base entre pasos, normalizado por la velocidad REAL
    #    [V_REF, V_REF, W_REF] (twist es medido, no comando -> ref real, no escala).
    #    Zona muerta -> deja pasar la aceleracion normal (ir rapido); solo se
    #    castiga el exceso al cuadrado (jolts que dan;an el robot).
    twist = np.asarray(fb["twist"], dtype=np.float32)
    accel = (twist - rs.last_twist) / np.array([V_REF_MPS, V_REF_MPS, W_REF_RADPS],
                                                dtype=np.float32)
    rs.last_twist = twist.copy()
    accel_pen = ACCEL_W * max(0.0, float(np.linalg.norm(accel)) - ACCEL_DEADZONE) ** 2

    penalties = (penalty_stuck + action_cost + flipper_pen + tilt_pen
                 + flipper_collision_pen + accel_pen)

    # 8. Waypoint cruzado -> solo bonus + penalizaciones ese paso
    if guidance["wp"] > rs.last_wp:
        rs.last_wp = guidance["wp"]
        rs.last_dist_to_target = float(np.linalg.norm(xy - target_xy))
        return WP_BONUS - penalties

    # 9. Progreso hacia el waypoint actual
    dist_to_target = float(np.linalg.norm(xy - target_xy))
    delta_dist = rs.last_dist_to_target - dist_to_target
    proximity_multiplier = float(np.exp(-dist_to_target))
    progress_reward = delta_dist * (50.0 + 100.0 * proximity_multiplier)
    rs.last_dist_to_target = dist_to_target

    # 10. Alcanzar la velocidad y orientacion que pide el vortex + castigo por
    #    retroceder. La DISTANCIA al punto-guia es la velocidad deseada (lejos ->
    #    rapido, cerca -> lento, p.ej. al llegar) y cos_t la orientacion. Se premia
    #    acercarse a esa velocidad encarando la guia; retroceder (v_fwd<0) se
    #    castiga -> evita la oscilacion.
    d_guide = float(np.linalg.norm(np.asarray(guidance["vortex"], dtype=float) - xy))
    v_des = V_REF_MPS * min(d_guide / GUIDE_SPEED_SCALE, 1.0)
    speed_reward = W_VELOCITY * (1.0 - abs(v_fwd - v_des) / V_REF_MPS) * max(0.0, float(cos_t))
    # Encarar la guia: cos_t>0 premia ir de frente, cos_t<0 (de espaldas) castiga
    # -> rompe la simetria adelante/reversa del chasis y evita que aprenda a ir
    # en reversa hacia la meta cobrando el progreso (que es ciego a la orientacion).
    direction_reward = W_DIRECTION * float(cos_t)
    backward_pen = BACKWARD_W * max(0.0, -v_fwd) / V_REF_MPS

    return (progress_reward + direction_reward + speed_reward
            - backward_pen - penalties - TIME_PENALTY)


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
    #if fb.get("obstacle_contact", 0) > 0:
    #    return True, False, "choco con el obstaculo"
    if float(np.linalg.norm(fb["xy"] - goal_xy)) < FINISH_DIST:
        return True, True, "META alcanzada"
    return False, False, None