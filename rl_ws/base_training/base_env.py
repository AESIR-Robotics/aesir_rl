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

# Constantes de la tarea — TODAS viven en config.py; aqui solo se importan las
# que este modulo usa de verdad (los consumidores leen config.py directamente).
from .config import (
    V_REF_MPS, W_REF_RADPS,
    FLIPPER_MIN_RAD, FLIPPER_MAX_RAD, CONTROL_FLIPPERS, FLIPPER_HOME_RAD,
    FINISH_DIST,
    STUCK_TIMEOUT_STEPS, STUCK_NO_PROGRESS_M,
    W_DIRECTION, W_VELOCITY, WP_BONUS, GOAL_BONUS, TIME_PENALTY, FALL_PENALTY,
    FALL_UPRIGHT_MIN, FALL_Z_MIN,
    STUCK_MAX, STUCK_ANGULAR_THRESH_RAD, ENERGY_W, FLIPPER_JERK_W, FLIPPER_COLLISION_W,
    FLIPPER_TERRAIN_W,
    FLIPPER_TERRAIN_MIN_STEP_M, FLIPPER_TERRAIN_MAX_CLIMB_M,
    FLIPPER_TERRAIN_MAX_DESCENT_M, FLIPPER_TERRAIN_SAMPLES,
    ACCEL_W, ACCEL_DEADZONE, GUIDE_SPEED_SCALE, BACKWARD_W,
    FLIPPER_MOUNTS, FLIPPER_AXIS_SIGN, FLIPPER_L, FLIPPER_COLLISION_DIST,
    OBSTACLE_PENALTY,
)


# ── Gate de flippers (compartido por los dos backends) ───────────────────────
def flipper_targets(action) -> Optional[np.ndarray]:
    """Angulo objetivo (rad) de los 4 flippers segun la accion, o None si NO se
    deben comandar (quedan como esten).
      - CONTROL_FLIPPERS=False -> None (fase 1: flippers en reposo, no controlados).
      - action[6] = gate DISCRETO (Bernoulli, 0.0 o 1.0 exacto -- ver
        CNNActorCritic/MLPActorCritic): 0 -> reposo (FLIPPER_HOME_RAD),
        1 -> la politica controla via action[2:6].
      - action[2:6] = muestra de una Beta por flipper, YA en [0,1] (soporte
        acotado, sin necesidad de clip) -- se reescala afin a
        [FLIPPER_MIN_RAD, FLIPPER_MAX_RAD]. Antes esto era una Gaussiana
        recortada con np.clip, lo que genera una inconsistencia entre la
        accion cruda que ve el log-prob y la accion recortada que ejecuta el
        entorno; con Beta el soporte ya es exacto, no hace falta recortar."""
    if not CONTROL_FLIPPERS:
        return None
    if float(action[6]) < 0.5:
        return np.full(4, FLIPPER_HOME_RAD, dtype=np.float32)
    # clip defensivo: la Beta ya garantiza [0,1], pero esta funcion tambien la
    # llaman scripts/tests con acciones a mano, y de aqui sale el comando real.
    u = np.clip(np.asarray(action[2:6], dtype=np.float32), 0.0, 1.0)
    return FLIPPER_MIN_RAD + u * (FLIPPER_MAX_RAD - FLIPPER_MIN_RAD)


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
        self.last_yaw = 0.0
        self.last_dist_to_target = 0.0
        self.last_wp = 0
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.stuck = 0
        self.best_dist_goal = np.inf
        self.best_wp = 0
        self.no_progress_steps = 0

    def reset(self, xy: np.ndarray, dist_to_target: float, yaw: float = 0.0):
        self.last_xy = xy.copy()
        self.last_yaw = float(yaw)
        self.last_dist_to_target = dist_to_target
        self.last_wp = 0
        self.last_twist = np.zeros(3, dtype=np.float32)
        self.stuck = 0
        self.best_dist_goal = np.inf
        self.best_wp = 0
        self.no_progress_steps = 0


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


def flipper_edge(map_ctx, fb: dict) -> Optional[dict]:
    """Borde real (subida O bajada) en la direccion de extension de cada
    flipper -- ver flipper_terrain_bonus. Al retroceder (fb["twist"][0] < 0)
    delanteros y traseros intercambian su lado de escaneo, ver
    flipper_travel_dir. None si no hay map_ctx (USE_HEATMAP=False). Compartido
    por los dos backends: no toca ni MuJoCo ni ROS, solo map_ctx + config."""
    if map_ctx is None:
        return None
    found, d, h, actionable = map_ctx.get_flipper_terrain_edges(
        robot_xy=fb["xy"], robot_yaw=fb["yaw"],
        mounts_xy=FLIPPER_MOUNTS[:, :2],
        axis_sign=FLIPPER_AXIS_SIGN * flipper_travel_dir(fb),
        look_ahead_m=FLIPPER_L, min_step_m=FLIPPER_TERRAIN_MIN_STEP_M,
        max_climb_m=FLIPPER_TERRAIN_MAX_CLIMB_M,
        max_descent_m=FLIPPER_TERRAIN_MAX_DESCENT_M,
        n_samples=FLIPPER_TERRAIN_SAMPLES)
    return dict(found=found, d=d, h=h, actionable=actionable)


def flipper_travel_dir(fb: dict) -> float:
    """Signo de la direccion de avance en frame local: +1 adelante, -1 en
    reversa. Misma convencion que backward_pen (v_fwd < 0 = retrocediendo,
    ver compute_reward). Al retroceder, delanteros y traseros intercambian
    su rol de escaneo/alcance -- ver flipper_terrain_bonus y
    mujoco_sim_base._get_flipper_edge / base_ros_env._get_flipper_edge, que
    deben usar este MISMO signo para el axis_sign efectivo del escaneo."""
    return 1.0 if float(fb["twist"][0]) >= 0.0 else -1.0


def flipper_terrain_bonus(fb: dict, flip_qpos: np.ndarray) -> float:
    """Premia poner la PUNTA del flipper donde sirve respecto del borde real
    (d_edge, h_edge) que detecto elevation_map.flipper_terrain_edge. Criterio
    geometrico, no un angulo "correcto" inventado: la punta debe pasar el borde
    y quedar del lado util, asi el mismo criterio pide mas o menos angulo segun
    la distancia y altura del obstaculo.

    Punta relativa al mount (pivote sobre eje y, ver flipper_tips), medida a lo
    largo de la direccion de ESCANEO (axis_sign * travel_dir; axis_sign**2 == 1
    lo cancela, por eso solo queda travel_dir):
        reach  = travel_dir * FLIPPER_L * sin(theta)
        height = FLIPPER_L * cos(theta)
    Region factible, segun el signo del borde (misma derivacion, espejada):
        SUBIDA (h>0): reach >= d  Y  height >= h   -- libra el escalon por arriba
        BAJADA (h<0): reach >= d  Y  height <= h   -- alcanza el suelo inferior,
                      baja controlado en vez de cabecear (exige |theta| > 90)

    DENSO y ANCLADO en vez de indicador binario:
        calidad(th) = clip(1 - hypot(max(0,d-reach), s_height)/FLIPPER_L, 0, 1)
        score       = (calidad(th) - calidad(reposo)) / (1 - calidad(reposo))
    Denso porque el indicador solo acertaba ~3% de las veces -> sin gradiente el
    97% restante (mismo enfoque que R_flipper de arXiv 2306.10352). Anclado
    porque sin ello el 66% del bonus se cobraba con los flippers EN REPOSO
    (medido 0.333 de 0.503), premiando merodear el borde sin actuar. calidad==1
    equivale al criterio binario, asi que es una generalizacion, no otro criterio.

    0.0 si no hay heightmap o si ningun flipper tiene borde atacable cerca."""
    edge = fb.get("flipper_edge")
    if edge is None:
        return 0.0
    actionable = edge["actionable"]
    if not actionable.any():         # caso comun (terreno plano) -- corta antes del trig
        return 0.0
    travel_dir = flipper_travel_dir(fb)
    d_edge, h_edge = edge["d"], edge["h"]

    def _quality(theta):
        reach = travel_dir * FLIPPER_L * np.sin(theta)
        height = FLIPPER_L * np.cos(theta)
        s_reach = np.maximum(0.0, d_edge - reach)
        s_height = np.where(h_edge > 0,
                            np.maximum(0.0, h_edge - height),    # subida
                            np.maximum(0.0, height - h_edge))    # bajada
        return np.clip(1.0 - np.hypot(s_reach, s_height) / FLIPPER_L, 0.0, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        q = _quality(np.asarray(flip_qpos, dtype=np.float64))
        q_rest = _quality(np.full(4, FLIPPER_HOME_RAD, dtype=np.float64))
        # los no-atacables tienen d/h = nan -> se anulan aqui (nan no se propaga)
        score = np.where(actionable & (q_rest < 1.0 - 1e-6),
                         np.maximum(0.0, (q - q_rest) / (1.0 - q_rest)), 0.0)
    n_act = int(np.count_nonzero(actionable))
    return FLIPPER_TERRAIN_W * float(np.sum(score) / max(n_act, 2))


# ── Caida (condicion UNICA, compartida por reward y terminacion) ─────────────
def is_fallen(fb: dict) -> bool:
    """El robot se cayo: volcado (inclinacion excesiva), chasis demasiado bajo,
    o tocando el piso. UNA sola definicion para que compute_reward (que cobra
    FALL_PENALTY) y terminated (que corta el episodio) usen EXACTAMENTE los
    mismos umbrales -- si no, terminar por volcarse saldria gratis y seria un
    escape barato de un episodio con retorno futuro negativo.

    El umbral de altura es POR PISTA (fb["fall_z_min"], que ponen los backends
    desde TRACK_DEFS; default = FALL_Z_MIN). No puede ser global: mide la altura
    ABSOLUTA del chasis, y cada pista tiene su suelo a distinta z. En maze el
    piso caminable es z=0 y el chasis reposa a 0.036 -> con el default de 0.10
    el robot nacia ya "caido" y el episodio moria en el paso 1."""
    return (fb["upright"] < FALL_UPRIGHT_MIN
            or fb["z"] < fb.get("fall_z_min", FALL_Z_MIN)
            or fb.get("floor_contact", 0) > 0)


# ── Reward ───────────────────────────────────────────────────────────────────
def compute_reward(fb: dict, guidance: dict, action: np.ndarray, rs: RewardState,
                   goal_xy: np.ndarray) -> float:
    """Caida letal (volcado/muy bajo/piso) o choque con el obstaculo (retorno
    temprano, termina el episodio) -> castigos (stuck, energia, jerk flippers,
    auto-colision flippers) -> bonus al cruzar waypoint (retorno temprano) ->
    progreso (delta_dist * boost exponencial de proximidad) + seguir
    velocidad/direccion de la guia del vortex (solo si avanza), y GOAL_BONUS
    si se completo la mision.

    `goal_xy` debe ser EL MISMO que recibe terminated(): el bonus terminal usa
    su misma condicion (dist_goal < FINISH_DIST) para que reward y metrica de
    exito coincidan exactamente. No vale usar guidance["goal"] en su lugar --
    en las pistas de pallets la ruta A* redondea a una grilla de 5 cm
    (GRID_RESOLUTION), asi que el ultimo waypoint puede no ser el goal exacto y
    el bonus se cobraria en un paso distinto al que cuenta como exito."""
    xy = fb["xy"]
    dist_norm, sin_t, cos_t = guidance["obs"]
    target_xy = np.asarray(guidance["target"], dtype=np.float64)
    v_fwd = float(fb["twist"][0])

    # 1. Caida letal: volcado, chasis muy bajo o tocar piso -- MISMAS condiciones
    #    que corta terminated(), asi caerse SIEMPRE cuesta FALL_PENALTY.
    if is_fallen(fb):
        return -FALL_PENALTY

    if fb.get("obstacle_contact", 0) > 0:
        return -OBSTACLE_PENALTY

    # 2. Inactividad: SOLO cuenta "atascado" si NI se traslada NI gira 
    move_dist = float(np.linalg.norm(xy - rs.last_xy))
    yaw = float(fb["yaw"])
    dyaw = abs((yaw - rs.last_yaw + np.pi) % (2.0 * np.pi) - np.pi)   # diff angular corta
    rs.last_xy = xy.copy()
    rs.last_yaw = yaw
    if move_dist < 0.005 and dyaw < STUCK_ANGULAR_THRESH_RAD:
        rs.stuck += 1
        penalty_stuck = min(STUCK_MAX, 0.01 * rs.stuck)
    else:
        rs.stuck = 0
        penalty_stuck = 0.0

    # 3. Costo de energia
    action_cost = ENERGY_W * float(np.square(action).mean())

    # 4. Movimiento erratico de flippers: velocidad angular REAL medida
    #    (fb["flip_qvel"], ya limitada por la rampa FLIPPER_MAX_VEL/ACCEL), no
    #    el delta del target comandado -- ese delta es ruido de muestreo de la
    #    politica estocastica (target resampleado cada step) y no refleja
    #    movimiento fisico real, lo que antes inflaba este castigo muy por
    #    encima de flipper_terrain_bonus incluso con el flipper quieto.
    flipper_pen = FLIPPER_JERK_W * float(np.square(fb["flip_qvel"]).mean())

    # 5. (La inclinacion ya no tiene castigo graduado: pasar FALL_UPRIGHT_MIN es
    #    caida letal -- FALL_PENALTY + fin de episodio, ver is_fallen arriba.)

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

    penalties = (penalty_stuck + action_cost + flipper_pen
                 + flipper_collision_pen + accel_pen)

    # 7b. Bonus por extender flippers cerca de terreno trepable (ver docstring
    #     de flipper_terrain_bonus). Se suma siempre, no es una "penalizacion".
    terrain_bonus = flipper_terrain_bonus(fb, fb["flip_qpos"])

    # 7c. Mision completada -> bonus TERMINAL. Condicion identica a la de
    #     terminated() para que el cobro caiga en el MISMO paso que cuenta como
    #     exito. Se suma a las dos salidas de abajo porque el paso en que se
    #     llega es tambien, casi siempre, el paso en que se cruza el ultimo
    #     waypoint (la meta ES ese waypoint), que retorna antes.
    goal_bonus = (GOAL_BONUS
                  if (float(np.linalg.norm(xy - np.asarray(goal_xy, dtype=np.float64)))
                      < fb.get("finish_dist", FINISH_DIST))
                  else 0.0)

    # 8. Waypoint cruzado -> solo bonus + penalizaciones ese paso
    if guidance["wp"] > rs.last_wp:
        rs.last_wp = guidance["wp"]
        rs.last_dist_to_target = float(np.linalg.norm(xy - target_xy))
        return WP_BONUS - penalties + terrain_bonus + goal_bonus

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

    return (progress_reward + direction_reward + speed_reward + terrain_bonus
            + goal_bonus - backward_pen - penalties - TIME_PENALTY)


# ── Terminacion ──────────────────────────────────────────────────────────────
def terminated(fb: dict, goal_xy: np.ndarray, ep_steps: int,
               max_steps: int, rs: "RewardState") -> Tuple[bool, bool, Optional[str]]:
    """Devuelve (done, reached_goal, reason). 'reason' es un texto corto para
    logear (o None si no termino) — el caller decide si imprimir (util para no
    spamear con N envs en paralelo). Las tres condiciones de caida usan los
    MISMOS umbrales que is_fallen() (que cobra FALL_PENALTY); aqui se checan
    por separado solo para poder logear el motivo exacto."""
    if ep_steps >= max_steps:
        return True, False, "limite de pasos"
    if fb["upright"] < FALL_UPRIGHT_MIN:
        return True, False, "caida (demasiado inclinado)"
    if fb["z"] < fb.get("fall_z_min", FALL_Z_MIN):
        return True, False, "caida (demasiado bajo)"
    if fb.get("floor_contact", 0) > 0:
        return True, False, "toco el piso"
    #if fb.get("obstacle_contact", 0) > 0:
    #    return True, False, "choco con el obstaculo"
    dist_goal = float(np.linalg.norm(fb["xy"] - goal_xy))
    if dist_goal < fb.get("finish_dist", FINISH_DIST):
        return True, True, "META alcanzada"
    # "Progreso" = acercarse a la meta O avanzar por la RUTA. Lo segundo no es
    # un extra: en un laberinto rodear una pared ALEJA en linea recta de la meta,
    # asi que con solo dist_goal el contador de atasco corre durante cada rodeo
    # legitimo y mata el episodio (medido en maze: cortaba a 554 pasos con el
    # robot avanzando bien por su ruta). rs.last_wp lo actualiza compute_reward,
    # que en los dos backends corre ANTES que terminated() en el mismo paso.
    if rs.last_wp > rs.best_wp:
        rs.best_wp = rs.last_wp
        rs.no_progress_steps = 0
    elif dist_goal < rs.best_dist_goal - STUCK_NO_PROGRESS_M:
        rs.best_dist_goal = dist_goal
        rs.no_progress_steps = 0
    else:
        rs.no_progress_steps += 1
        if rs.no_progress_steps >= STUCK_TIMEOUT_STEPS:
            return True, False, "atascado (sin progreso)"
    return False, False, None