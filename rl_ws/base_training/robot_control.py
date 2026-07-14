#!/usr/bin/env python3
"""
robot_control.py — Interfaz de control de actuadores COMPARTIDA (sin ROS).

Cinematica diferencial de las orugas + rampas trapezoidales tipo AVR446 para la
VELOCIDAD de los tracks y la POSICION de los flippers — identicas a las que
aplica mujoco_ros_bridge.py por substep. La idea es que el env de entrenamiento
directo (base_mujoco_env.py) comande los actuadores EXACTAMENTE igual que el
bridge de despliegue, para que la respuesta dinamica sea la misma y la politica
entrenada rapido sea desplegable sin gap sim-to-real.

No depende de ROS ni de rclpy (el bridge si, por eso este modulo va aparte y lo
importan los dos).
"""
from __future__ import annotations

import math
import mujoco

# ── Cinematica diferencial de las orugas (igual que el bridge) ───────────────
TRACK_SEPARATION = 0.36
WHEEL_RADIUS     = 0.15

# ── Limites AVR446 (rampa trapezoidal) ───────────────────────────────────────
VELOCITY_MAX_ACCEL = 15.0                       # rad/s^2 de los tracks (velocidad)
FLIPPER_MAX_VEL    = 3.0                         # rad/s   de los flippers (posicion)
FLIPPER_MAX_ACCEL  = 10.0                        # rad/s^2 de los flippers
FLIPPER_ACTS = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]


def ramp_toward_position(pos, vel, target, max_vel, max_accel, dt):
    """Un substep de rampa trapezoidal (espejo de _ramp_toward_position del
    bridge): acelera/frena respetando max_accel, sin superar max_vel, y frena a
    tiempo para llegar a 'target' con velocidad cero (sin overshoot)."""
    error = target - pos
    direction = 1.0 if error >= 0.0 else -1.0
    stoppable = math.sqrt(max(2.0 * max_accel * abs(error), 0.0))
    desired = direction * min(max_vel, stoppable)
    max_dv = max_accel * dt
    dv = max(-max_dv, min(max_dv, desired - vel))
    new_vel = vel + dv
    new_pos = pos + new_vel * dt
    if (target - new_pos) * error < 0.0:
        return target, 0.0
    return new_pos, new_vel


def ramp_toward_velocity(vel, target, max_accel, dt):
    """Igual pero solo limita la aceleracion de la velocidad comandada (tracks)."""
    max_dv = max_accel * dt
    return vel + max(-max_dv, min(max_dv, target - vel))


class RampController:
    """Estado + aplicacion de las rampas AVR446 sobre (model, data) de MuJoCo.

    Uso por paso de control:
        ctrl.set_base_twist(v, w)      # fija objetivo de velocidad de tracks
        ctrl.set_flippers(rad4)        # fija objetivo de posicion de flippers
        for _ in range(decimation):
            ctrl.apply(dt)             # avanza rampas un substep y escribe ctrl
            mujoco.mj_step(model, data)
    Tras un reset (mj_resetData / teleport): ctrl.sync_from_data().
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self._a_l = aid("vel_track_left")
        self._a_r = aid("vel_track_right")
        if self._a_l < 0 or self._a_r < 0:
            raise ValueError("Actuadores vel_track_left/right no encontrados")
        self._a_flip = [aid(n) for n in FLIPPER_ACTS]
        self._flip_qadr = [int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])])
                           for a in self._a_flip]

        # Estado de rampa: velocidad (tracks) y posicion (flippers).
        self._vt = {self._a_l: 0.0, self._a_r: 0.0}   # target
        self._vc = {self._a_l: 0.0, self._a_r: 0.0}   # current
        self._pt = [0.0, 0.0, 0.0, 0.0]               # target
        self._pp = [0.0, 0.0, 0.0, 0.0]               # current pos
        self._pv = [0.0, 0.0, 0.0, 0.0]               # current vel

    def set_base_twist(self, v: float, w: float):
        self._vt[self._a_l] = (v - w * TRACK_SEPARATION / 2.0) / WHEEL_RADIUS
        self._vt[self._a_r] = (v + w * TRACK_SEPARATION / 2.0) / WHEEL_RADIUS

    def set_flippers(self, rad4):
        for k in range(4):
            self._pt[k] = float(rad4[k])

    def sync_from_data(self):
        """Re-sincroniza el estado de rampas con data.qpos actual (tras reset):
        flippers al angulo real, tracks a velocidad cero."""
        for k, qadr in enumerate(self._flip_qadr):
            p = float(self.data.qpos[qadr])
            self._pt[k] = p; self._pp[k] = p; self._pv[k] = 0.0
        for a in (self._a_l, self._a_r):
            self._vt[a] = 0.0; self._vc[a] = 0.0

    def apply(self, dt: float):
        """Avanza las rampas un substep y escribe data.ctrl (no hace mj_step)."""
        for a in (self._a_l, self._a_r):
            self._vc[a] = ramp_toward_velocity(self._vc[a], self._vt[a], VELOCITY_MAX_ACCEL, dt)
            self.data.ctrl[a] = self._vc[a]
        for k, a in enumerate(self._a_flip):
            self._pp[k], self._pv[k] = ramp_toward_position(
                self._pp[k], self._pv[k], self._pt[k],
                FLIPPER_MAX_VEL, FLIPPER_MAX_ACCEL, dt)
            self.data.ctrl[a] = self._pp[k]
