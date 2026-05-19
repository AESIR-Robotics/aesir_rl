"""
arm_env.py  —  Ambiente MuJoCo para entrenamiento del Modelo A (brazo 6-DOF + garra).

El agente emite 9 acciones en [-1, 1]:
  [0..5]  velocidades articulares normalizadas para joint_1..6
  [6]     garra izquierda (pos_left_finger)
  [7]     garra derecha   (pos_right_finger)

  (En el robot real estas velocidades se pasan a MoveIt Servo,
   que resuelve la cinemática y manda esfuerzos. Aquí en MuJoCo
   usamos los actuadores de posición con integración explícita.)

Observación (flat, 31 valores):
  joint_qpos(6) | joint_qvel(6) | ee_pos(3) | ee_quat(4) |
  finger_pos(2) | lidar_local(7) | base_qpos(3)
  = 31 valores

El brazo SE ENTRENA CON LA BASE QUIETA (o con un policy B congelado).
Para combinarlos después se usa una wrapper que corre ambos en paralelo.

Uso:
    from arm_env import ArmMuJoCoEnv
    env = ArmMuJoCoEnv("../model_robot/aesir_mujoco.xml")
"""
from __future__ import annotations

import numpy as np
import mujoco
from typing import Optional

# ── Actuadores del brazo ───────────────────────────────────────────────────
ARM_JOINTS   = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
                "pos_joint_4", "pos_joint_5", "pos_joint_6"]
FINGER_L     = "pos_left_finger"
FINGER_R     = "pos_right_finger"
LIDAR_SPIN   = "vel_lidar_spin"
LIDAR_SPIN_V = 20.0

MAX_JOINT_VEL = 1.0    # rad/s — qué tan rápido puede mover la articulación por step
JOINT_RANGE   = 3.1416

NUM_LIDAR = 7
LIDAR_MAX = 15.0

CONTROL_DECIMATION = 10
EPISODE_MAX_STEPS  = 500

# Ángulos de reposo del brazo (plegado sobre el robot)
REST_ANGLES = {
    "joint_1": -0.314,
    "joint_2": -3.14,
    "joint_3":  3.14,
    "joint_4":  0.0,
    "joint_5":  0.0,
    "joint_6":  0.0,
}


class ArmMuJoCoEnv:
    """Env MuJoCo para Modelo A — brazo 6-DOF con integración de velocidad.

    La acción es una velocidad articular normalizada. El env integra la
    velocidad para obtener la posición objetivo, que se pasa al actuador
    de posición de MuJoCo (equivalente a lo que haría MoveIt Servo).
    """

    def __init__(self, xml_path: str, render: bool = False,
                 max_steps: int = EPISODE_MAX_STEPS,
                 timestep_override: Optional[float] = None):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self._step_count = 0
        self.dt = timestep_override or (self.model.opt.timestep * CONTROL_DECIMATION)

        # ── índices de actuadores ──────────────────────────────────────────
        def _aid(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise ValueError(f"Actuador no encontrado: {name}")
            return i

        self.ids_arm    = [_aid(n) for n in ARM_JOINTS]
        self.id_fing_l  = _aid(FINGER_L)
        self.id_fing_r  = _aid(FINGER_R)
        self.id_lidar   = _aid(LIDAR_SPIN)

        # qpos addresses de las articulaciones del brazo
        self.arm_jnt_ids = [int(self.model.actuator_trnid[i, 0]) for i in self.ids_arm]
        self.arm_qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.arm_jnt_ids]
        self.arm_qvel_adr = [int(self.model.jnt_dofadr[j])  for j in self.arm_jnt_ids]

        # finger joints (slide type)
        fid_l = int(self.model.actuator_trnid[self.id_fing_l, 0])
        fid_r = int(self.model.actuator_trnid[self.id_fing_r, 0])
        self.fing_qpos_l = int(self.model.jnt_qposadr[fid_l])
        self.fing_qpos_r = int(self.model.jnt_qposadr[fid_r])

        # ── end-effector body ──────────────────────────────────────────────
        # Buscamos el body de la garra (link_6 o gripper_assembly)
        self.ee_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "logitech_gripper_assembly"
        )
        if self.ee_body < 0:
            self.ee_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link_6"
            )

        # ── base body ──────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── lidar ──────────────────────────────────────────────────────────
        self.lidar_adr = []
        for i in range(NUM_LIDAR):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid >= 0:
                self.lidar_adr.append(int(self.model.sensor_adr[sid]))

        # acción: 6 vel_joint + 2 dedos = 8 valores en [-1,1]
        self.act_len = 8
        # obs: joint_qpos(6) + joint_qvel(6) + ee_pos(3) + ee_quat(4) +
        #      finger_pos(2) + lidar(7) + base_qpos(3) = 31
        self.obs_len = 31

        # posición articular actual (integrada)
        self._joint_pos = np.zeros(6)

        # target gripper (goal) — puede setearse externamente
        self.goal_pos: Optional[np.ndarray] = None

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ── Integración de velocidad (MoveIt Servo equivalent) ─────────────────
    def _integrate_arm_velocity(self, vel_normalized: np.ndarray) -> np.ndarray:
        """Integra velocidades articulares para obtener la posición objetivo."""
        delta = np.clip(vel_normalized, -1.0, 1.0) * MAX_JOINT_VEL * self.dt
        self._joint_pos += delta
        # clamp a rango articular
        self._joint_pos = np.clip(self._joint_pos, -JOINT_RANGE, JOINT_RANGE)
        return self._joint_pos.copy()

    # ── Aplicar acción ─────────────────────────────────────────────────────
    def _apply_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)

        # brazo: integrar velocidad → setear posición objetivo
        target_pos = self._integrate_arm_velocity(a[:6])
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = target_pos[k]

        # garra: control de posición directo [0, 0.03]
        self.data.ctrl[self.id_fing_l] = (a[6] + 1.0) / 2.0 * 0.03
        self.data.ctrl[self.id_fing_r] = (a[7] + 1.0) / 2.0 * 0.03

        self.data.ctrl[self.id_lidar] = LIDAR_SPIN_V

    # ── Observación ────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        qpos_arm  = np.array([self.data.qpos[a] for a in self.arm_qpos_adr])
        qvel_arm  = np.array([self.data.qvel[a] for a in self.arm_qvel_adr])

        if self.ee_body >= 0:
            ee_pos  = self.data.xpos[self.ee_body].copy()
            ee_quat = self.data.xquat[self.ee_body].copy()
        else:
            ee_pos  = np.zeros(3)
            ee_quat = np.array([1.0, 0, 0, 0])

        fing = np.array([self.data.qpos[self.fing_qpos_l],
                         self.data.qpos[self.fing_qpos_r]])

        lidar = np.array([
            min(float(self.data.sensordata[a]), LIDAR_MAX) / LIDAR_MAX
            for a in self.lidar_adr
        ]) if self.lidar_adr else np.zeros(NUM_LIDAR)

        base_pos = self.data.xpos[self.base_id].copy()

        return np.concatenate([
            qpos_arm, qvel_arm, ee_pos, ee_quat, fing, lidar, base_pos
        ]).astype(np.float32)

    # ── Reward (ajustar según tarea) ───────────────────────────────────────
    def _reward(self) -> float:
        """Reward básico: minimizar distancia EE → goal si existe, else alive."""
        if self.goal_pos is not None and self.ee_body >= 0:
            ee = self.data.xpos[self.ee_body]
            dist = float(np.linalg.norm(ee - self.goal_pos))
            return -dist + 0.01   # bonus de supervivencia
        # sin goal: el trainer externo maneja la recompensa
        return 0.01

    # ── Terminación ────────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        return self._step_count >= self.max_steps

    # ── API pública ────────────────────────────────────────────────────────
    def reset(self, keep_base: bool = True) -> np.ndarray:
        """
        keep_base=True: solo resetea el brazo, deja la base donde está.
        keep_base=False: reset total del modelo.
        """
        if keep_base:
            # guardar qpos de la base (freejoint = primeros 7)
            base_qpos = self.data.qpos[:7].copy()
            base_qvel = self.data.qvel[:6].copy()
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = base_qpos
            self.data.qvel[:6] = base_qvel
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0] = -1.5
            self.data.qpos[1] =  3.5
            self.data.qpos[2] =  0.2

        # posición de reposo del brazo
        for nombre, angulo in REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo

        self._joint_pos = np.array([REST_ANGLES[f"joint_{i+1}"] for i in range(6)])

        self.data.ctrl[self.id_lidar] = LIDAR_SPIN_V

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

        return self._get_obs()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        self._step_count += 1
        obs  = self._get_obs()
        rew  = self._reward()
        done = self._terminated()
        return obs, rew, done, {}

    def set_goal(self, goal_xyz: np.ndarray):
        """Setea el objetivo del EE para el cálculo de reward."""
        self.goal_pos = np.array(goal_xyz, dtype=np.float64)

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
