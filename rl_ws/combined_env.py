"""
combined_env.py  —  Ambiente para entrenamiento conjunto de Modelo A + B.

El agente puede correr en tres modos, controlado por `mode`:

  "joint"     — Una sola red, 14 acciones, reward compartido.
                Útil para finetunear después de entrenar por separado.

  "separate"  — Dos redes independientes (política B y política A) que
                actúan sobre el mismo env. Cada una recibe su propia obs
                y su propio reward. Se actualizan con sus propios optimizadores.

  "hierarchical" — Política B congelada, solo se entrena A (o viceversa).
                   Útil para finetunear un modelo sin romper el otro.

Acciones (modo joint, 14 valores en [-1,1]):
  [0]   v_lin       base velocidad lineal
  [1]   ω_ang       base velocidad angular
  [2..5] flipper_1..4  posición flippers
  [6..11] joint_1..6  velocidad articular brazo
  [12]  dedo izq
  [13]  dedo der

Observación:
  base_obs (24) + arm_obs (31) = 55 valores concatenados

Uso:
    from combined_env import CombinedMuJoCoEnv
    env = CombinedMuJoCoEnv("../model_robot/aesir_mujoco.xml", mode="joint")
"""
from __future__ import annotations

import numpy as np
import mujoco
from typing import Optional, Dict, Tuple

# Importar capas de los envs individuales
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from base_env import (
    BaseMuJoCoEnv, TRACK_HALF_WIDTH, WHEEL_RADIUS,
    MAX_WHEEL_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
    DRIVE_LEFT, DRIVE_RIGHT, FLIPPERS, FLIP_WHEELS,
    NUM_LIDAR, LIDAR_MAX, LIDAR_SPIN, LIDAR_SPIN_VEL,
    CONTROL_DECIMATION
)
from arm_env import (
    ArmMuJoCoEnv, ARM_JOINTS, FINGER_L, FINGER_R,
    MAX_JOINT_VEL, JOINT_RANGE, REST_ANGLES
)

EPISODE_MAX_STEPS = 1000

# Índices en el vector de acción conjunta (14 dims)
IDX_VLIN    = 0
IDX_WANG    = 1
IDX_FLIP    = slice(2, 6)    # [2,3,4,5]
IDX_JOINTS  = slice(6, 12)   # [6..11]
IDX_FING_L  = 12
IDX_FING_R  = 13

# Reward weights
W_BASE_PROGRESS = 8.0
W_ARM_GOAL      = 2.0
W_ALIVE         = 0.01


class CombinedMuJoCoEnv:
    """Env MuJoCo que controla base + brazo en el mismo paso de simulación."""

    def __init__(self, xml_path: str,
                 mode: str = "joint",
                 render: bool = False,
                 max_steps: int = EPISODE_MAX_STEPS,
                 freeze_base: bool = False,
                 freeze_arm: bool = False):
        """
        mode: "joint" | "separate" | "hierarchical"
        freeze_base: en modo joint, la base actúa con acción 0 (para entrenar solo brazo)
        freeze_arm:  en modo joint, el brazo actúa con acción 0 (para entrenar solo base)
        """
        assert mode in ("joint", "separate", "hierarchical")
        self.mode        = mode
        self.freeze_base = freeze_base
        self.freeze_arm  = freeze_arm

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.max_steps   = max_steps
        self._step_count = 0

        # ── Reutilizar la lógica de índices de ambos envs ──────────────────
        # (sin crear otro MjModel, solo tomamos los IDs)
        self._init_actuator_ids()

        # Dimensiones de obs y acción
        self.obs_len_base = 24
        self.obs_len_arm  = 31
        self.obs_len      = self.obs_len_base + self.obs_len_arm  # 55
        self.act_len      = 14   # acciones conjuntas

        # Estado interno del integrador del brazo
        self._joint_pos = np.zeros(6)
        self._dt = self.model.opt.timestep * CONTROL_DECIMATION

        # Estado para rewards
        self._last_x      = 0.0
        self._stuck_cnt   = 0
        self.goal_pos: Optional[np.ndarray] = None

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuador no encontrado: {name}")
        return i

    def _init_actuator_ids(self):
        """Carga todos los IDs de actuadores en una sola pasada."""
        self.ids_drive_l  = [self._aid(n) for n in DRIVE_LEFT]
        self.ids_drive_r  = [self._aid(n) for n in DRIVE_RIGHT]
        self.ids_flippers = [self._aid(n) for n in FLIPPERS]
        # ruedas de flippers: flipper_name -> [wheel_ids]
        self.ids_flip_wh  = {}
        for fname, wnames in FLIP_WHEELS.items():
            self.ids_flip_wh[self._aid(fname)] = [self._aid(w) for w in wnames]
        self.id_lidar_spin = self._aid(LIDAR_SPIN)

        self.ids_arm   = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l = self._aid(FINGER_L)
        self.id_fing_r = self._aid(FINGER_R)

        # qpos addresses del brazo
        self.arm_jnt_ids  = [int(self.model.actuator_trnid[i, 0]) for i in self.ids_arm]
        self.arm_qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.arm_jnt_ids]
        self.arm_qvel_adr = [int(self.model.jnt_dofadr[j])  for j in self.arm_jnt_ids]

        # flippers
        self.flip_jnt_ids  = [int(self.model.actuator_trnid[i, 0]) for i in self.ids_flippers]
        self.flip_qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.flip_jnt_ids]
        self.flip_qvel_adr = [int(self.model.jnt_dofadr[j])  for j in self.flip_jnt_ids]

        # dedos
        fid_l = int(self.model.actuator_trnid[self.id_fing_l, 0])
        fid_r = int(self.model.actuator_trnid[self.id_fing_r, 0])
        self.fing_qpos_l = int(self.model.jnt_qposadr[fid_l])
        self.fing_qpos_r = int(self.model.jnt_qposadr[fid_r])

        # lidar
        self.lidar_adr = []
        for i in range(NUM_LIDAR):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid >= 0:
                self.lidar_adr.append(int(self.model.sensor_adr[sid]))

        # base y EE bodies
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0: self.base_id = 1
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "logitech_gripper_assembly"
        )
        if self.ee_id < 0:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")

    # ── Capa de traducción differential drive ─────────────────────────────
    def _diff_drive(self, v_lin: float, omega: float) -> Tuple[float, float]:
        vl = (v_lin - omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS
        vr = (v_lin + omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS
        return (float(np.clip(vl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)),
                float(np.clip(vr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)))

    # ── Aplicar acción conjunta ────────────────────────────────────────────
    def _apply_joint_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)

        # --- Base ---
        if not self.freeze_base:
            v_lin = float(a[IDX_VLIN]) * MAX_LINEAR_VEL
            omega = float(a[IDX_WANG]) * MAX_ANGULAR_VEL
            vl, vr = self._diff_drive(v_lin, omega)
            for i in self.ids_drive_l: self.data.ctrl[i] = vl
            for i in self.ids_drive_r: self.data.ctrl[i] = vr

            for k, fid in enumerate(self.ids_flippers):
                fp = float(a[IDX_FLIP][k]) * 3.1416
                self.data.ctrl[fid] = np.clip(fp, -3.1416, 3.1416)
                wvel = vl if k in (0, 2) else vr
                for wid in self.ids_flip_wh.get(fid, []):
                    self.data.ctrl[wid] = np.clip(wvel, -1.0, 1.0)

        # --- Brazo ---
        if not self.freeze_arm:
            delta = np.clip(a[IDX_JOINTS], -1.0, 1.0) * MAX_JOINT_VEL * self._dt
            self._joint_pos = np.clip(self._joint_pos + delta, -JOINT_RANGE, JOINT_RANGE)
            for k, aid in enumerate(self.ids_arm):
                self.data.ctrl[aid] = self._joint_pos[k]
            self.data.ctrl[self.id_fing_l] = (float(a[IDX_FING_L]) + 1.0) / 2.0 * 0.03
            self.data.ctrl[self.id_fing_r] = (float(a[IDX_FING_R]) + 1.0) / 2.0 * 0.03

        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

    # ── Observación ────────────────────────────────────────────────────────
    def _get_base_obs(self) -> np.ndarray:
        lin_vel  = float(self.data.qvel[0])
        ang_vel  = float(self.data.qvel[5])
        pos      = self.data.qpos[0:3].copy()
        quat     = self.data.qpos[3:7].copy()
        fp_qpos  = np.array([self.data.qpos[a] for a in self.flip_qpos_adr])
        fp_qvel  = np.array([self.data.qvel[a] for a in self.flip_qvel_adr])
        lidar    = np.array([
            min(float(self.data.sensordata[a]), LIDAR_MAX) / LIDAR_MAX
            for a in self.lidar_adr
        ]) if self.lidar_adr else np.zeros(NUM_LIDAR)
        return np.concatenate([[lin_vel], [ang_vel], pos, quat, fp_qpos, fp_qvel, lidar]).astype(np.float32)

    def _get_arm_obs(self) -> np.ndarray:
        qpos = np.array([self.data.qpos[a] for a in self.arm_qpos_adr])
        qvel = np.array([self.data.qvel[a] for a in self.arm_qvel_adr])
        ee_pos  = self.data.xpos[self.ee_id].copy() if self.ee_id >= 0 else np.zeros(3)
        ee_quat = self.data.xquat[self.ee_id].copy() if self.ee_id >= 0 else np.array([1.,0,0,0])
        fing = np.array([self.data.qpos[self.fing_qpos_l], self.data.qpos[self.fing_qpos_r]])
        lidar = np.array([
            min(float(self.data.sensordata[a]), LIDAR_MAX) / LIDAR_MAX
            for a in self.lidar_adr
        ]) if self.lidar_adr else np.zeros(NUM_LIDAR)
        base_pos = self.data.xpos[self.base_id].copy()
        return np.concatenate([qpos, qvel, ee_pos, ee_quat, fing, lidar, base_pos]).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._get_base_obs(), self._get_arm_obs()])

    def _get_obs_split(self) -> Dict[str, np.ndarray]:
        """Para modo separate: devuelve obs por agente."""
        return {"base": self._get_base_obs(), "arm": self._get_arm_obs()}

    # ── Reward ────────────────────────────────────────────────────────────
    def _reward_base(self) -> float:
        x = float(self.data.xpos[self.base_id, 0])
        dx = x - self._last_x
        self._last_x = x
        if abs(dx) < 0.003:
            self._stuck_cnt += 1
            pen = 0.05
        else:
            self._stuck_cnt = 0
            pen = 0.0
        lidar = (np.array([
            min(float(self.data.sensordata[a]), LIDAR_MAX) / LIDAR_MAX
            for a in self.lidar_adr
        ]) if self.lidar_adr else np.ones(NUM_LIDAR))
        obs_pen = max(0.0, 0.15 - float(lidar.min())) * 3.0
        return W_BASE_PROGRESS * dx + W_ALIVE - obs_pen - pen

    def _reward_arm(self) -> float:
        if self.goal_pos is not None and self.ee_id >= 0:
            dist = float(np.linalg.norm(self.data.xpos[self.ee_id] - self.goal_pos))
            return -W_ARM_GOAL * dist + W_ALIVE
        return W_ALIVE

    def _reward(self) -> float:
        return self._reward_base() + self._reward_arm()

    def _reward_split(self) -> Dict[str, float]:
        return {"base": self._reward_base(), "arm": self._reward_arm()}

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        if self._step_count >= self.max_steps: return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.2: return True
        if self._stuck_cnt > 60: return True
        return False

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = -1.5
        self.data.qpos[1] =  3.5
        self.data.qpos[2] =  0.2

        for nombre, angulo in REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo

        self._joint_pos = np.array([REST_ANGLES[f"joint_{i+1}"] for i in range(6)])
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._stuck_cnt  = 0
        self._last_x     = float(self.data.xpos[self.base_id, 0])

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

        if self.mode == "separate":
            return self._get_obs_split()
        return self._get_obs()

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, action):
        """
        modo joint:      action = np.ndarray (14,)
        modo separate:   action = {"base": np.ndarray(6,), "arm": np.ndarray(8,)}
        modo hierarchical: igual que joint pero con freeze_base/freeze_arm
        """
        if self.mode == "separate":
            a_base = np.clip(action["base"], -1.0, 1.0)
            a_arm  = np.clip(action["arm"],  -1.0, 1.0)
            a_full = np.zeros(self.act_len)
            a_full[IDX_VLIN]       = a_base[0]
            a_full[IDX_WANG]       = a_base[1]
            a_full[IDX_FLIP]       = a_base[2:6]
            a_full[IDX_JOINTS]     = a_arm[:6]
            a_full[IDX_FING_L]     = a_arm[6]
            a_full[IDX_FING_R]     = a_arm[7]
        else:
            a_full = action

        self._apply_joint_action(a_full)

        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

        self._step_count += 1
        done = self._terminated()

        if self.mode == "separate":
            obs = self._get_obs_split()
            rew = self._reward_split()
        else:
            obs = self._get_obs()
            rew = self._reward()

        return obs, rew, done, {}

    def set_goal(self, goal_xyz: np.ndarray):
        self.goal_pos = np.array(goal_xyz, dtype=np.float64)

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
