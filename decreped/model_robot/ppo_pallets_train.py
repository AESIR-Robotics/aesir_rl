"""
PPO trainer — MAPA PALLETS
==========================
Archivo específico para entrenar al robot Aesir en la pista de pallets
(solo_pallets.xml + aesir_mujoco_robot_only.xml).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SISTEMA DE RECOMPENSAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  POSITIVAS
  ─────────
  +path_progress       Avance hacia el siguiente checkpoint del corredor
                       (medido sobre la curva S, no solo en X).  Escala: 8.0
  +checkpoint_bonus    Bono único al cruzar cada uno de los 8 checkpoints
                       del corredor. Monto: 40 por CP, deben cruzarse en orden.
  +completion_bonus    +500 al alcanzar el checkpoint final (pallet 18 zone).
  +alive_bonus         +0.01 por step para desincentivar suicidios rápidos.
  +smooth_drive_bonus  +0.003 si la velocidad lineal es consistente y hacia
                       adelante en el corredor (evita zigzag).

  PENALIZACIONES
  ──────────────
  -arm_fatal_penalty   -50 + terminal: brazo toca geom con prefijo "fatal_"
  -muerte_penalty      -10 por step + terminal: chasis toca "muerte_" (suelo)
  -stuck_penalty       -0.05/step si dx < umbral; terminal tras 60 steps quieto
  -lidar_penalty       Penaliza acercarse demasiado a obstáculos (lidar < 0.1)
  -action_cost         Penaliza consumo energético (norma al cuadrado del ctrl)
  -arm_energy_penalty  Penaliza velocidad angular del brazo: el brazo debe
                       mantenerse plegado mientras navega
  -wrong_direction_pen Penaliza alejarse del siguiente checkpoint activo
  -flip_overuse_pen    Penaliza uso excesivo de flippers en terreno plano

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEÑO DEL PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  El corredor serpentea en S entre las 18 tarimas.
  Se definen 8 checkpoints (CPs) en los cuellos de botella del corredor.
  CP0 = posición de spawn. El robot debe cruzarlos en orden 0→8.

  CP0  (-1.50,  3.50)   spawn
  CP1  (-1.10,  2.90)   salida gate pallets 9-10
  CP2  ( 0.00,  1.70)   gate pallets 7-6
  CP3  ( 1.20,  0.50)   gate pallets 5-4
  CP4  ( 2.50, -0.20)   gate pallets 3-14
  CP5  ( 4.00, -1.60)   corredor derecho pallets 15-16
  CP6  ( 4.70, -3.20)   gate pallets 17-1
  CP7  ( 3.50, -3.79)   META — zona pallet 18
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uso:
    cd /home/<user>/aesir_rl/workspace/src/aesir_robot_description
    MUJOCO_GL=egl python3 ppo_pallets_train.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import mujoco
import mujoco.viewer

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ──────────────────────── Configuración ────────────────────────────────────
XML_PATH = "/home/aesir/aesir_rl/models/aesir_pallets.xml"          # ajustar según tu estructura

CAMERA_NAMES       = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W = 84, 84
NUM_LIDAR_RAYS     = 7
LIDAR_MAX_RANGE    = 15.0
LIDAR_SPIN_VEL     = 20.0

ACTUATOR_NAMES = [
    "vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3",
    "vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3",
    "pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4",
    "pos_joint_1", "pos_joint_2", "pos_joint_3",
    "pos_joint_4", "pos_joint_5", "pos_joint_6",
    "pos_left_finger", "pos_right_finger",
    "vel_flip1_back", "vel_flip1_front",
    "vel_flip2_back", "vel_flip2_front",
    "vel_flip3_back", "vel_flip3_front",
    "vel_flip4_back", "vel_flip4_front",
]

# Índices dentro de ACTUATOR_NAMES para subconjuntos de actuadores
_IDX_DRIVE_L  = slice(0, 3)    # vel_drive_l_1..3
_IDX_DRIVE_R  = slice(3, 6)    # vel_drive_r_1..3
_IDX_FLIPPER  = slice(6, 10)   # pos_flipper_1..4
_IDX_ARM      = slice(10, 16)  # pos_joint_1..6
_IDX_GRIPPER  = slice(16, 18)

CONTROL_DECIMATION = 10
EPISODE_MAX_STEPS  = 1200       # más pasos para un corredor largo
STUCK_MAX_STEPS    = 60         # pasos quieto antes de terminar

CHECKPOINT_DIR = Path("./checkpoints_pallets")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ────────────── Path de navegación (checkpoints del corredor) ───────────────
# Coordenadas (x, y) en el frame mundo.
# El robot debe cruzarlos en orden estricto 0 → 7.
PATH_CHECKPOINTS: List[Tuple[float, float]] = [
    (-1.50,  3.50),   # CP0 spawn
    (-1.10,  2.90),   # CP1 salida gate 9-10
    ( 0.00,  1.70),   # CP2 gate 7-6
    ( 1.20,  0.50),   # CP3 gate 5-4
    ( 2.50, -0.20),   # CP4 gate 3-14
    ( 4.00, -1.60),   # CP5 corredor derecho
    ( 4.70, -3.20),   # CP6 gate 17-1
    ( 3.50, -3.79),   # CP7 META pallet 18
]
# Radio para considerar "cruzado" un checkpoint (metros)
CP_REACH_RADIUS = 0.90
# Checkpoint final (índice)
CP_FINAL_IDX = len(PATH_CHECKPOINTS) - 1

# Posición de spawn del robot
SPAWN_XYZ = (-1.5, 3.5, 0.20)

# Ángulos de reposo del brazo (joints 1-6 en radianes)
ARM_REST_ANGLES = {
    "joint_1":  -0.314,
    "joint_2":  -3.14,
    "joint_3":   3.14,
    "joint_4":  -1.35,
    "joint_5":  -1.54,
    "joint_6":   1.54,
}

# ────────────── Magnitudes de recompensa (ajustables) ──────────────────────
R_PATH_PROGRESS      =  8.0    # por metro avanzado hacia siguiente CP
R_CHECKPOINT         = 40.0    # bono único por CP cruzado
R_COMPLETION         = 500.0   # bono final al completar el recorrido
R_ALIVE              =  0.01   # por step
R_SMOOTH_DRIVE       =  0.003  # si velocidad lineal alineada hacia el path

P_ARM_FATAL          = -50.0   # brazo toca "fatal_" → terminal
P_MUERTE             = -10.0   # toca "muerte_" por step → terminal
P_STUCK              = -0.05   # por step quieto
P_LIDAR_NEAR         = -5.0    # escala cuando lidar < LIDAR_DANGER_THRESH
P_ACTION_COST        = -1e-3   # escala por norma² del ctrl
P_ARM_ENERGY         = -0.005  # escala por |qvel| del brazo
P_WRONG_DIR          = -2.0    # por metro alejado del siguiente CP
P_FLIP_OVERUSE       = -0.002  # escala por |flipper_cmd| en terreno plano
P_Z_BOUNCE           = -10.0   # NUEVO: Penalización por saltar o moverse bruscamente en Z

LIDAR_DANGER_THRESH  = 0.12    # porcentaje del rango máximo (normalizado)


# ══════════════════════════════════════════════════════════════════════════════
#  PathMonitor — lógica de checkpoints
# ══════════════════════════════════════════════════════════════════════════════
class PathMonitor:
    """
    Mantiene el estado del recorrido (qué checkpoints se han cruzado)
    y calcula la recompensa de progreso a lo largo de la curva S.

    Expone:
        .update(xy)  → (path_reward, cp_bonus, completed)
        .reset()
        .next_cp_xy  → (x,y) del próximo checkpoint
        .current_cp  → índice del CP activo (el que sigue por cruzar)
    """

    def __init__(self):
        self._cps = np.array(PATH_CHECKPOINTS, dtype=np.float64)
        self.reset()

    def reset(self, start_cp_idx: int = 0, start_xy: np.ndarray = None):
        if start_xy is None:
            start_xy = np.array(SPAWN_XYZ[:2])
            
        # El checkpoint activo será el siguiente al punto de aparición
        self.current_cp   = min(start_cp_idx + 1, CP_FINAL_IDX)
        self._prev_dist   = self._dist_to_cp(self.current_cp, start_xy)
        self.cps_crossed  = start_cp_idx  # Asumimos que "cruzó" los anteriores al aparecer ahí
        self.completed    = False

    @property
    def next_cp_xy(self) -> np.ndarray:
        idx = min(self.current_cp, CP_FINAL_IDX)
        return self._cps[idx]

    def _dist_to_cp(self, cp_idx: int, xy: np.ndarray) -> float:
        return float(np.linalg.norm(xy - self._cps[cp_idx]))

    def update(self, xy: np.ndarray) -> Tuple[float, float, bool]:
        """
        Retorna (path_reward, cp_bonus, episode_completed).
        Debe llamarse UNA vez por step.
        """
        if self.completed:
            return 0.0, 0.0, True

        cp_idx = self.current_cp
        dist_now = self._dist_to_cp(cp_idx, xy)

        # ── recompensa de progreso: cuánto nos acercamos al CP activo ──────
        delta_dist     = self._prev_dist - dist_now    # positivo = acercarse
        path_reward    = R_PATH_PROGRESS * delta_dist
        self._prev_dist = dist_now

        # ── penalización por alejarse del CP activo ─────────────────────────
        wrong_dir_pen = 0.0
        if delta_dist < -0.01:                          # nos alejamos
            wrong_dir_pen = P_WRONG_DIR * abs(delta_dist)

        # ── detección de checkpoint cruzado ─────────────────────────────────
        cp_bonus = 0.0
        if dist_now < CP_REACH_RADIUS:
            cp_bonus        = R_CHECKPOINT
            self.cps_crossed += 1

            if cp_idx == CP_FINAL_IDX:
                self.completed = True
                cp_bonus += R_COMPLETION
            else:
                self.current_cp  = cp_idx + 1
                self._prev_dist  = self._dist_to_cp(self.current_cp, xy)

        return (path_reward + wrong_dir_pen), cp_bonus, self.completed


# ══════════════════════════════════════════════════════════════════════════════
#  ContactMonitor — lógica de colisiones
# ══════════════════════════════════════════════════════════════════════════════
class ContactMonitor:
    """
    Clasifica cada contacto en la simulación usando los prefijos de nombre:
      "fatal_"  → sólo mata si lo toca el brazo
      "muerte_" → mata sin importar qué parte del robot
    Expone flags que se resetean en cada step.
    """

    _CHASSIS_PARTS = frozenset({
        "base_link", "tracked_1", "tracked_2",
        "flipper_1_1", "flipper_2_1", "flipper_3_1", "flipper_4_1",
        "footprint_link",
    })
    _ARM_PARTS = frozenset({
        "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
        "logitech_gripper_assembly", "left_finger_link", "right_finger_link",
    })

    def __init__(self, model: mujoco.MjModel):
        self._model = model
        self.arm_hit_fatal  = False
        self.robot_hit_muerte = False

    def reset_flags(self):
        self.arm_hit_fatal    = False
        self.robot_hit_muerte = False

    def _body_name(self, body_id: int) -> str:
        n = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return n or ""

    def _geom_name(self, geom_id: int) -> str:
        n = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return n or ""

    def scan(self, data: mujoco.MjData):
        self.reset_flags()
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = self._geom_name(c.geom1)
            g2 = self._geom_name(c.geom2)
            b1 = self._body_name(self._model.geom_bodyid[c.geom1])
            b2 = self._body_name(self._model.geom_bodyid[c.geom2])

            is_chassis1 = g1 in self._CHASSIS_PARTS or b1 in self._CHASSIS_PARTS
            is_chassis2 = g2 in self._CHASSIS_PARTS or b2 in self._CHASSIS_PARTS
            is_arm1     = g1 in self._ARM_PARTS     or b1 in self._ARM_PARTS
            is_arm2     = g2 in self._ARM_PARTS     or b2 in self._ARM_PARTS
            is_robot    = is_chassis1 or is_chassis2 or is_arm1 or is_arm2

            has_fatal  = "fatal_"  in g1 or "fatal_"  in g2
            has_muerte = "muerte_" in g1 or "muerte_" in g2

            # muerte_ + cualquier pieza del robot → kill
            if has_muerte and is_robot:
                self.robot_hit_muerte = True

            # fatal_ + brazo → kill brazo
            if has_fatal:
                if ("fatal_" in g1 and is_arm2) or ("fatal_" in g2 and is_arm1):
                    self.arm_hit_fatal = True


# ══════════════════════════════════════════════════════════════════════════════
#  AesirPalletsEnv
# ══════════════════════════════════════════════════════════════════════════════
class AesirPalletsEnv:
    """
    Entorno MuJoCo para la pista de pallets.
    Reemplaza AesirMuJoCoEnv del trainer genérico.
    """

    def __init__(self,
                 xml_path: str = XML_PATH,
                 camera_names: List[str] = CAMERA_NAMES,
                 image_hw: Tuple[int, int] = (CAMERA_H, CAMERA_W),
                 num_lidar_rays: int = NUM_LIDAR_RAYS,
                 control_decimation: int = CONTROL_DECIMATION,
                 max_steps: int = EPISODE_MAX_STEPS,
                 lidar_max_range: float = LIDAR_MAX_RANGE,
                 render: bool = False):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.image_h, self.image_w = image_hw
        self.renderer = mujoco.Renderer(self.model,
                                        height=self.image_h,
                                        width=self.image_w)
        self.camera_names     = list(camera_names)
        self.num_cameras      = len(camera_names)
        self.num_lidar        = num_lidar_rays
        self.lidar_max        = lidar_max_range
        self.control_decimation = control_decimation
        self.max_steps        = max_steps

        # ── actuadores ──────────────────────────────────────────────────────
        self.act_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ACTUATOR_NAMES
        ], dtype=np.int32)
        missing = [n for n, i in zip(ACTUATOR_NAMES, self.act_ids) if i < 0]
        if missing:
            raise ValueError(f"Actuadores no encontrados en el modelo: {missing}")

        self.ctrlrange = self.model.actuator_ctrlrange[self.act_ids].copy()
        self.act_low   = self.ctrlrange[:, 0]
        self.act_high  = self.ctrlrange[:, 1]
        self.act_len   = len(self.act_ids)

        # ── joint addresses para obs ────────────────────────────────────────
        self.joint_ids = np.array(
            [self.model.actuator_trnid[i, 0] for i in self.act_ids], dtype=np.int32
        )
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[j]  for j in self.joint_ids], dtype=np.int32
        )
        self.joint_len = 2 * self.act_len

        # ── joint address del brazo (para penalizar energía) ────────────────
        arm_joint_names = [
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"
        ]
        self._arm_dof_adrs = []
        for jn in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self._arm_dof_adrs.append(int(self.model.jnt_dofadr[jid]))

        # ── flipper joint ids (para penalizar overuse) ───────────────────────
        _flip_names = ["flipper_joint_1","flipper_joint_2",
                        "flipper_joint_3","flipper_joint_4"]
        self._flip_act_indices = [
            ACTUATOR_NAMES.index(f"pos_flipper_{i+1}") for i in range(4)
        ]

        # ── lidar spin ───────────────────────────────────────────────────────
        self.lidar_spin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_lidar_spin"
        )

        # ── sensores lidar ───────────────────────────────────────────────────
        self.lidar_sensor_adr = []
        for i in range(self.num_lidar):
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}"
            )
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} no encontrado")
            self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        # ── body base ────────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "footprint_link"
        )
        if self.base_id < 0:
            self.base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
        if self.base_id < 0:
            self.base_id = 1

        # ── shapes ───────────────────────────────────────────────────────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (self.num_lidar,)
        self.joint_shape = (self.joint_len,)

        # ── monitores ────────────────────────────────────────────────────────
        self._path_monitor    = PathMonitor()
        self._contact_monitor = ContactMonitor(self.model)

        # ── estado interno ───────────────────────────────────────────────────
        self._step_counter  = 0
        self._stuck_counter = 0
        self._last_xy       = np.array(SPAWN_XYZ[:2], dtype=np.float64)
        self._last_z        = float(SPAWN_XYZ[2])
        self._last_vel_dir  = np.zeros(2)   # para smooth_drive_bonus

        # ── viewer ───────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 6.0
            self.viewer.cam.elevation = -25

    # ── helpers ─────────────────────────────────────────────────────────────

    def _read_cameras(self) -> np.ndarray:
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            frames.append(img.astype(np.float32) / 255.0)
        stacked = np.concatenate(frames, axis=-1)
        return np.transpose(stacked, (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        lidar = np.empty(self.num_lidar, dtype=np.float32)
        for i, adr in enumerate(self.lidar_sensor_adr):
            d = float(self.data.sensordata[adr])
            lidar[i] = d / self.lidar_max if 0 < d < self.lidar_max else 1.0
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        qpos = self.data.qpos[self.qpos_adr]
        qvel = self.data.qvel[self.qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0)
        return self.act_low + 0.5 * (a + 1.0) * (self.act_high - self.act_low)

    def _set_arm_rest(self):
        """Coloca el brazo en posición de reposo al hacer reset."""
        for jname, angle in ARM_REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angle

    # ── API pública ──────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

        # ── NUEVO: Spawn Aleatorio sobre la ruta de Pallets ──
        start_idx = int(np.random.randint(0, len(PATH_CHECKPOINTS) - 1))
        spawn_x, spawn_y = PATH_CHECKPOINTS[start_idx]
        spawn_z = 0.25  # Altura segura sobre la madera

        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = spawn_z
        self.data.qpos[3:7] = [1, 0, 0, 0]   # cuaternión identidad

        self._set_arm_rest()

        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

        # Settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

        self._step_counter  = 0
        self._stuck_counter = 0

        # Registrar posiciones iniciales para el tracking de recompensas
        base_pos = self.data.xpos[self.base_id]
        self._last_xy       = base_pos[:2].copy()
        self._last_z        = float(base_pos[2])  # NUEVO: Seguimiento del eje Z
        self._last_vel_dir  = np.zeros(2)

        # Sincronizar el monitor con el punto de aparición aleatorio
        self._path_monitor.reset(start_cp_idx=start_idx, start_xy=self._last_xy)
        self._contact_monitor.reset_flags()

        return self._observation()

    def step(self, action: np.ndarray):
        scaled = self._scale_action(action)
        self.data.ctrl[self.act_ids] = scaled
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

        self._step_counter += 1
        self._contact_monitor.scan(self.data)

        obs    = self._observation()
        reward = self._compute_reward(obs, scaled)
        done   = self._terminated()

        return obs, reward, done, {
            "checkpoint":   self._path_monitor.current_cp,
            "cps_crossed":  self._path_monitor.cps_crossed,
            "completed":    self._path_monitor.completed,
            "arm_fatal":    self._contact_monitor.arm_hit_fatal,
            "muerte":       self._contact_monitor.robot_hit_muerte,
            "stuck_steps":  self._stuck_counter,
        }

    # ── reward ──────────────────────────────────────────────────────────────

    def _compute_reward(self,
                        obs: Dict[str, np.ndarray],
                        scaled_ctrl: np.ndarray) -> float:
        reward = 0.0
        base_pos = self.data.xpos[self.base_id]
        xy = base_pos[:2].copy()
        current_z = float(base_pos[2])

        # ── 1. PROGRESO EN EL PATH ─────────────────────────────────────────
        path_r, cp_bonus, _ = self._path_monitor.update(xy)
        reward += path_r + cp_bonus

        # ── NUEVO: PENALIZACIÓN EN EJE Z (Saltos/Botes) ────────────────────
        dz = current_z - getattr(self, '_last_z', current_z)
        self._last_z = current_z
        
        z_penalty = P_Z_BOUNCE * abs(dz)
        reward += z_penalty

        # ── 2. ALIVE BONUS ────────────────────────────────────────────────
        reward += R_ALIVE

        # ── 3. SMOOTH DRIVE BONUS ─────────────────────────────────────────
        # Recompensa si la velocidad va alineada con el vector hacia el CP
        vel = xy - self._last_xy
        next_cp = self._path_monitor.next_cp_xy
        to_cp   = next_cp - xy
        to_cp_norm = np.linalg.norm(to_cp)
        if to_cp_norm > 1e-4 and np.linalg.norm(vel) > 1e-4:
            alignment = float(np.dot(vel, to_cp) / (np.linalg.norm(vel) * to_cp_norm))
            if alignment > 0.5:          # va en la dirección correcta
                reward += R_SMOOTH_DRIVE * alignment
        self._last_xy = xy.copy()

        # ── 4. STUCK PENALTY ─────────────────────────────────────────────
        dist_moved = float(np.linalg.norm(xy - (xy - (xy - self._last_xy))))
        if dist_moved < 0.005:
            self._stuck_counter += 1
            reward += P_STUCK
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)

        # ── 5. LIDAR OBSTACLE PENALTY ─────────────────────────────────────
        min_lidar = float(obs["lidar"].min())
        if min_lidar < LIDAR_DANGER_THRESH:
            # penalty proporcional: mayor mientras más cerca
            reward += P_LIDAR_NEAR * (LIDAR_DANGER_THRESH - min_lidar)

        # ── 6. ACTION COST (energía total) ────────────────────────────────
        reward += P_ACTION_COST * float(np.square(scaled_ctrl).mean())

        # ── 7. ARM ENERGY PENALTY (brazo debe quedarse quieto) ────────────
        if self._arm_dof_adrs:
            arm_vel = np.array(
                [self.data.qvel[adr] for adr in self._arm_dof_adrs]
            )
            reward += P_ARM_ENERGY * float(np.sum(np.abs(arm_vel)))

        # ── 8. FLIPPER OVERUSE PENALTY ────────────────────────────────────
        # En terreno plano, los flippers deben estar plegados (ctrl ≈ 0)
        flip_cmds = scaled_ctrl[_IDX_FLIPPER]
        reward += P_FLIP_OVERUSE * float(np.sum(np.abs(flip_cmds)))

        # ── 9. CONTACTO FATAL / MUERTE ───────────────────────────────────
        if self._contact_monitor.arm_hit_fatal:
            reward += P_ARM_FATAL         # ya generará done=True

        if self._contact_monitor.robot_hit_muerte:
            reward += P_MUERTE            # ya generará done=True

        return float(reward)

    # ── terminated ──────────────────────────────────────────────────────────

    def _terminated(self) -> bool:
        # Máximo de steps
        if self._step_counter >= self.max_steps:
            return True

        # Volcado del chasis (z_col del eje Z del cuerpo < umbral)
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.20:
            return True

        # Quieto demasiado tiempo
        if self._stuck_counter >= STUCK_MAX_STEPS:
            return True

        # Contacto letal
        if self._contact_monitor.robot_hit_muerte:
            return True
        if self._contact_monitor.arm_hit_fatal:
            return True

        # Misión completada
        if self._path_monitor.completed:
            return True

        return False

    def close(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  Redes neurales (igual que el trainer genérico, sin cambios)
# ══════════════════════════════════════════════════════════════════════════════
class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32,          64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,          64, 3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, h, w)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU(inplace=True))
        self.out_dim = out_dim

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, out_dim), nn.Tanh(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class ConvActorCritic(nn.Module):
    def __init__(self, image_shape, lidar_dim, joint_dim, act_dim,
                 img_feat=256, vec_feat=128, hidden=256, log_std_init=-0.5):
        super().__init__()
        c, h, w = image_shape
        self.img_enc = ImageEncoder(c, h, w, out_dim=img_feat)
        self.vec_enc = StateEncoder(lidar_dim + joint_dim, out_dim=vec_feat)
        fused = img_feat + vec_feat
        self.trunk    = nn.Sequential(
            nn.Linear(fused,  hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_dim  = act_dim

    def _fuse(self, images, lidar, joints):
        return self.trunk(torch.cat(
            [self.img_enc(images), self.vec_enc(torch.cat([lidar, joints], -1))], -1
        ))

    def forward(self, images, lidar, joints):
        z       = self._fuse(images, lidar, joints)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, obs, device):
        im = obs["images"].unsqueeze(0).to(device)
        li = obs["lidar"].unsqueeze(0).to(device)
        jo = obs["joint_states"].unsqueeze(0).to(device)
        mu, std, value = self(im, li, jo)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        return raw.squeeze(0).cpu().numpy(), float(logp), float(value)

    def evaluate(self, images, lidar, joints, actions):
        mu, std, value = self(images, lidar, joints)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, value, entropy


# ══════════════════════════════════════════════════════════════════════════════
#  Rollout buffer
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RolloutBuffer:
    capacity:    int
    image_shape: Tuple[int, int, int]
    lidar_dim:   int
    joint_dim:   int
    act_dim:     int

    images:  np.ndarray = field(init=False)
    lidars:  np.ndarray = field(init=False)
    joints:  np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    logps:   np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values:  np.ndarray = field(init=False)
    dones:   np.ndarray = field(init=False)
    idx:     int        = 0

    def __post_init__(self):
        c, h, w = self.image_shape
        self.images  = np.zeros((self.capacity, c, h, w),        np.float32)
        self.lidars  = np.zeros((self.capacity, self.lidar_dim),  np.float32)
        self.joints  = np.zeros((self.capacity, self.joint_dim),  np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim),    np.float32)
        self.logps   = np.zeros((self.capacity,),                 np.float32)
        self.rewards = np.zeros((self.capacity,),                 np.float32)
        self.values  = np.zeros((self.capacity,),                 np.float32)
        self.dones   = np.zeros((self.capacity,),                 np.float32)

    def store(self, obs, action, logp, reward, value, done) -> bool:
        i = self.idx
        self.images[i]  = obs["images"]
        self.lidars[i]  = obs["lidar"]
        self.joints[i]  = obs["joint_states"]
        self.actions[i] = action
        self.logps[i]   = logp
        self.rewards[i] = reward
        self.values[i]  = value
        self.dones[i]   = float(done)
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0
            return True
        return False

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv = last_value if t == self.capacity - 1 else self.values[t + 1]
            nt = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nv * nt - self.values[t]
            gae   = delta + gamma * lam * nt * gae
            adv[t] = gae
        ret  = adv + self.values
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret


# ══════════════════════════════════════════════════════════════════════════════
#  PPO update
# ══════════════════════════════════════════════════════════════════════════════
def ppo_update(policy, optimizer, buf, advantages, returns,
               epochs, batch_size, clip, vf_coef, ent_coef, device):
    images  = torch.as_tensor(buf.images,  dtype=torch.float32, device=device)
    lidar   = torch.as_tensor(buf.lidars,  dtype=torch.float32, device=device)
    joints  = torch.as_tensor(buf.joints,  dtype=torch.float32, device=device)
    actions = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_lp  = torch.as_tensor(buf.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv     = torch.as_tensor(advantages,  dtype=torch.float32, device=device).unsqueeze(-1)
    ret     = torch.as_tensor(returns,     dtype=torch.float32, device=device).unsqueeze(-1)

    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)),
                                batch_size, drop_last=False):
            lp, val, ent = policy.evaluate(images[idx], lidar[idx],
                                            joints[idx], actions[idx])
            ratio  = torch.exp(lp - old_lp[idx])
            s1     = ratio * adv[idx]
            s2     = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]
            pi_l   = -torch.min(s1, s2).mean()
            v_l    = F.smooth_l1_loss(val, ret[idx])
            loss   = pi_l + vf_coef * v_l - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            stats["pi"]  = pi_l.item()
            stats["v"]   = v_l.item()
            stats["ent"] = ent.item()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  Loop de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════
def obs_to_tensor(obs):
    return {k: torch.from_numpy(v).float() for k, v in obs.items()}


def save_checkpoint(path, policy, optimizer, it, avg_r):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "avg_ep_r": avg_r}, path)


def make_camera_panel(obs_images, camera_names):
    n = len(camera_names)
    h, w = obs_images.shape[1], obs_images.shape[2]
    panel = np.zeros((h, w * n, 3), dtype=np.uint8)
    for k in range(n):
        cam = (obs_images[k*3:(k+1)*3].transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
        panel[:, k*w:(k+1)*w] = cam
    return wandb.Image(panel, caption=" | ".join(camera_names))


def train(num_iterations: int  = 600,
          steps_per_iter:  int  = 2048,
          ppo_epochs:      int  = 10,
          batch_size:      int  = 256,
          gamma:           float = 0.99,
          gae_lambda:      float = 0.95,
          clip_param:      float = 0.20,
          vf_coef:         float = 0.50,
          ent_coef:        float = 0.005,
          lr:              float = 3e-4,
          save_every:      int   = 10,
          device_str:      str   = "auto",
          render:          bool  = False,
          use_wandb:       bool  = True,
          wandb_project:   str   = "AESIR-PPO-PALLETS",
          wandb_run_name:  str   = None,
          image_log_every: int   = 25):

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))
    print(f"Device: {device}")

    env = AesirPalletsEnv(render=render)
    print(f"act_len={env.act_len}  image={env.image_shape}  "
          f"lidar={env.num_lidar}  joints={env.joint_len}")

    policy = ConvActorCritic(
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    buf = RolloutBuffer(
        capacity=steps_per_iter,
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    )

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name, config={
            "num_iterations": num_iterations, "steps_per_iter": steps_per_iter,
            "gamma": gamma, "clip": clip_param, "lr": lr,
            "checkpoints": PATH_CHECKPOINTS,
            "R_PATH_PROGRESS": R_PATH_PROGRESS, "R_CHECKPOINT": R_CHECKPOINT,
            "R_COMPLETION": R_COMPLETION,
        })
        wandb.watch(policy, log="gradients", log_freq=100)

    obs = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_history: List[float] = []
    ep_cps_history: List[int] = []    # checkpoints cruzados por episodio
    best_avg = -1e9

    try:
        for it in range(num_iterations):
            t0 = time.time()

            for _ in range(steps_per_iter):
                obs_t  = obs_to_tensor(obs)
                action, logp, value = policy.act(obs_t, device)
                next_obs, reward, done, info = env.step(action)
                buf.store(obs, action, logp, reward, value, done)
                obs       = next_obs
                ep_reward += reward
                ep_len    += 1

                if done:
                    ep_history.append(ep_reward)
                    ep_cps_history.append(info["cps_crossed"])
                    if len(ep_history) > 20:
                        ep_history.pop(0)
                        ep_cps_history.pop(0)
                    ep_reward, ep_len = 0.0, 0
                    obs = env.reset()

            # GAE bootstrap
            obs_t = obs_to_tensor(obs)
            with torch.no_grad():
                _, _, last_val = policy(
                    obs_t["images"].unsqueeze(0).to(device),
                    obs_t["lidar"].unsqueeze(0).to(device),
                    obs_t["joint_states"].unsqueeze(0).to(device),
                )
            adv, ret = buf.compute_gae(float(last_val), gamma, gae_lambda)
            stats = ppo_update(policy, optimizer, buf, adv, ret,
                               ppo_epochs, batch_size, clip_param,
                               vf_coef, ent_coef, device)

            avg_r   = float(np.mean(ep_history))   if ep_history else float("nan")
            avg_cps = float(np.mean(ep_cps_history)) if ep_cps_history else 0.0
            dt      = time.time() - t0

            print(f"[{it:4d}] avg_r={avg_r:8.2f}  avg_cps={avg_cps:.1f}  "
                  f"pi={stats['pi']:+.4f}  v={stats['v']:.4f}  "
                  f"ent={stats['ent']:.3f}  ({dt:.1f}s)")

            if use_wandb:
                log = {
                    "iter": it, "global_step": (it+1)*steps_per_iter,
                    "avg_ep_reward": avg_r, "avg_cps_crossed": avg_cps,
                    "policy_loss": stats["pi"], "value_loss": stats["v"],
                    "entropy": stats["ent"],
                    "mean_log_std": policy.log_std.detach().mean().item(),
                    "iter_time_s": dt,
                }
                if it % image_log_every == 0:
                    log["cameras"] = make_camera_panel(obs["images"], env.camera_names)
                wandb.log(log, step=(it+1)*steps_per_iter)

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"pallets_iter{it+1:05d}.pt"
                save_checkpoint(p, policy, optimizer, it+1, avg_r)
                print(f"  ↳ checkpoint: {p}")

            if ep_history and avg_r > best_avg:
                best_avg = avg_r
                p = CHECKPOINT_DIR / "pallets_best.pt"
                save_checkpoint(p, policy, optimizer, it+1, avg_r)
                print(f"  ↳ NUEVO BEST ({avg_r:.2f})")
                if use_wandb:
                    wandb.run.summary["best_avg_ep_r"] = best_avg

    finally:
        env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train(render=True)