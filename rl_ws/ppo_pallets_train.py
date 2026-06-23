"""
PPO trainer — MAPA PALLETS (V3 - Anti-Saltos y Estabilidad)
=============================================================
Combina el sistema de misiones espaciales (PathMonitor) con el 
acelerador de aprendizaje (Differential Drive + Arm Integrator)
reduciendo el espacio de acción de 26 a 14 dimensiones.

Acciones (14 dims, normalizadas en [-1,1]):
  [0]    v_lin       velocidad lineal de la base   → diferencial
  [1]    ω_ang       velocidad angular de la base  → diferencial
  [2..5] flipper_1..4  posición objetivo de cada flipper
  [6..11] joint_1..6  velocidad articular del brazo (integrada a posición)
  [12]   dedo izquierdo
  [13]   dedo derecho
"""
from __future__ import annotations

import os
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


# ──────────────────────────── Ruta XML (Auto-detecta) ──────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_FULL    = os.path.join(_HERE, "../aesir_robot_description/launch/aesir_complete.xml")
_ROBOT   = os.path.join(_HERE, "aesir_mujoco.xml")
# Ajusta a tu preferencia, por defecto asume el path del entorno real:
XML_PATH = _FULL if os.path.exists(_FULL) else "/home/aesir/aesir_rl/models/aesir_pallets.xml"

# ──────────────────────────── Configuración General ────────────────────────
CAMERA_NAMES       = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W = 84, 84
NUM_LIDAR_RAYS     = 7
LIDAR_MAX_RANGE    = 15.0
LIDAR_SPIN_VEL     = 20.0

# ── Parámetros físicos del control ─────────────────────────────────────────
TRACK_HALF_WIDTH    = 0.21        # Distancia Y centro-oruga (m)
WHEEL_RADIUS        = 0.05        # Radio rueda tracción (m)
MAX_WHEEL_VEL       = 20.0        # rad/s (= ctrlrange vel_drive_*)
MAX_LINEAR_VEL      = 1.5         # m/s
MAX_ANGULAR_VEL     = 2.0         # rad/s
MAX_JOINT_VEL       = 1.0         # rad/s por articulación del brazo

# ── Nombres de actuadores agrupados ──────────────────────────────────────
DRIVE_LEFT   = ["vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3"]
DRIVE_RIGHT  = ["vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3"]
FLIPPERS     = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
FLIP_WHEELS  = {
    "pos_flipper_1": ["vel_flip1_back", "vel_flip1_front"],
    "pos_flipper_2": ["vel_flip2_back", "vel_flip2_front"],
    "pos_flipper_3": ["vel_flip3_back", "vel_flip3_front"],
    "pos_flipper_4": ["vel_flip4_back", "vel_flip4_front"],
}
ARM_JOINTS   = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
                "pos_joint_4", "pos_joint_5", "pos_joint_6"]
FINGER_L     = "pos_left_finger"
FINGER_R     = "pos_right_finger"
LIDAR_SPIN   = "vel_lidar_spin"

OBS_ACTUATORS = DRIVE_LEFT + DRIVE_RIGHT + FLIPPERS + ARM_JOINTS + [FINGER_L, FINGER_R]

CONTROL_DECIMATION = 10
EPISODE_MAX_STEPS  = 1200       
STUCK_MAX_STEPS    = 60         

CHECKPOINT_DIR = Path("./checkpoints_pallets")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ────────────── Path de navegación (checkpoints del corredor) ───────────────
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
CP_REACH_RADIUS = 0.90
CP_FINAL_IDX    = len(PATH_CHECKPOINTS) - 1
SPAWN_XYZ       = (-1.5, 3.5, 0.20)

ARM_REST_ANGLES = {
    "joint_1": -0.314, "joint_2": -3.14, "joint_3":  3.14,
    "joint_4": -1.35,  "joint_5": -1.54, "joint_6":  1.54,
}

# ────────────── Magnitudes de recompensa (ajustables) ──────────────────────
R_PATH_PROGRESS      =  8.0    
R_CHECKPOINT         = 40.0    
R_COMPLETION         = 500.0   
R_ALIVE              =  0.01   
R_SMOOTH_DRIVE       =  0.003  

P_ARM_FATAL          = -50.0   
P_MUERTE             = -10.0   
P_STUCK              = -0.05   
P_LIDAR_NEAR         = -5.0    
P_ACTION_COST        = -1e-3   
P_ARM_ENERGY         = -0.005  
P_WRONG_DIR          = -2.0    
P_FLIP_OVERUSE       = -0.002  

# NUEVAS PENALIZACIONES DE FÍSICAS
P_JUMP_FATAL         = -50.0   # Si Z supera 0.45m (salto/vuelo)
P_STABILITY          = -10.0   # Castigo por inclinación severa (wheelies)
P_SHAKE              = -0.02   # Castigo por velocidades angulares bruscas del chasis

LIDAR_DANGER_THRESH  = 0.12    


# ══════════════════════════════════════════════════════════════════════════════
#  PathMonitor y ContactMonitor
# ══════════════════════════════════════════════════════════════════════════════
class PathMonitor:
    def __init__(self):
        self._cps = np.array(PATH_CHECKPOINTS, dtype=np.float64)
        self.reset()

    def reset(self, start_cp_idx: int = 0, start_xy: np.ndarray = None):
        if start_xy is None:
            start_xy = np.array(SPAWN_XYZ[:2])
        self.current_cp   = min(start_cp_idx + 1, CP_FINAL_IDX)
        self._prev_dist   = self._dist_to_cp(self.current_cp, start_xy)
        self.cps_crossed  = start_cp_idx  
        self.completed    = False

    @property
    def next_cp_xy(self) -> np.ndarray:
        return self._cps[min(self.current_cp, CP_FINAL_IDX)]

    def _dist_to_cp(self, cp_idx: int, xy: np.ndarray) -> float:
        return float(np.linalg.norm(xy - self._cps[cp_idx]))

    def update(self, xy: np.ndarray) -> Tuple[float, float, bool]:
        if self.completed: return 0.0, 0.0, True

        cp_idx = self.current_cp
        dist_now = self._dist_to_cp(cp_idx, xy)

        delta_dist     = self._prev_dist - dist_now
        path_reward    = R_PATH_PROGRESS * delta_dist
        self._prev_dist = dist_now

        wrong_dir_pen = P_WRONG_DIR * abs(delta_dist) if delta_dist < -0.01 else 0.0

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


class ContactMonitor:
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
            g1, g2 = self._geom_name(c.geom1), self._geom_name(c.geom2)
            b1 = self._body_name(self._model.geom_bodyid[c.geom1])
            b2 = self._body_name(self._model.geom_bodyid[c.geom2])

            is_chassis1 = g1 in self._CHASSIS_PARTS or b1 in self._CHASSIS_PARTS
            is_chassis2 = g2 in self._CHASSIS_PARTS or b2 in self._CHASSIS_PARTS
            is_arm1     = g1 in self._ARM_PARTS     or b1 in self._ARM_PARTS
            is_arm2     = g2 in self._ARM_PARTS     or b2 in self._ARM_PARTS
            is_robot    = is_chassis1 or is_chassis2 or is_arm1 or is_arm2

            if ("muerte_" in g1 or "muerte_" in g2) and is_robot:
                self.robot_hit_muerte = True

            if ("fatal_" in g1 and is_arm2) or ("fatal_" in g2 and is_arm1):
                self.arm_hit_fatal = True


# ══════════════════════════════════════════════════════════════════════════════
#  AesirPalletsEnv (Arquitectura 14-Dim)
# ══════════════════════════════════════════════════════════════════════════════
class AesirPalletsEnv:
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
        self.renderer = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)
        self.num_lidar    = num_lidar_rays
        self.lidar_max    = lidar_max_range

        # ── Setup Actuadores 14-Dim ─────────────────────────────────────────
        self.ids_drive_l  = [self._aid(n) for n in DRIVE_LEFT]
        self.ids_drive_r  = [self._aid(n) for n in DRIVE_RIGHT]
        self.ids_flippers = [self._aid(n) for n in FLIPPERS]
        self.ids_flip_wh  = {self._aid(fn): [self._aid(w) for w in wns] for fn, wns in FLIP_WHEELS.items()}
        self.ids_arm      = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l    = self._aid(FINGER_L)
        self.id_fing_r    = self._aid(FINGER_R)

        # ── joint_states: qpos + qvel de OBS_ACTUATORS ────────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)
        self.joint_len    = 2 * len(self._obs_act_ids)

        # ── penalización energía del brazo ──────────────────────────────────
        self._arm_dof_adrs = []
        for jn in ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self._arm_dof_adrs.append(int(self.model.jnt_dofadr[jid]))

        self.lidar_sensor_adr = []
        for i in range(self.num_lidar):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid >= 0: self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        self.lidar_spin_id = self._aid(LIDAR_SPIN)
        self.base_id       = max(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "footprint_link"), 1)

        # ── dims ───────────────────────────────────────────────────────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (self.num_lidar,)
        self.act_len     = 14

        self.control_decimation = control_decimation
        self._dt                = self.model.opt.timestep * control_decimation
        self.max_steps          = max_steps
        self._joint_pos         = np.zeros(6, dtype=np.float64)

        # ── monitores ──────────────────────────────────────────────────────
        self._path_monitor    = PathMonitor()
        self._contact_monitor = ContactMonitor(self.model)

        self._step_counter  = 0
        self._stuck_counter = 0
        self._last_xy       = np.array(SPAWN_XYZ[:2], dtype=np.float64)
        self._jumped_fatal  = False

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 6.0
            self.viewer.cam.elevation = -25

    def _aid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _read_cameras(self) -> np.ndarray:
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            img = np.flip(img, axis=(0, 1))  # Parche cámara invertida
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        lidar = np.empty(self.num_lidar, dtype=np.float32)
        for i, adr in enumerate(self.lidar_sensor_adr):
            d = float(self.data.sensordata[adr])
            lidar[i] = d / self.lidar_max if 0 < d < self.lidar_max else 1.0
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        qpos = self.data.qpos[self._qpos_adr]
        qvel = self.data.qvel[self._qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    # ── Mapeo 14-Dim ───────────────────────────────────────────────────────
    def _apply_action(self, action: np.ndarray):
        a     = np.clip(action, -1.0, 1.0)
        v_lin = float(a[0]) * MAX_LINEAR_VEL
        omega = float(a[1]) * MAX_ANGULAR_VEL

        # Base Diferencial
        vl = float(np.clip((v_lin - omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
        vr = float(np.clip((v_lin + omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
        for i in self.ids_drive_l: self.data.ctrl[i] = vl
        for i in self.ids_drive_r: self.data.ctrl[i] = vr

        # Flippers y sus Ruedas
        for k, fid in enumerate(self.ids_flippers):
            fp = float(np.clip(a[2+k] * 3.1416, -3.1416, 3.1416))
            self.data.ctrl[fid] = fp
            wvel = vl if k in (0, 2) else vr
            for wid in self.ids_flip_wh.get(fid, []):
                self.data.ctrl[wid] = float(np.clip(wvel, -1.0, 1.0))

        # Brazo (Integrador)
        delta = a[6:12] * MAX_JOINT_VEL * self._dt
        self._joint_pos = np.clip(self._joint_pos + delta, -3.1416, 3.1416)
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]
        
        # Gripper
        self.data.ctrl[self.id_fing_l] = float(np.clip((a[12] + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_fing_r] = float(np.clip((a[13] + 1.0) / 2.0 * 0.03, 0.0, 0.03))

        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

    def _set_arm_rest(self):
        for nombre, angulo in ARM_REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo
        self._joint_pos = np.array([ARM_REST_ANGLES[f"joint_{i+1}"] for i in range(6)], dtype=np.float64)

    # ── Reset y Step ───────────────────────────────────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

        # Domain Randomization: Spawn a lo largo del path
        start_idx = int(np.random.randint(0, len(PATH_CHECKPOINTS) - 1))
        spawn_x, spawn_y = PATH_CHECKPOINTS[start_idx]
        
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = 0.25
        self.data.qpos[3:7] = [1, 0, 0, 0]

        self._set_arm_rest()

        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

        self._step_counter  = 0
        self._stuck_counter = 0
        self._jumped_fatal  = False

        base_pos = self.data.xpos[self.base_id]
        self._last_xy = base_pos[:2].copy()

        self._path_monitor.reset(start_cp_idx=start_idx, start_xy=self._last_xy)
        self._contact_monitor.reset_flags()

        return self._observation()

    def step(self, action: np.ndarray):
        self._apply_action(action)

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

        self._step_counter += 1
        self._contact_monitor.scan(self.data)

        obs    = self._observation()
        reward = self._compute_reward(obs, action)
        done   = self._terminated()

        return obs, reward, done, {
            "checkpoint":   self._path_monitor.current_cp,
            "cps_crossed":  self._path_monitor.cps_crossed,
            "completed":    self._path_monitor.completed,
        }

    def _compute_reward(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
        reward = 0.0
        base_pos = self.data.xpos[self.base_id]
        xy = base_pos[:2].copy()
        current_z = float(base_pos[2])

        # ── 1. PROGRESO EN EL PATH ─────────────────────────────────────────
        path_r, cp_bonus, _ = self._path_monitor.update(xy)
        reward += path_r + cp_bonus

        # ── 2. ESTABILIDAD Y ANTI-SALTOS (Mecánicas Físicas) ───────────────
        # Si el robot se eleva por encima de los 45cm (volando)
        if current_z > 0.45:
            self._jumped_fatal = True
            reward += P_JUMP_FATAL

        # Penalización por perder el paralelismo con el suelo (Anti-Wheelie)
        # zmat[2,2] es 1.0 si el robot está plano, y se acerca a 0 si se inclina
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        uprightness = float(zmat[2, 2])
        if uprightness < 0.90:  
            reward += P_STABILITY * (0.90 - uprightness)

        # Penalización por sacudidas y giros bruscos del chasis
        # qvel[3:6] son las velocidades angulares del freejoint (roll, pitch, yaw)
        base_ang_vel = self.data.qvel[3:6]
        reward += P_SHAKE * float(np.sum(np.square(base_ang_vel)))

        # ── 3. ALIVE & SMOOTH DRIVE ────────────────────────────────────────
        reward += R_ALIVE
        vel = xy - self._last_xy
        to_cp = self._path_monitor.next_cp_xy - xy
        to_cp_norm = np.linalg.norm(to_cp)
        if to_cp_norm > 1e-4 and np.linalg.norm(vel) > 1e-4:
            alignment = float(np.dot(vel, to_cp) / (np.linalg.norm(vel) * to_cp_norm))
            if alignment > 0.5: reward += R_SMOOTH_DRIVE * alignment
        
        # ── 4. STUCK PENALTY ───────────────────────────────────────────────
        if float(np.linalg.norm(vel)) < 0.005:
            self._stuck_counter += 1
            reward += P_STUCK
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)
        self._last_xy = xy.copy()

        # ── 5. LIDAR ───────────────────────────────────────────────────────
        min_lidar = float(obs["lidar"].min())
        if min_lidar < LIDAR_DANGER_THRESH:
            reward += P_LIDAR_NEAR * (LIDAR_DANGER_THRESH - min_lidar)

        # ── 6. ACTION COST (Usa las señales de control reales) ─────────────
        reward += P_ACTION_COST * float(np.square(self.data.ctrl[self._obs_act_ids]).mean())

        # ── 7. ARM ENERGY ──────────────────────────────────────────────────
        if self._arm_dof_adrs:
            arm_vel = np.array([self.data.qvel[adr] for adr in self._arm_dof_adrs])
            reward += P_ARM_ENERGY * float(np.sum(np.abs(arm_vel)))

        # ── 8. FLIPPER OVERUSE (Basado en la acción normalizada -1..1) ─────
        reward += P_FLIP_OVERUSE * float(np.sum(np.abs(action[2:6])))

        # ── 9. MUERTE ──────────────────────────────────────────────────────
        if self._contact_monitor.arm_hit_fatal: reward += P_ARM_FATAL
        if self._contact_monitor.robot_hit_muerte: reward += P_MUERTE

        return float(reward)

    def _terminated(self) -> bool:
        if self._step_counter >= self.max_steps: return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        # Volteo drástico
        if float(zmat[2, 2]) < 0.20: return True
        if self._stuck_counter >= STUCK_MAX_STEPS: return True
        
        # Nuevas condiciones fatales
        if self._jumped_fatal: return True
        if self._contact_monitor.robot_hit_muerte: return True
        if self._contact_monitor.arm_hit_fatal: return True
        if self._path_monitor.completed: return True
        return False

    def close(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  Redes Neurales y PPO
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
#  Loop de entrenamiento principal
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
          image_log_every: int   = 25,
          resume_from:     str   = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Device: {device}")

    # PREVENCIÓN CORE DUMP: Previene bug de concurrencia PyTorch+MuJoCo Viewer
    _ = torch.optim.Adam([torch.nn.Parameter(torch.empty(1))])

    env = AesirPalletsEnv(render=render)
    print(f"act_len={env.act_len} (14-Dim) image={env.image_shape} "
          f"lidar={env.num_lidar} joints={env.joint_len}")

    policy = ConvActorCritic(
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    start_iter = 0
    best_avg   = -1e9

    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0)
        best_avg   = ckpt.get("avg_ep_r", -1e9)
        print(f"Resumiendo desde iter {start_iter}  (best_avg={best_avg:.2f})")

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
            "act_dim": env.act_len, "differential_drive": True
        })
        wandb.watch(policy, log="gradients", log_freq=100)

    obs = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_history: List[float] = []
    ep_cps_history: List[int] = []    
    
    try:
        for it in range(start_iter, start_iter + num_iterations):
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
                    "iter": it, "global_step": (it - start_iter + 1)*steps_per_iter,
                    "avg_ep_reward": avg_r, "avg_cps_crossed": avg_cps,
                    "policy_loss": stats["pi"], "value_loss": stats["v"],
                    "entropy": stats["ent"],
                    "mean_log_std": policy.log_std.detach().mean().item(),
                    "iter_time_s": dt,
                }
                if it % image_log_every == 0:
                    log["cameras"] = make_camera_panel(obs["images"], env.camera_names)
                wandb.log(log, step=(it - start_iter + 1)*steps_per_iter)

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
    # Recuerda: render=True para ver la ventana, render=False para entrenar rápido
    train(render=True)