"""
ppo_conv_train.py  —  PPO para el robot Aesir con política convolucional multi-modal.

Observaciones (por step):
  images      : 3 cámaras RGB (cam_gripper, cam_oakd, cam_back) apiladas
                como tensor (9, H, W) float en [0,1]  → CNN encoder
  lidar       : 7 rangefinder rays, normalizados por LIDAR_MAX_RANGE  → MLP
  joint_states: qpos + qvel de los actuadores relevantes               → MLP

Acciones (14 dims, normalizadas en [-1,1]):
  [0]    v_lin       velocidad lineal de la base   → capa differential drive
  [1]    ω_ang       velocidad angular de la base  → capa differential drive
  [2..5] flipper_1..4  posición objetivo de cada flipper
  [6..11] joint_1..6  velocidad articular del brazo (integrada a posición)
  [12]   dedo izquierdo
  [13]   dedo derecho

La capa differential drive convierte (v_lin, ω_ang) en velocidades
individuales para los 6 actuadores vel_drive_* del modelo.
Las 8 ruedas de los flippers (vel_flip*) se sincronizan con el tracker
al que pertenecen (izq: flippers 1,3 — der: flippers 2,4).
El brazo usa un integrador de velocidad equivalente a MoveIt Servo.

vel_lidar_spin NO es parte de la acción; el env lo mantiene constante.

Uso:
    cd model_robot
    MUJOCO_GL=egl python3 ppo_conv_train.py

    # Sin render, más rápido:
    python3 -c "from ppo_conv_train import train; train(render=False)"

    # Reanudar desde checkpoint:
    python3 -c "from ppo_conv_train import train; train(resume_from='checkpoints/ppo_conv_best.pt')"
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


# ──────────────────────────── Ruta XML (auto-detecta) ──────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_FULL    = os.path.join(_HERE, "../aesir_robot_description/launch/aesir_complete.xml")
_ROBOT   = os.path.join(_HERE, "aesir_mujoco.xml")
XML_PATH = _FULL if os.path.exists(_FULL) else _ROBOT


# ──────────────────────────── Config (editar aquí) ─────────────────────────
CAMERA_NAMES        = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W  = 84, 84
NUM_LIDAR_RAYS      = 7
LIDAR_MAX_RANGE     = 15.0
LIDAR_SPIN_VEL      = 20.0        # rad/s, mantenido constante

# ── Parámetros físicos del differential drive ─────────────────────────────
# Ajusta TRACK_HALF_WIDTH y WHEEL_RADIUS midiendo en tu XML:
#   TRACK_HALF_WIDTH = distancia_Y_entre_trackers / 2
#   WHEEL_RADIUS     = radio de la ruedita de tracción en MuJoCo
TRACK_HALF_WIDTH    = 0.21        # m
WHEEL_RADIUS        = 0.05        # m
MAX_WHEEL_VEL       = 20.0        # rad/s (= ctrlrange vel_drive_*)
MAX_LINEAR_VEL      = 1.5         # m/s
MAX_ANGULAR_VEL     = 2.0         # rad/s
MAX_JOINT_VEL       = 1.0         # rad/s por articulación del brazo

# ── Nombres de actuadores agrupados ──────────────────────────────────────
DRIVE_LEFT   = ["vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3"]
DRIVE_RIGHT  = ["vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3"]
FLIPPERS     = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
FLIP_WHEELS  = {                  # flipper actuator → sus rueditas
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

# Actuadores usados en la observación joint_states (sin rueditas de flipper)
OBS_ACTUATORS = DRIVE_LEFT + DRIVE_RIGHT + FLIPPERS + ARM_JOINTS + [FINGER_L, FINGER_R]

REST_ANGLES = {
    "joint_1": -0.314,
    "joint_2": -3.14,
    "joint_3":  3.14,
    "joint_4":  0.0,
    "joint_5":  0.0,
    "joint_6":  0.0,
}

CONTROL_DECIMATION  = 10
EPISODE_MAX_STEPS   = 1000
CHECKPOINT_DIR      = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────── Env ──────────────────────────────────────
class AesirMuJoCoEnv:
    """
    Env multi-modal: cámaras + lidar + joint states → 14 acciones.

    Acciones en [-1, 1]:
      [0]    v_lin   → differential drive → vel_drive_l_* y vel_drive_r_*
      [1]    ω_ang   → differential drive
      [2..5] flipper_1..4 (posición, ±π)
      [6..11] joint_1..6 (velocidad integrada → posición)
      [12]   dedo izq
      [13]   dedo der
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

        # ── cámaras ────────────────────────────────────────────────────────
        self.image_h, self.image_w = image_hw
        self.renderer     = mujoco.Renderer(self.model,
                                            height=self.image_h,
                                            width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)

        # ── lidar ──────────────────────────────────────────────────────────
        self.num_lidar = num_lidar_rays
        self.lidar_max = lidar_max_range
        self.lidar_sensor_adr = []
        for i in range(self.num_lidar):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} no encontrado en el modelo")
            self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        self.lidar_spin_id = self._aid(LIDAR_SPIN)

        # ── actuadores de base (differential drive) ────────────────────────
        self.ids_drive_l  = [self._aid(n) for n in DRIVE_LEFT]
        self.ids_drive_r  = [self._aid(n) for n in DRIVE_RIGHT]
        self.ids_flippers = [self._aid(n) for n in FLIPPERS]
        self.ids_flip_wh  = {
            self._aid(fname): [self._aid(w) for w in wnames]
            for fname, wnames in FLIP_WHEELS.items()
        }

        # ── actuadores del brazo ───────────────────────────────────────────
        self.ids_arm   = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l = self._aid(FINGER_L)
        self.id_fing_r = self._aid(FINGER_R)

        # ── joint_states: qpos + qvel de OBS_ACTUATORS ────────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)
        self.joint_len    = 2 * len(self._obs_act_ids)

        # ── bodies ─────────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── tamaños de obs y acción ────────────────────────────────────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (self.num_lidar,)
        self.joint_shape = (self.joint_len,)
        self.act_len     = 14

        # ── integrador de velocidad del brazo ─────────────────────────────
        self._joint_pos = np.zeros(6, dtype=np.float64)
        self._dt        = self.model.opt.timestep * control_decimation

        # ── estado ─────────────────────────────────────────────────────────
        self.control_decimation = control_decimation
        self.max_steps          = max_steps
        self._step_counter      = 0
        self._stuck_counter     = 0
        self._last_base_xy      = np.zeros(2)
        self._brazo_toco_fatal  = False
        self._toco_zona_muerte  = False

        # ── misiones ───────────────────────────────────────────────────────
        self.piezas_robot = {
            "base_link", "tracked_1", "tracked_2",
            "flipper_1_1", "flipper_2_1", "flipper_3_1", "flipper_4_1"
        }
        self.piezas_brazo = {
            "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
            "logitech_gripper_assembly", "left_finger_link", "right_finger_link"
        }
        self.nombres_pallets = [f"fatal_pallet {i}" for i in range(1, 19)]
        self._reset_estado_misiones()

        # ── viewer ─────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 4.0
            self.viewer.cam.elevation = -20

    # ── utilidad ───────────────────────────────────────────────────────────
    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuador no encontrado: '{name}'")
        return i

    # ── Capa differential drive ────────────────────────────────────────────
    def _apply_base_action(self, v_lin: float, omega: float, flipper_cmds: np.ndarray):
        vl = float(np.clip(
            (v_lin - omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
            -MAX_WHEEL_VEL, MAX_WHEEL_VEL
        ))
        vr = float(np.clip(
            (v_lin + omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
            -MAX_WHEEL_VEL, MAX_WHEEL_VEL
        ))
        for i in self.ids_drive_l:
            self.data.ctrl[i] = vl
        for i in self.ids_drive_r:
            self.data.ctrl[i] = vr
        for k, fid in enumerate(self.ids_flippers):
            fp = float(np.clip(flipper_cmds[k] * 3.1416, -3.1416, 3.1416))
            self.data.ctrl[fid] = fp
            wvel = vl if k in (0, 2) else vr   # izq: 0,2 — der: 1,3
            for wid in self.ids_flip_wh.get(fid, []):
                self.data.ctrl[wid] = float(np.clip(wvel, -1.0, 1.0))

    # ── Integrador de velocidad del brazo ─────────────────────────────────
    def _apply_arm_action(self, vel_joints: np.ndarray, fing_l: float, fing_r: float):
        delta = np.clip(vel_joints, -1.0, 1.0) * MAX_JOINT_VEL * self._dt
        self._joint_pos = np.clip(self._joint_pos + delta, -3.1416, 3.1416)
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]
        self.data.ctrl[self.id_fing_l] = float(np.clip((fing_l + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_fing_r] = float(np.clip((fing_r + 1.0) / 2.0 * 0.03, 0.0, 0.03))

    def _apply_action(self, action: np.ndarray):
        a     = np.clip(action, -1.0, 1.0)
        v_lin = float(a[0]) * MAX_LINEAR_VEL
        omega = float(a[1]) * MAX_ANGULAR_VEL
        self._apply_base_action(v_lin, omega, a[2:6])
        self._apply_arm_action(a[6:12], float(a[12]), float(a[13]))
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

    # ── Observaciones ──────────────────────────────────────────────────────
    def _read_cameras(self) -> np.ndarray:
        """→ (9, H, W) float32 en [0,1].

        np.flip(..., axis=(0,1)) equivale a cv2.flip(img, -1):
        voltea tanto vertical como horizontalmente para corregir la
        orientación del renderer de MuJoCo.
        """
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            img = np.flip(img, axis=(0, 1))          # equivale a cv2.flip(..., -1)
            frames.append(img.astype(np.float32) / 255.0)
        stacked = np.concatenate(frames, axis=-1)    # (H, W, 9)
        return np.transpose(stacked, (2, 0, 1))      # (9, H, W)

    def _read_lidar(self) -> np.ndarray:
        """→ (7,) float32 en [0,1]."""
        lidar = np.empty(self.num_lidar, dtype=np.float32)
        for i, adr in enumerate(self.lidar_sensor_adr):
            d = float(self.data.sensordata[adr])
            if d <= 0.0 or d >= self.lidar_max:
                d = self.lidar_max
            lidar[i] = d / self.lidar_max
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        """→ (2*N_obs_act,) float32: qpos ‖ qvel."""
        qpos = self.data.qpos[self._qpos_adr]
        qvel = self.data.qvel[self._qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    # ── Misiones y colisiones ──────────────────────────────────────────────
    def _reset_estado_misiones(self):
        self.pallets_visitados   = {n: False for n in self.nombres_pallets}
        self.puerta_desbloqueada = False

    def _obtener_contactos_del_robot(self) -> set:
        objetos_tocados = set()
        self._brazo_toco_fatal = False
        self._toco_zona_muerte = False
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[c.geom2]) or ""
            is_rob1 = g1 in self.piezas_robot or b1 in self.piezas_robot
            is_rob2 = g2 in self.piezas_robot or b2 in self.piezas_robot
            is_arm1 = g1 in self.piezas_brazo or b1 in self.piezas_brazo
            is_arm2 = g2 in self.piezas_brazo or b2 in self.piezas_brazo
            if "muerte_" in g1 or "muerte_" in g2:
                if is_rob1 or is_rob2 or is_arm1 or is_arm2:
                    self._toco_zona_muerte = True
            if "fatal_" in g1 or "fatal_" in g2:
                if ("fatal_" in g1 and is_arm2) or ("fatal_" in g2 and is_arm1):
                    self._brazo_toco_fatal = True
            if is_rob1 and g2: objetos_tocados.add(g2)
            elif is_rob2 and g1: objetos_tocados.add(g1)
        return objetos_tocados

    # ── Reward ────────────────────────────────────────────────────────────
    def _reward(self, obs: Dict[str, np.ndarray]) -> float:
        base_xy = self.data.xpos[self.base_id, :2]
        dx      = float(base_xy[0] - self._last_base_xy[0])
        self._last_base_xy = base_xy.copy()

        if abs(dx) < 0.005:
            self._stuck_counter += 1
            pen_inactiv = 0.05
        else:
            self._stuck_counter = 0
            pen_inactiv = 0.0

        min_lidar    = float(obs["lidar"].min())
        obstacle_pen = max(0.0, 0.1 - min_lidar) * 5.0

        # action_cost sobre TODOS los actuadores (base + brazo + flippers + rueditas)
        action_cost = 1e-3 * float(np.square(self.data.ctrl[self._obs_act_ids]).mean())

        alive_bonus = 0.01
        return (10.0 * dx) + alive_bonus - obstacle_pen - action_cost - pen_inactiv

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        if self._step_counter >= self.max_steps: return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.2: return True
        if self._stuck_counter > 50: return True
        if self._toco_zona_muerte: return True
        if self._brazo_toco_fatal: return True
        return False

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = -1.5
        self.data.qpos[1] =  3.5
        self.data.qpos[2] =  0.2
        for nombre, angulo in REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo
        self._joint_pos = np.array([REST_ANGLES[f"joint_{i+1}"] for i in range(6)])
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL
        door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        if door_id >= 0:
            self.model.jnt_range[door_id][0] = 0.0
            self.model.jnt_range[door_id][1] = 0.0
        self._reset_estado_misiones()
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
        self._step_counter     = 0
        self._stuck_counter    = 0
        self._brazo_toco_fatal = False
        self._toco_zona_muerte = False
        self._last_base_xy     = self.data.xpos[self.base_id, :2].copy()
        return self._observation()

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
        self._step_counter += 1
        obs         = self._observation()
        step_reward = self._reward(obs)
        objetos     = self._obtener_contactos_del_robot()

        for pallet in self.nombres_pallets:
            if pallet in objetos and not self.pallets_visitados[pallet]:
                self.pallets_visitados[pallet] = True
                if pallet == "fatal_pallet 18":
                    step_reward += 500.0
                    for p in self.nombres_pallets:
                        self.pallets_visitados[p] = False
                else:
                    step_reward += 50.0

        handle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "handle_hinge")
        door_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        if handle_id >= 0 and door_id >= 0:
            ha = self.data.qpos[self.model.jnt_qposadr[handle_id]]
            if 0.9 <= abs(ha) <= 1.0 and not self.puerta_desbloqueada:
                self.puerta_desbloqueada = True
                self.model.jnt_range[door_id][0] = -1.5
                self.model.jnt_range[door_id][1] =  1.5
                step_reward += 50.0

        if self._toco_zona_muerte:
            step_reward -= 10.0
        if self._brazo_toco_fatal:
            step_reward -= 50.0

        done = self._terminated()
        return obs, step_reward, done, {}

    def close(self) -> None:
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass


# ──────────────────────────────── Red neuronal ─────────────────────────────
class ImageEncoder(nn.Module):
    """Nature-CNN para las 3 cámaras apiladas (9 canales)."""

    def __init__(self, in_channels: int, h: int, w: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32,         64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,         64, 3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, h, w)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class StateEncoder(nn.Module):
    """MLP para el vector (lidar ‖ joint_states)."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, out_dim), nn.Tanh(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvActorCritic(nn.Module):
    """Actor-crítico multi-modal. Gaussiana diagonal con log-std aprendible."""

    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 lidar_dim: int,
                 joint_dim: int,
                 act_dim: int,
                 img_feat: int = 256,
                 vec_feat: int = 128,
                 hidden: int = 256,
                 log_std_init: float = -0.5):
        super().__init__()
        c, h, w = image_shape
        self.img_enc = ImageEncoder(c, h, w, out_dim=img_feat)
        self.vec_enc = StateEncoder(lidar_dim + joint_dim, out_dim=vec_feat)
        fused_dim = img_feat + vec_feat
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,    hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_dim  = act_dim

    def _fuse(self, images, lidar, joints):
        return self.trunk(torch.cat([
            self.img_enc(images),
            self.vec_enc(torch.cat([lidar, joints], dim=-1))
        ], dim=-1))

    def forward(self, images, lidar, joints):
        z       = self._fuse(images, lidar, joints)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        std     = log_std.exp().expand_as(mu)
        return mu, std, value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], device):
        images = obs["images"].unsqueeze(0).to(device)
        lidar  = obs["lidar"].unsqueeze(0).to(device)
        joints = obs["joint_states"].unsqueeze(0).to(device)
        mu, std, value = self(images, lidar, joints)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return (raw.squeeze(0).cpu().numpy(),
                float(logp.item()),
                float(value.item()))

    def evaluate(self, images, lidar, joints, actions):
        mu, std, value = self(images, lidar, joints)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return logp, value, entropy


# ──────────────────────────── Rollout buffer ───────────────────────────────
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
        self.images  = np.zeros((self.capacity, c, h, w),       dtype=np.float32)
        self.lidars  = np.zeros((self.capacity, self.lidar_dim), dtype=np.float32)
        self.joints  = np.zeros((self.capacity, self.joint_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim),   dtype=np.float32)
        self.logps   = np.zeros(self.capacity,  dtype=np.float32)
        self.rewards = np.zeros(self.capacity,  dtype=np.float32)
        self.values  = np.zeros(self.capacity,  dtype=np.float32)
        self.dones   = np.zeros(self.capacity,  dtype=np.float32)

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

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        advantages = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv    = last_value if t == self.capacity - 1 else self.values[t + 1]
            nd    = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nv * nd - self.values[t]
            gae   = delta + gamma * gae_lambda * nd * gae
            advantages[t] = gae
        returns = advantages + self.values
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        return (advantages - adv_mean) / adv_std, returns


# ──────────────────────────── PPO update ───────────────────────────────────
def ppo_update(policy, optimizer, buffer, advantages, returns,
               ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device):
    images  = torch.as_tensor(buffer.images,  dtype=torch.float32, device=device)
    lidar   = torch.as_tensor(buffer.lidars,  dtype=torch.float32, device=device)
    joints  = torch.as_tensor(buffer.joints,  dtype=torch.float32, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.float32, device=device)
    old_log = torch.as_tensor(buffer.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv     = torch.as_tensor(advantages,     dtype=torch.float32, device=device).unsqueeze(-1)
    ret     = torch.as_tensor(returns,        dtype=torch.float32, device=device).unsqueeze(-1)

    metrics = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(ppo_epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buffer.capacity)),
                                batch_size, drop_last=False):
            logp, value, entropy = policy.evaluate(
                images[idx], lidar[idx], joints[idx], actions[idx]
            )
            ratio  = torch.exp(logp - old_log[idx])
            surr1  = ratio * adv[idx]
            surr2  = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[idx]
            pl     = -torch.min(surr1, surr2).mean()
            vl     = F.smooth_l1_loss(value, ret[idx])
            loss   = pl + vf_coef * vl - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            metrics["pi"]  = pl.item()
            metrics["v"]   = vl.item()
            metrics["ent"] = entropy.item()
    return metrics


# ──────────────────────────── Helpers ──────────────────────────────────────
def obs_to_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float() for k, v in obs.items()}


def save_checkpoint(path: Path, policy, optimizer, iter_idx: int, avg_ep_r: float):
    torch.save({
        "iter":      iter_idx,
        "policy":    policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "avg_ep_r":  avg_ep_r,
    }, path)


def make_camera_panel(obs_images: np.ndarray, camera_names: List[str]):
    n = len(camera_names)
    h, w = obs_images.shape[1], obs_images.shape[2]
    panel = np.zeros((h, w * n, 3), dtype=np.uint8)
    for k in range(n):
        cam = obs_images[k * 3:(k + 1) * 3]
        cam = (cam.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        panel[:, k * w:(k + 1) * w, :] = cam
    return wandb.Image(panel, caption=" | ".join(camera_names))


# ──────────────────────────── Training loop ────────────────────────────────
def train(num_iterations:  int   = 500,
          steps_per_iter:  int   = 2048,
          ppo_epochs:      int   = 10,
          batch_size:      int   = 256,
          gamma:           float = 0.99,
          gae_lambda:      float = 0.95,
          clip_param:      float = 0.2,
          vf_coef:         float = 0.5,
          ent_coef:        float = 0.005,
          lr:              float = 3e-4,
          save_every:      int   = 50,
          device_str:      str   = "auto",
          render:          bool  = False,
          use_wandb:       bool  = True,
          wandb_project:   str   = "AIDL-PPO-AESIR-CONV",
          wandb_run_name:  str   = None,
          image_log_every: int   = 25,
          resume_from:     str   = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Dispositivo: {device}")

    env = AesirMuJoCoEnv(render=render)
    print(f"act_len     = {env.act_len}  (14 = v_lin, ω, flip×4, joint×6, dedos×2)")
    print(f"image_shape = {env.image_shape}  (3 cámaras × 3 canales)")
    print(f"lidar_dim   = {env.num_lidar}")
    print(f"joint_dim   = {env.joint_len}  (qpos+qvel de {len(OBS_ACTUATORS)} actuadores)")
    print(f"XML         = {XML_PATH}")

    policy    = ConvActorCritic(
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

    buffer = RolloutBuffer(
        capacity=steps_per_iter,
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    )

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=wandb_project, name=wandb_run_name,
            config=dict(
                num_iterations=num_iterations, steps_per_iter=steps_per_iter,
                ppo_epochs=ppo_epochs, batch_size=batch_size, gamma=gamma,
                gae_lambda=gae_lambda, clip_param=clip_param, vf_coef=vf_coef,
                ent_coef=ent_coef, lr=lr, act_dim=env.act_len,
                image_shape=list(env.image_shape), lidar_dim=env.num_lidar,
                joint_dim=env.joint_len, control_decimation=env.control_decimation,
                episode_max_steps=env.max_steps,
                track_half_width=TRACK_HALF_WIDTH, wheel_radius=WHEEL_RADIUS,
            ),
        )
        wandb.watch(policy, log="gradients", log_freq=100)

    obs        = env.reset()
    ep_reward  = 0.0
    ep_history: List[float] = []

    try:
        for it in range(start_iter, start_iter + num_iterations):
            t0 = time.time()
            for _ in range(steps_per_iter):
                obs_t             = obs_to_tensor(obs)
                action, logp, val = policy.act(obs_t, device)
                next_obs, rew, done, _ = env.step(action)
                buffer.store(obs, action, logp, rew, val, done)
                obs       = next_obs
                ep_reward += rew
                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20: ep_history.pop(0)
                    ep_reward = 0.0
                    obs = env.reset()

            obs_t = obs_to_tensor(obs)
            with torch.no_grad():
                _, _, lv = policy(
                    obs_t["images"].unsqueeze(0).to(device),
                    obs_t["lidar"].unsqueeze(0).to(device),
                    obs_t["joint_states"].unsqueeze(0).to(device),
                )
            advantages, returns = buffer.compute_gae(float(lv.item()), gamma, gae_lambda)
            metrics = ppo_update(policy, optimizer, buffer, advantages, returns,
                                 ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)

            avg_ep_r = float(np.mean(ep_history)) if ep_history else float("nan")
            dt       = time.time() - t0
            print(f"[Iter {it:4d}] avg_ep_r={avg_ep_r:8.2f}  "
                  f"pi={metrics['pi']:+.4f}  v={metrics['v']:.4f}  "
                  f"ent={metrics['ent']:.3f}  ({dt:.1f}s)")

            if use_wandb:
                log_dict = {
                    "iter": it, "avg_ep_r": avg_ep_r,
                    "policy_loss": metrics["pi"], "value_loss": metrics["v"],
                    "entropy": metrics["ent"],
                    "mean_log_std": policy.log_std.detach().mean().item(),
                    "iter_time_s": dt,
                    "global_step": (it - start_iter + 1) * steps_per_iter,
                }
                if (it % image_log_every) == 0:
                    log_dict["camera_views"] = make_camera_panel(obs["images"], env.camera_names)
                wandb.log(log_dict, step=(it - start_iter + 1) * steps_per_iter)

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"ppo_conv_iter{it+1:05d}.pt"
                save_checkpoint(p, policy, optimizer, it + 1, avg_ep_r)
                print(f"  ↳ checkpoint: {p}")
                if use_wandb: wandb.save(str(p), base_path=str(CHECKPOINT_DIR))

            if ep_history and avg_ep_r > best_avg:
                best_avg  = avg_ep_r
                best_path = CHECKPOINT_DIR / "ppo_conv_best.pt"
                save_checkpoint(best_path, policy, optimizer, it + 1, avg_ep_r)
                print(f"  ↳ NUEVO MEJOR ({avg_ep_r:.2f}) → {best_path}")
                if use_wandb:
                    wandb.run.summary["best_avg_ep_r"] = best_avg
                    wandb.save(str(best_path), base_path=str(CHECKPOINT_DIR))

    finally:
        env.close()
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    train(render=True)