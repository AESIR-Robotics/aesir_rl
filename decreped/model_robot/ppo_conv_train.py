"""
PPO trainer for the Aesir rescue robot with a convolutional multi-modal policy.

Observations (per env step):
  * images        : 3 RGB cameras (cam_gripper, cam_oakd, cam_back) stacked as
                    a (9, H, W) float tensor in [0, 1]  -> Conv encoder
  * lidar         : 7 rangefinder rays, normalized by LIDAR_MAX_RANGE  -> MLP
  * joint_states  : qpos + qvel of the 26 actuated joints              -> MLP

Actions (continuous, 26 dims, normalized to [-1, 1] and rescaled to ctrlrange):
  * 6 drive wheels   (vel_drive_l_{1..3}, vel_drive_r_{1..3})
  * 4 flippers       (pos_flipper_{1..4})
  * 6 arm joints     (pos_joint_{1..6})
  * 2 gripper        (pos_left_finger, pos_right_finger)
  * 8 flipper wheels (vel_flip{1..4}_{back,front})

`vel_lidar_spin` is *not* part of the action; the env keeps it at a constant
speed so the policy doesn't have to learn to spin the lidar itself.

Checkpoints (policy + optimizer + iter + avg reward) are saved to
./checkpoints/ every `save_every` iterations, plus a `ppo_conv_best.pt`
that tracks the best running average episode reward.

Usage:
    cd ./model_robot
    MUJOCO_GL=egl python3 ppo_conv_train.py
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

from path_utils import get_xml_path, get_checkpoint_dir


# ──────────────────────────── Config (edit me) ─────────────────────────────
XML_PATH = str(get_xml_path())

CAMERA_NAMES      = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W = 84, 84               # downscaled images for the CNN
NUM_LIDAR_RAYS    = 7
LIDAR_MAX_RANGE   = 15.0                  # used for normalization
LIDAR_SPIN_VEL    = 20.0                  # rad/s, held constant by the env

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

CONTROL_DECIMATION = 10                   # physics steps per env.step()
EPISODE_MAX_STEPS  = 1000                 # env steps per episode

CHECKPOINT_DIR = get_checkpoint_dir()


# ──────────────────────────────── Env ──────────────────────────────────────
class AesirMuJoCoEnv:
    """Multi-modal MuJoCo env: cameras + lidar + joint states -> action."""

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
        self.timestep = self.model.opt.timestep

        self.image_h, self.image_w = image_hw
        self.renderer = mujoco.Renderer(self.model,
                                        height=self.image_h,
                                        width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)
        self.num_lidar    = num_lidar_rays
        self.lidar_max    = lidar_max_range
        self.control_decimation = control_decimation
        self.max_steps    = max_steps

        # ── actuator indices and ctrl ranges ───────────────────────────────
        self.act_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ACTUATOR_NAMES
        ], dtype=np.int32)
        if np.any(self.act_ids < 0):
            missing = [n for n, i in zip(ACTUATOR_NAMES, self.act_ids) if i < 0]
            raise ValueError(f"Actuators missing in model: {missing}")
        self.ctrlrange = self.model.actuator_ctrlrange[self.act_ids].copy()
        self.act_low   = self.ctrlrange[:, 0]
        self.act_high  = self.ctrlrange[:, 1]
        self.act_len   = len(self.act_ids)

        # ── joints driven by these actuators (for qpos / qvel) ─────────────
        self.joint_ids = np.array([
            self.model.actuator_trnid[i, 0] for i in self.act_ids
        ], dtype=np.int32)
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.qvel_adr = np.array(
            [self.model.jnt_dofadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.joint_len = 2 * self.act_len   

        # ── lidar spin (held constant) ─────────────────────────────────────
        self.lidar_spin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_lidar_spin"
        )

        # ── lidar sensors ──────────────────────────────────────────────────
        self.lidar_sensor_adr = []
        for i in range(self.num_lidar):
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}"
            )
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} not found in model")
            self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (self.num_lidar,)
        self.joint_shape = (self.joint_len,)

        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_id < 0:
            self.base_id = 1  

        self._step_counter = 0
        self._stuck_counter = 0  
        self._last_base_xy = np.zeros(2)

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 4.0
            self.viewer.cam.elevation = -20

        # ── Inicialización de Estado y Colisiones ──────────────────────────
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

    # ── helpers ────────────────────────────────────────────────────────────
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
            if d <= 0.0 or d >= self.lidar_max:
                d = self.lidar_max
            lidar[i] = d / self.lidar_max
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

    def _reset_estado_misiones(self):
        """Reinicia el estado de los pallets y la puerta."""
        self.pallets_visitados = {nombre: False for nombre in self.nombres_pallets}
        self.puerta_desbloqueada = False

    def _obtener_contactos_del_robot(self) -> set:
        """Escanea contactos y administra reglas de vida/muerte."""
        objetos_tocados = set()
        self._brazo_toco_fatal = False
        self._toco_zona_muerte = False 
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ""
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ""
            
            is_rob1 = (geom1_name in self.piezas_robot) or (body1_name in self.piezas_robot)
            is_rob2 = (geom2_name in self.piezas_robot) or (body2_name in self.piezas_robot)
            
            is_arm1 = (geom1_name in self.piezas_brazo) or (body1_name in self.piezas_brazo)
            is_arm2 = (geom2_name in self.piezas_brazo) or (body2_name in self.piezas_brazo)

            # 1. "muerte_" mata al robot sin importar qué pieza lo toque
            if "muerte_" in geom1_name or "muerte_" in geom2_name:
                if is_rob1 or is_rob2 or is_arm1 or is_arm2:
                    self._toco_zona_muerte = True

            # 2. "fatal_" mata SOLO al brazo
            if "fatal_" in geom1_name or "fatal_" in geom2_name:
                if ("fatal_" in geom1_name and is_arm2) or ("fatal_" in geom2_name and is_arm1):
                    self._brazo_toco_fatal = True

            # 3. Recolección de misiones (Aplica solo al chasis)
            if is_rob1 and geom2_name: objetos_tocados.add(geom2_name)
            elif is_rob2 and geom1_name: objetos_tocados.add(geom1_name)
                
        return objetos_tocados

    # ── public API ─────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        
        # ======== SPAWN POINT MANUAL ========
        # Ajusta estas coordenadas (X, Y, Z) para cambiar dónde inicia el robot.
        self.data.qpos[0] = -1.5    # X pos 
        self.data.qpos[1] = 3.5   # Y pos 
        self.data.qpos[2] = 0.2    # Z pos (Un poco arriba para que caiga)
        # ====================================

        # ======== INICIALIZACIÓN DEL BRAZO EN REPOSO ========
        # Modifica estos radianes para encontrar el pliegue perfecto del brazo
        angulos_reposo = {
            "joint_1": -0.314,
            "joint_2": -3.14,  # -90 grados (hacia atrás)
            "joint_3": 3.14,   # +90 grados (plegado hacia la base)
            "joint_4": -1.35,
            "joint_5": -1.54,
            "joint_6": 1.54
        }
        
        for nombre_joint, angulo in angulos_reposo.items():
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre_joint)
            if jnt_id >= 0:
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = angulo
        # ====================================================

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
                
        self._step_counter = 0
        self._stuck_counter = 0  
        self._brazo_toco_fatal = False 
        self._toco_zona_muerte = False
        self._last_base_xy = self.data.xpos[self.base_id, :2].copy()
        
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
        obs = self._observation()
        step_reward = self._reward(obs)
        objetos_tocados = self._obtener_contactos_del_robot()

        # Recompensas: Pallets INCREMENTADAS
        for pallet in self.nombres_pallets:
            if pallet in objetos_tocados and not self.pallets_visitados[pallet]:
                self.pallets_visitados[pallet] = True
                
                # Checkeamos contra el pallet final
                if pallet == "fatal_pallet 18":
                    step_reward += 500.0   # <-- GRAN PREMIO FINAL
                    for p in self.nombres_pallets:
                        self.pallets_visitados[p] = False
                else:
                    step_reward += 50.0    # <-- INCENTIVO CONSTANTE FUERTE

        # Side Quest: Puerta
        handle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "handle_hinge")
        door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")

        if handle_id >= 0 and door_id >= 0:
            handle_qpos_adr = self.model.jnt_qposadr[handle_id]
            handle_angle = self.data.qpos[handle_qpos_adr]

            if 0.9 <= abs(handle_angle) <= 1.0:
                if not self.puerta_desbloqueada:
                    self.puerta_desbloqueada = True
                    self.model.jnt_range[door_id][0] = -1.5
                    self.model.jnt_range[door_id][1] =  1.5
                    step_reward += 50.0  

        # PENALIZACIONES DE TAGS
        if getattr(self, "_toco_zona_muerte", False):
            step_reward -= 10.0  

        if getattr(self, "_brazo_toco_fatal", False):
            step_reward -= 50.0  

        done = self._terminated()
        return obs, step_reward, done, {}

    def _reward(self, obs: Dict[str, np.ndarray]) -> float:
        base_xy = self.data.xpos[self.base_id, :2]
        dx = float(base_xy[0] - self._last_base_xy[0])     
        self._last_base_xy = base_xy.copy()

        if abs(dx) < 0.005:  
            self._stuck_counter += 1
            penalizacion_inactividad = 0.05
        else:
            self._stuck_counter = 0
            penalizacion_inactividad = 0.0

        min_lidar = float(obs["lidar"].min())              
        obstacle_penalty = max(0.0, 0.1 - min_lidar) * 5.0
        action_cost = 1e-3 * float(np.square(self.data.ctrl[self.act_ids]).mean())
        alive_bonus = 0.01
        
        # AUMENTADO: de 5.0 a 10.0 para motivar el avance agresivo en X
        return (10.0 * dx) + alive_bonus - obstacle_penalty - action_cost - penalizacion_inactividad

    def _terminated(self) -> bool:
        if self._step_counter >= self.max_steps:
            return True
            
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.2:
            return True
            
        if getattr(self, "_stuck_counter", 0) > 50:
            return True
            
        if getattr(self, "_toco_zona_muerte", False):
            return True
            
        if getattr(self, "_brazo_toco_fatal", False):
            return True
            
        return False

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        try:
            self.renderer.close()
        except Exception:
            pass


# ──────────────────────────────── Network ──────────────────────────────────
class ImageEncoder(nn.Module):
    """Nature-CNN-style encoder for stacked RGB cameras (9 channels)."""

    def __init__(self, in_channels: int, h: int, w: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32,         64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,         64, 3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, h, w)) \
                           .flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class StateEncoder(nn.Module):
    """MLP for the (lidar || joint_states) vector."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128,    out_dim), nn.Tanh(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvActorCritic(nn.Module):
    """Multi-modal actor-critic. Diagonal Gaussian with learnable log-std."""

    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 lidar_dim:   int,
                 joint_dim:   int,
                 act_dim:     int,
                 img_feat:    int = 256,
                 vec_feat:    int = 128,
                 hidden:      int = 256,
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
        img_z = self.img_enc(images)
        vec_z = self.vec_enc(torch.cat([lidar, joints], dim=-1))
        return self.trunk(torch.cat([img_z, vec_z], dim=-1))

    def forward(self, images, lidar, joints):
        z       = self._fuse(images, lidar, joints)
        mu      = torch.tanh(self.actor_mu(z))                # mean in [-1,1]
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
        return (
            raw.squeeze(0).cpu().numpy(),     # env clips & scales
            float(logp.item()),
            float(value.item()),
        )

    def evaluate(self, images, lidar, joints, actions):
        mu, std, value = self(images, lidar, joints)
        dist = Normal(mu, std)
        logp = dist.log_prob(actions).sum(dim=-1, keepdim=True)
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
        self.logps   = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.values  = np.zeros((self.capacity,), dtype=np.float32)
        self.dones   = np.zeros((self.capacity,), dtype=np.float32)

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
            next_value = last_value if t == self.capacity - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + gamma * next_value * next_nonterminal
                     - self.values[t])
            gae   = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + self.values
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return advantages, returns


# ──────────────────────────── PPO update step ──────────────────────────────
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
        for idx in BatchSampler(
            SubsetRandomSampler(range(buffer.capacity)),
            batch_size,
            drop_last=False,
        ):
            logp, value, entropy = policy.evaluate(
                images[idx], lidar[idx], joints[idx], actions[idx]
            )
            ratio = torch.exp(logp - old_log[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.smooth_l1_loss(value, ret[idx])
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            metrics["pi"]  = policy_loss.item()
            metrics["v"]   = value_loss.item()
            metrics["ent"] = entropy.item()
    return metrics


# ────────────────────────────── Training loop ──────────────────────────────
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
    """Build a wandb.Image laying the N cameras side-by-side."""
    n = len(camera_names)
    h, w = obs_images.shape[1], obs_images.shape[2]
    panel = np.zeros((h, w * n, 3), dtype=np.uint8)
    for k in range(n):
        cam = obs_images[k * 3:(k + 1) * 3]                    # (3, H, W)
        cam = (cam.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        panel[:, k * w:(k + 1) * w, :] = cam
    return wandb.Image(panel, caption=" | ".join(camera_names))


def train(num_iterations: int = 500,
          steps_per_iter:  int = 2048,
          ppo_epochs:      int = 10,
          batch_size:      int = 256,
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
          image_log_every: int   = 25):

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    env = AesirMuJoCoEnv(render=render)
    print(f"Action dim:  {env.act_len}")
    print(f"Image shape: {env.image_shape}")
    print(f"Lidar dim:   {env.num_lidar}")
    print(f"Joint dim:   {env.joint_len}")

    policy = ConvActorCritic(
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    buffer = RolloutBuffer(
        capacity=steps_per_iter,
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    )

    # ── wandb init ─────────────────────────────────────────────────────────
    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "num_iterations":  num_iterations,
                "steps_per_iter":  steps_per_iter,
                "ppo_epochs":      ppo_epochs,
                "batch_size":      batch_size,
                "gamma":           gamma,
                "gae_lambda":      gae_lambda,
                "clip_param":      clip_param,
                "vf_coef":         vf_coef,
                "ent_coef":        ent_coef,
                "lr":              lr,
                "image_shape":     list(env.image_shape),
                "lidar_dim":       env.num_lidar,
                "joint_dim":       env.joint_len,
                "act_dim":         env.act_len,
                "control_decimation": env.control_decimation,
                "episode_max_steps":  env.max_steps,
            },
        )
        wandb.watch(policy, log="gradients", log_freq=100)

    obs = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_history: List[float] = []
    best_avg = -1e9

    try:
        for it in range(num_iterations):
            t0 = time.time()
            for _ in range(steps_per_iter):
                obs_t = obs_to_tensor(obs)
                action, logp, value = policy.act(obs_t, device)
                next_obs, reward, done, _ = env.step(action)
                buffer.store(obs, action, logp, reward, value, done)
                obs = next_obs
                ep_reward += reward
                ep_len    += 1
                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20:
                        ep_history.pop(0)
                    ep_reward, ep_len = 0.0, 0
                    obs = env.reset()

            # bootstrap value for GAE on the last (possibly mid-episode) obs
            obs_t = obs_to_tensor(obs)
            with torch.no_grad():
                _, _, last_value = policy(
                    obs_t["images"].unsqueeze(0).to(device),
                    obs_t["lidar"].unsqueeze(0).to(device),
                    obs_t["joint_states"].unsqueeze(0).to(device),
                )
            advantages, returns = buffer.compute_gae(
                last_value=float(last_value.item()),
                gamma=gamma, gae_lambda=gae_lambda,
            )

            metrics = ppo_update(
                policy, optimizer, buffer, advantages, returns,
                ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device,
            )

            avg_ep_r = float(np.mean(ep_history)) if ep_history else float("nan")
            dt = time.time() - t0
            print(f"[Iter {it:4d}] avg_ep_r={avg_ep_r:8.2f}  "
                  f"pi_loss={metrics['pi']:+.4f}  "
                  f"v_loss={metrics['v']:.4f}  "
                  f"ent={metrics['ent']:.3f}  "
                  f"({dt:.1f}s)")

            # ── wandb scalar + image logging ───────────────────────────────
            if use_wandb:
                log_dict = {
                    "iter":         it,
                    "global_step":  (it + 1) * steps_per_iter,
                    "avg_ep_r":     avg_ep_r,
                    "policy_loss":  metrics["pi"],
                    "value_loss":   metrics["v"],
                    "entropy":      metrics["ent"],
                    "mean_log_std": policy.log_std.detach().mean().item(),
                    "iter_time_s":  dt,
                    "steps_per_s":  steps_per_iter / max(dt, 1e-6),
                }
                if (it % image_log_every) == 0:
                    log_dict["camera_views"] = make_camera_panel(
                        obs["images"], env.camera_names
                    )
                wandb.log(log_dict, step=(it + 1) * steps_per_iter)

            # ── checkpointing ──────────────────────────────────────────────
            if (it + 1) % save_every == 0:
                ckpt_path = CHECKPOINT_DIR / f"ppo_conv_iter{it+1:05d}.pt"
                save_checkpoint(ckpt_path, policy, optimizer, it + 1, avg_ep_r)
                print(f"  ↳ checkpoint saved to {ckpt_path}")
                if use_wandb:
                    wandb.save(str(ckpt_path), base_path=str(CHECKPOINT_DIR))

            if ep_history and avg_ep_r > best_avg:
                best_avg = avg_ep_r
                best_path = CHECKPOINT_DIR / "ppo_conv_best.pt"
                save_checkpoint(best_path, policy, optimizer, it + 1, avg_ep_r)
                print(f"  ↳ NEW BEST ({avg_ep_r:.2f}) saved to {best_path}")
                if use_wandb:
                    wandb.run.summary["best_avg_ep_r"] = best_avg
                    wandb.save(str(best_path), base_path=str(CHECKPOINT_DIR))
    finally:
        env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train(render=True)
