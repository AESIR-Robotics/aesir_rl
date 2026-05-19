"""
Multi-Agent PPO (IPPO) for the Aesir rescue robot.
Both agents share the same cooperative reward signal (IPPO).
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

# Triton (PyTorch JIT compiler) segfaults when CUDA is absent.
import torch._dynamo
torch._dynamo.disable()


# ──────────────────────────── Config ─────────────────────────────────────────

XML_PATH           = str(Path(__file__).parent / "assets/aesir_complete.xml")
CAMERA_NAMES       = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W = 84, 84
NUM_LIDAR_RAYS     = 7
LIDAR_MAX_RANGE    = 15.0
LIDAR_SPIN_VEL     = 20.0

# ── Agent A: actuators written to data.ctrl ───────────────────────────────────
ACTUATOR_NAMES_A = [
    "pos_joint_1", "pos_joint_2", "pos_joint_3",
    "pos_joint_4", "pos_joint_5", "pos_joint_6",
    "pos_left_finger", "pos_right_finger",
]

# ── Agent B: actuators written to data.ctrl ───────────────────────────────────
# Order matters: adapt_tracks() output must match this order exactly.
ACTUATOR_NAMES_B = [
    "vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3",   # [0:3]  left track
    "vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3",   # [3:6]  right track
    "pos_flipper_1", "pos_flipper_2",                     # [6:8]  left flippers
    "pos_flipper_3", "pos_flipper_4",                     # [8:10] right flippers
    "vel_flip1_back", "vel_flip1_front",                  # [10:12] flipper-1 wheels (left)
    "vel_flip2_back", "vel_flip2_front",                  # [12:14] flipper-2 wheels (left)
    "vel_flip3_back", "vel_flip3_front",                  # [14:16] flipper-3 wheels (right)
    "vel_flip4_back", "vel_flip4_front",                  # [16:18] flipper-4 wheels (right)
]

# ── Joint names each agent observes (qpos + qvel) ────────────────────────────
OBS_JOINT_NAMES_A = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
    "left_finger_joint", "right_finger_joint",
]  # → 16-dim vector

OBS_JOINT_NAMES_B = [
    "flipper_joint_1", "flipper_joint_2", "flipper_joint_3", "flipper_joint_4",
    "drive_l_1", "drive_l_2", "drive_l_3",
    "drive_r_1", "drive_r_2", "drive_r_3",
]  # → 20-dim vector

# ── Semantic action dimensions (network outputs) ──────────────────────────────
ACT_DIM_A = 7   # [j1..j6, gripper_open]
ACT_DIM_B = 6   # [v_linear, v_angular, flip1..flip4]

# ── Adapter constants ─────────────────────────────────────────────────────────
ARM_JOINT_RANGE = np.pi    # joints [-1,1] → [-π, π] rad
GRIPPER_MAX     = 0.03     # max finger displacement [m]
MAX_TRACK_VEL   = 20.0     # ctrlrange of vel_drive actuators [rad/s]
MAX_FLIPPER_VEL = 1.0      # ctrlrange of vel_flip wheel actuators [rad/s]
# At v_ang=±1, each track gets ±ANGULAR_GAIN * MAX_TRACK_VEL offset.
# 0.5 → turns at half the linear speed; increase for sharper turns.
ANGULAR_GAIN    = 0.5

CONTROL_DECIMATION = 10
EPISODE_MAX_STEPS  = 1000

CHECKPOINT_DIR = Path("model_robot/outputs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── Adapters ───────────────────────────────────────

def adapt_arm(raw: np.ndarray) -> np.ndarray:
    """
    7-dim network output → 8 ctrl values for Agent A.

    raw : [j1..j6 ∈ [-1,1],  gripper_open ∈ [-1,1]]
    out : [pos_joint_1..6 ∈ [-π,π],  pos_left_finger, pos_right_finger ∈ [0, 0.03]]

    gripper_open=-1 → fully closed (0 m), gripper_open=+1 → fully open (0.03 m).
    Both fingers receive the same command (parallel-jaw gripper).
    """
    joints = np.clip(raw[:6], -1.0, 1.0).astype(np.float64) * ARM_JOINT_RANGE
    g      = (float(np.clip(raw[6], -1.0, 1.0)) + 1.0) * 0.5 * GRIPPER_MAX
    return np.concatenate([joints, [g, g]]).astype(np.float32)


def adapt_tracks(raw: np.ndarray) -> np.ndarray:
    """
    6-dim network output → 18 ctrl values for Agent B.

    raw : [v_lin ∈ [-1,1], v_ang ∈ [-1,1], flip1..flip4 ∈ [-1,1]]
    out : [vel_drive_l_{1..3}, vel_drive_r_{1..3},
           pos_flipper_{1..4},
           vel_flip1_back, vel_flip1_front, ..., vel_flip4_back, vel_flip4_front]

    Differential drive:
      v_left  = (v_lin - v_ang * ANGULAR_GAIN) * MAX_TRACK_VEL
      v_right = (v_lin + v_ang * ANGULAR_GAIN) * MAX_TRACK_VEL
    The 3 wheels on each side are treated as a single stack (same velocity).
    Flipper wheels mirror the track velocity of their side, scaled to [-1, 1].
    """
    v_lin = float(np.clip(raw[0], -1.0, 1.0))
    v_ang = float(np.clip(raw[1], -1.0, 1.0))

    v_left  = np.clip((v_lin - v_ang * ANGULAR_GAIN) * MAX_TRACK_VEL,
                      -MAX_TRACK_VEL, MAX_TRACK_VEL)
    v_right = np.clip((v_lin + v_ang * ANGULAR_GAIN) * MAX_TRACK_VEL,
                      -MAX_TRACK_VEL, MAX_TRACK_VEL)

    track_l  = np.full(3, v_left,  dtype=np.float32)
    track_r  = np.full(3, v_right, dtype=np.float32)
    flippers = (np.clip(raw[2:6], -1.0, 1.0) * ARM_JOINT_RANGE).astype(np.float32)

    # Flipper wheels follow their side's track; scale [−MAX,MAX] → [−1,1]
    vfl = float(np.clip(v_left  / MAX_TRACK_VEL, -1.0, 1.0)) * MAX_FLIPPER_VEL
    vfr = float(np.clip(v_right / MAX_TRACK_VEL, -1.0, 1.0)) * MAX_FLIPPER_VEL
    flip_wheels = np.array(
        [vfl, vfl,   # flip1 back + front  (left)
         vfl, vfl,   # flip2 back + front  (left)
         vfr, vfr,   # flip3 back + front  (right)
         vfr, vfr],  # flip4 back + front  (right)
        dtype=np.float32,
    )

    return np.concatenate([track_l, track_r, flippers, flip_wheels])


# ──────────────────────────────── Env ────────────────────────────────────────

class AesirMultiAgentEnv:
    """
    Multi-agent MuJoCo env
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
        self.renderer = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)
        self.camera_names     = list(camera_names)
        self.num_cameras      = len(self.camera_names)
        self.num_lidar        = num_lidar_rays
        self.lidar_max        = lidar_max_range
        self.control_decimation = control_decimation
        self.max_steps        = max_steps

        # ── actuator indices ──────────────────────────────────────────────
        def _resolve_actuators(names):
            ids = np.array([
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in names
            ], dtype=np.int32)
            missing = [n for n, i in zip(names, ids) if i < 0]
            if missing:
                raise ValueError(f"Actuators not found in model: {missing}")
            return ids

        self.act_ids_a   = _resolve_actuators(ACTUATOR_NAMES_A)
        self.act_ids_b   = _resolve_actuators(ACTUATOR_NAMES_B)
        self.all_act_ids = np.concatenate([self.act_ids_a, self.act_ids_b])

        # ── observation joint addresses ───────────────────────────────────
        def _joint_addrs(names):
            qpos_adr, qvel_adr = [], []
            for n in names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if jid < 0:
                    raise ValueError(f"Joint not found: {n}")
                qpos_adr.append(int(self.model.jnt_qposadr[jid]))
                qvel_adr.append(int(self.model.jnt_dofadr[jid]))
            return np.array(qpos_adr, dtype=np.int32), np.array(qvel_adr, dtype=np.int32)

        self.qpos_a, self.qvel_a = _joint_addrs(OBS_JOINT_NAMES_A)
        self.qpos_b, self.qvel_b = _joint_addrs(OBS_JOINT_NAMES_B)

        # ── lidar spin (held constant by env) ─────────────────────────────
        self.lidar_spin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_lidar_spin"
        )

        # ── lidar sensor addresses ────────────────────────────────────────
        self.lidar_sensor_adr = []
        for i in range(self.num_lidar):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} not found")
            self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (self.num_lidar,)
        self.joint_dim_a = 2 * len(OBS_JOINT_NAMES_A)   # 16
        self.joint_dim_b = 2 * len(OBS_JOINT_NAMES_B)   # 20

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        self._step_counter  = 0
        self._stuck_counter = 0
        self._last_base_xy  = np.zeros(2)

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 4.0
            self.viewer.cam.elevation = -20

        self.piezas_robot = {
            "base_link", "tracked_1", "tracked_2",
            "flipper_1_1", "flipper_2_1", "flipper_3_1", "flipper_4_1",
        }
        self.piezas_brazo = {
            "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
            "logitech_gripper_assembly", "left_finger_link", "right_finger_link",
        }
        self.nombres_pallets = [f"fatal_pallet {i}" for i in range(1, 19)]
        self._reset_estado_misiones()

    # ── helpers ───────────────────────────────────────────────────────────

    def _read_cameras(self) -> np.ndarray:
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            frames.append(self.renderer.render().astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        lidar = np.empty(self.num_lidar, dtype=np.float32)
        for i, adr in enumerate(self.lidar_sensor_adr):
            d = float(self.data.sensordata[adr])
            lidar[i] = (d if 0.0 < d < self.lidar_max else self.lidar_max) / self.lidar_max
        return lidar

    def _read_joints_a(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self.qpos_a], self.data.qvel[self.qvel_a],
        ]).astype(np.float32)

    def _read_joints_b(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self.qpos_b], self.data.qvel[self.qvel_b],
        ]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        images = self._read_cameras()
        lidar  = self._read_lidar()
        return {
            "images":   images,
            "lidar":    lidar,
            "joints_a": self._read_joints_a(),
            "joints_b": self._read_joints_b(),
        }

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

            is_rob1 = (g1 in self.piezas_robot) or (b1 in self.piezas_robot)
            is_rob2 = (g2 in self.piezas_robot) or (b2 in self.piezas_robot)
            is_arm1 = (g1 in self.piezas_brazo) or (b1 in self.piezas_brazo)
            is_arm2 = (g2 in self.piezas_brazo) or (b2 in self.piezas_brazo)

            if "muerte_" in g1 or "muerte_" in g2:
                if is_rob1 or is_rob2 or is_arm1 or is_arm2:
                    self._toco_zona_muerte = True

            if "fatal_" in g1 or "fatal_" in g2:
                if ("fatal_" in g1 and is_arm2) or ("fatal_" in g2 and is_arm1):
                    self._brazo_toco_fatal = True

            if is_rob1 and g2:
                objetos_tocados.add(g2)
            elif is_rob2 and g1:
                objetos_tocados.add(g1)

        return objetos_tocados

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = -1.5
        self.data.qpos[1] =  3.5
        self.data.qpos[2] =  0.2

        arm_rest = {
            "joint_1": -0.314, "joint_2": -3.14, "joint_3": 3.14,
            "joint_4":  0.0,   "joint_5":  0.0,  "joint_6": 0.0,
        }
        for name, angle in arm_rest.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angle

        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

        door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        if door_id >= 0:
            self.model.jnt_range[door_id][:] = [0.0, 0.0]

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

    def step(self,
             action_a: np.ndarray,
             action_b: np.ndarray,
             ) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """
        action_a : 7-dim raw from Agent A  [j1..j6, gripper_open]
        action_b : 6-dim raw from Agent B  [v_linear, v_angular, flip1..flip4]
        """
        self.data.ctrl[self.act_ids_a] = adapt_arm(action_a)
        self.data.ctrl[self.act_ids_b] = adapt_tracks(action_b)
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

        self._step_counter += 1
        obs         = self._observation()
        step_reward = self._reward(obs)
        objetos_tocados = self._obtener_contactos_del_robot()

        for pallet in self.nombres_pallets:
            if pallet in objetos_tocados and not self.pallets_visitados[pallet]:
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
            angle = self.data.qpos[self.model.jnt_qposadr[handle_id]]
            if 0.9 <= abs(angle) <= 1.0 and not self.puerta_desbloqueada:
                self.puerta_desbloqueada = True
                self.model.jnt_range[door_id][:] = [-1.5, 1.5]
                step_reward += 50.0

        if self._toco_zona_muerte:
            step_reward -= 10.0
        if self._brazo_toco_fatal:
            step_reward -= 50.0

        return obs, step_reward, self._terminated(), {}

    def _reward(self, obs: Dict[str, np.ndarray]) -> float:
        base_xy = self.data.xpos[self.base_id, :2]
        dx = float(base_xy[0] - self._last_base_xy[0])
        self._last_base_xy = base_xy.copy()

        if abs(dx) < 0.005:
            self._stuck_counter += 1
            penalty_stuck = 0.05
        else:
            self._stuck_counter = 0
            penalty_stuck = 0.0

        min_lidar    = float(obs["lidar"].min())
        obstacle_pen = max(0.0, 0.1 - min_lidar) * 5.0
        action_cost  = 1e-3 * float(np.square(self.data.ctrl[self.all_act_ids]).mean())
        return 10.0 * dx + 0.01 - obstacle_pen - action_cost - penalty_stuck

    def _terminated(self) -> bool:
        if self._step_counter >= self.max_steps:
            return True
        if float(self.data.xmat[self.base_id].reshape(3, 3)[2, 2]) < 0.2:
            return True
        if self._stuck_counter > 50:
            return True
        if self._toco_zona_muerte or self._brazo_toco_fatal:
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


# ──────────────────────────────── Network ────────────────────────────────────
# ConvActorCritic is identical to ppo_conv_train.py but act() receives
# a `joint_key` so each agent pulls its own joint observation.

class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32,          64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64,          64, 3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, h, w)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class StateEncoder(nn.Module):
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
    """Multi-modal actor-critic. Instantiated once per agent with different dims."""

    def __init__(self,
                 image_shape:  Tuple[int, int, int],
                 lidar_dim:    int,
                 joint_dim:    int,
                 act_dim:      int,
                 img_feat:     int   = 256,
                 vec_feat:     int   = 128,
                 hidden:       int   = 256,
                 log_std_init: float = -0.5):
        super().__init__()
        c, h, w = image_shape
        self.img_enc = ImageEncoder(c, h, w, out_dim=img_feat)
        self.vec_enc = StateEncoder(lidar_dim + joint_dim, out_dim=vec_feat)
        fused_dim    = img_feat + vec_feat
        self.trunk   = nn.Sequential(
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
            self.vec_enc(torch.cat([lidar, joints], dim=-1)),
        ], dim=-1))

    def forward(self, images, lidar, joints):
        z       = self._fuse(images, lidar, joints)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], joint_key: str, device):
        images = obs["images"].unsqueeze(0).to(device)
        lidar  = obs["lidar"].unsqueeze(0).to(device)
        joints = obs[joint_key].unsqueeze(0).to(device)
        mu, std, value = self(images, lidar, joints)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())

    def evaluate(self, images, lidar, joints, actions):
        mu, std, value = self(images, lidar, joints)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return logp, value, entropy


# ──────────────────────────── Rollout buffer ─────────────────────────────────

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
        self.logps   = np.zeros((self.capacity,),                dtype=np.float32)
        self.rewards = np.zeros((self.capacity,),                dtype=np.float32)
        self.values  = np.zeros((self.capacity,),                dtype=np.float32)
        self.dones   = np.zeros((self.capacity,),                dtype=np.float32)

    def store(self, obs: Dict[str, np.ndarray], joint_key: str,
              action, logp, reward, value, done) -> bool:
        i = self.idx
        self.images[i]  = obs["images"]
        self.lidars[i]  = obs["lidar"]
        self.joints[i]  = obs[joint_key]
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
            nv  = last_value if t == self.capacity - 1 else self.values[t + 1]
            nnt = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nv * nnt - self.values[t]
            gae   = delta + gamma * gae_lambda * nnt * gae
            advantages[t] = gae
        returns  = advantages + self.values
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return adv_norm, returns


# ──────────────────────────── PPO update ─────────────────────────────────────

def ppo_update(policy, optimizer, buffer: RolloutBuffer,
               advantages, returns,
               ppo_epochs, batch_size, clip_param,
               vf_coef, ent_coef, device):
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
            SubsetRandomSampler(range(buffer.capacity)), batch_size, drop_last=False
        ):
            logp, value, entropy = policy.evaluate(
                images[idx], lidar[idx], joints[idx], actions[idx]
            )
            ratio  = torch.exp(logp - old_log[idx])
            surr1  = ratio * adv[idx]
            surr2  = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[idx]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.smooth_l1_loss(value, ret[idx])
            loss   = p_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            metrics["pi"]  = p_loss.item()
            metrics["v"]   = v_loss.item()
            metrics["ent"] = entropy.item()
    return metrics


# ────────────────────────────── Utilities ────────────────────────────────────

def obs_to_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float() for k, v in obs.items()}


def save_checkpoint(path: Path, policy_a, opt_a, policy_b, opt_b,
                    iter_idx: int, avg_ep_r: float):
    torch.save({
        "iter":        iter_idx,
        "policy_a":    policy_a.state_dict(),
        "optimizer_a": opt_a.state_dict(),
        "policy_b":    policy_b.state_dict(),
        "optimizer_b": opt_b.state_dict(),
        "avg_ep_r":    avg_ep_r,
    }, path)


def make_camera_panel(obs_images: np.ndarray, camera_names: List[str]):
    n = len(camera_names)
    h, w = obs_images.shape[1], obs_images.shape[2]
    panel = np.zeros((h, w * n, 3), dtype=np.uint8)
    for k in range(n):
        cam = obs_images[k * 3:(k + 1) * 3]
        cam = (cam.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        panel[:, k * w:(k + 1) * w] = cam
    return wandb.Image(panel, caption=" | ".join(camera_names))


# ────────────────────────────── Training loop ────────────────────────────────

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
          wandb_project:   str   = "AIDL-PPO-AESIR-MULTIAGENT",
          wandb_run_name:  str   = None,
          image_log_every: int   = 25):

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))
    print(f"Device: {device}")

    env = AesirMultiAgentEnv(render=render)
    print(f"Image shape : {env.image_shape}")
    print(f"Lidar dim   : {env.num_lidar}")
    print(f"Joint dim A : {env.joint_dim_a}  (arm + gripper qpos+qvel)")
    print(f"Joint dim B : {env.joint_dim_b}  (tracks + flippers qpos+qvel)")
    print(f"Act dim A   : {ACT_DIM_A}  → adapt_arm  → {len(ACTUATOR_NAMES_A)} ctrl values")
    print(f"Act dim B   : {ACT_DIM_B}  → adapt_tracks → {len(ACTUATOR_NAMES_B)} ctrl values")

    # ── policies (one per agent) ──────────────────────────────────────────
    policy_a = ConvActorCritic(
        image_shape=env.image_shape, lidar_dim=env.num_lidar,
        joint_dim=env.joint_dim_a,  act_dim=ACT_DIM_A,
    ).to(device)

    policy_b = ConvActorCritic(
        image_shape=env.image_shape, lidar_dim=env.num_lidar,
        joint_dim=env.joint_dim_b,  act_dim=ACT_DIM_B,
    ).to(device)

    opt_a = torch.optim.Adam(policy_a.parameters(), lr=lr)
    opt_b = torch.optim.Adam(policy_b.parameters(), lr=lr)

    # ── rollout buffers (separate per agent) ─────────────────────────────
    buf_a = RolloutBuffer(
        capacity=steps_per_iter, image_shape=env.image_shape,
        lidar_dim=env.num_lidar, joint_dim=env.joint_dim_a, act_dim=ACT_DIM_A,
    )
    buf_b = RolloutBuffer(
        capacity=steps_per_iter, image_shape=env.image_shape,
        lidar_dim=env.num_lidar, joint_dim=env.joint_dim_b, act_dim=ACT_DIM_B,
    )

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=wandb_project, name=wandb_run_name,
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
                "act_dim_a":       ACT_DIM_A,
                "act_dim_b":       ACT_DIM_B,
                "joint_dim_a":     env.joint_dim_a,
                "joint_dim_b":     env.joint_dim_b,
                "angular_gain":    ANGULAR_GAIN,
            },
        )
        wandb.watch(policy_a, log="gradients", log_freq=100)
        wandb.watch(policy_b, log="gradients", log_freq=100)

    obs = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_history: List[float] = []
    best_avg = -1e9

    try:
        for it in range(num_iterations):
            t0 = time.time()

            for _ in range(steps_per_iter):
                obs_t = obs_to_tensor(obs)

                act_a, logp_a, val_a = policy_a.act(obs_t, "joints_a", device)
                act_b, logp_b, val_b = policy_b.act(obs_t, "joints_b", device)

                next_obs, reward, done, _ = env.step(act_a, act_b)

                buf_a.store(obs, "joints_a", act_a, logp_a, reward, val_a, done)
                buf_b.store(obs, "joints_b", act_b, logp_b, reward, val_b, done)

                obs        = next_obs
                ep_reward += reward
                ep_len    += 1

                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20:
                        ep_history.pop(0)
                    ep_reward, ep_len = 0.0, 0
                    obs = env.reset()

            # ── bootstrap last value for GAE ──────────────────────────────
            obs_t = obs_to_tensor(obs)
            with torch.no_grad():
                _, _, lv_a = policy_a(
                    obs_t["images"].unsqueeze(0).to(device),
                    obs_t["lidar"].unsqueeze(0).to(device),
                    obs_t["joints_a"].unsqueeze(0).to(device),
                )
                _, _, lv_b = policy_b(
                    obs_t["images"].unsqueeze(0).to(device),
                    obs_t["lidar"].unsqueeze(0).to(device),
                    obs_t["joints_b"].unsqueeze(0).to(device),
                )

            adv_a, ret_a = buf_a.compute_gae(float(lv_a.item()), gamma, gae_lambda)
            adv_b, ret_b = buf_b.compute_gae(float(lv_b.item()), gamma, gae_lambda)

            # ── independent PPO updates ───────────────────────────────────
            m_a = ppo_update(policy_a, opt_a, buf_a, adv_a, ret_a,
                             ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)
            m_b = ppo_update(policy_b, opt_b, buf_b, adv_b, ret_b,
                             ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)

            avg_ep_r = float(np.mean(ep_history)) if ep_history else float("nan")
            dt       = time.time() - t0
            print(
                f"[Iter {it:4d}] avg_r={avg_ep_r:8.2f} | "
                f"A: pi={m_a['pi']:+.4f} v={m_a['v']:.4f} ent={m_a['ent']:.3f} | "
                f"B: pi={m_b['pi']:+.4f} v={m_b['v']:.4f} ent={m_b['ent']:.3f} | "
                f"{dt:.1f}s"
            )

            if use_wandb:
                log_dict = {
                    "iter": it, "global_step": (it + 1) * steps_per_iter,
                    "avg_ep_r":             avg_ep_r,
                    "agent_a/policy_loss":  m_a["pi"],
                    "agent_a/value_loss":   m_a["v"],
                    "agent_a/entropy":      m_a["ent"],
                    "agent_a/mean_log_std": policy_a.log_std.detach().mean().item(),
                    "agent_b/policy_loss":  m_b["pi"],
                    "agent_b/value_loss":   m_b["v"],
                    "agent_b/entropy":      m_b["ent"],
                    "agent_b/mean_log_std": policy_b.log_std.detach().mean().item(),
                    "iter_time_s":          dt,
                    "steps_per_s":          steps_per_iter / max(dt, 1e-6),
                }
                if it % image_log_every == 0:
                    log_dict["camera_views"] = make_camera_panel(
                        obs["images"], env.camera_names
                    )
                wandb.log(log_dict, step=(it + 1) * steps_per_iter)

            if (it + 1) % save_every == 0:
                ckpt = CHECKPOINT_DIR / f"ppo_ma_iter{it+1:05d}.pt"
                save_checkpoint(ckpt, policy_a, opt_a, policy_b, opt_b, it + 1, avg_ep_r)
                print(f"  ↳ checkpoint → {ckpt}")
                if use_wandb:
                    wandb.save(str(ckpt), base_path=str(CHECKPOINT_DIR))

            if ep_history and avg_ep_r > best_avg:
                best_avg  = avg_ep_r
                best_path = CHECKPOINT_DIR / "ppo_ma_best.pt"
                save_checkpoint(best_path, policy_a, opt_a, policy_b, opt_b, it + 1, avg_ep_r)
                print(f"  ↳ NEW BEST ({avg_ep_r:.2f}) → {best_path}")
                if use_wandb:
                    wandb.run.summary["best_avg_ep_r"] = best_avg
                    wandb.save(str(best_path), base_path=str(CHECKPOINT_DIR))

    finally:
        env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train(render=True)
