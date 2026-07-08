"""
ppo_pallets_train.py — Navegacion optima por corredor de pallets.

Objetivo: el robot navega de SPAWN_XYZ hasta GOAL_XY recorriendo
SOLO los pallets (superficie teal), encontrando el camino mas corto.

No necesita tocar todos los pallets — solo llegar a la meta.

Recompensa:
  + Progreso hacia la meta (distancia geodesica a lo largo del path)
  + Bonus grande al llegar
  - Penalizacion por quedar atascado
  - Penalizacion por estar cerca de obstaculos (lidar)
  - Terminacion inmediata si el robot se cae o voltea
"""
from __future__ import annotations

import os, time
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
    import wandb; _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ── Ruta XML ──────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_XML     = os.path.join(_HERE, "../models/aesir_pallets.xml")
XML_PATH = _XML if os.path.exists(_XML) else os.path.join(_HERE, "../aesir_robot_description/launch/aesir_complete.xml")

# ── Sensores y control ────────────────────────────────────────────────────────
CAMERA_NAMES       = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W = 84, 84
NUM_LIDAR_RAYS     = 7
LIDAR_MAX_RANGE    = 15.0
LIDAR_SPIN_VEL     = 20.0
TRACK_HALF_WIDTH   = 0.21
WHEEL_RADIUS       = 0.05
MAX_WHEEL_VEL      = 20.0
MAX_LINEAR_VEL     = 1.5
MAX_ANGULAR_VEL    = 2.0
MAX_JOINT_VEL      = 1.0

DRIVE_LEFT  = ["vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3"]
DRIVE_RIGHT = ["vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3"]
FLIPPERS    = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
FLIP_WHEELS = {
    "pos_flipper_1": ["vel_flip1_back", "vel_flip1_front"],
    "pos_flipper_2": ["vel_flip2_back", "vel_flip2_front"],
    "pos_flipper_3": ["vel_flip3_back", "vel_flip3_front"],
    "pos_flipper_4": ["vel_flip4_back", "vel_flip4_front"],
}
ARM_JOINTS  = ["pos_joint_1","pos_joint_2","pos_joint_3",
               "pos_joint_4","pos_joint_5","pos_joint_6"]
FINGER_L    = "pos_left_finger"
FINGER_R    = "pos_right_finger"
LIDAR_SPIN  = "vel_lidar_spin"
OBS_ACTUATORS = DRIVE_LEFT + DRIVE_RIGHT + FLIPPERS + ARM_JOINTS + [FINGER_L, FINGER_R]

ARM_REST = {"joint_1":-0.314,"joint_2":-3.14,"joint_3":3.14,
            "joint_4":-1.35, "joint_5":-1.54,"joint_6":1.54}
ARM_REST_ANGLES = ARM_REST

CONTROL_DECIMATION = 10
EPISODE_MAX_STEPS  = 2000    # el robot debe ser rapido
STUCK_MAX_STEPS    = 80
CHECKPOINT_DIR     = Path("./checkpoints_nav")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Navegacion ────────────────────────────────────────────────────────────────
_NAV_JSON  = os.path.join(_HERE, "obstacles.json")
SPAWN_XYZ  = (-1.5, 3.5, 0.25)
GOAL_XY    = (7.8,  1.01)        # P11 — meta final
GOAL_REACH = 0.80                # metros para considerar la meta alcanzada

def _build_path() -> List[Tuple[float, float]]:
    """Genera el camino mas corto por el corredor de pallets."""
    from global_navigator import plan_route
    if not os.path.exists(_NAV_JSON):
        print(f"[AVISO] {_NAV_JSON} no encontrado. Usa waypoints de emergencia.")
        return [SPAWN_XYZ[:2], GOAL_XY]
    try:
        wps = plan_route(_NAV_JSON, SPAWN_XYZ[:2], GOAL_XY)
        print(f"  PATH: {len(wps)} waypoints | "
              f"{wps[0]} -> ... -> {wps[-1]}")
        return wps
    except Exception as e:
        print(f"[AVISO] plan_route_fastest fallo: {e}. Waypoints de emergencia.")
        return [SPAWN_XYZ[:2], GOAL_XY]

PATH_WAYPOINTS: List[Tuple[float, float]] = _build_path()
VIA_POINTS = PATH_WAYPOINTS[1:-1]

# ── Recompensas ───────────────────────────────────────────────────────────────
R_PROGRESS      =  8.0   # por metro avanzado hacia la meta
R_COMPLETE      = 500.0  # bonus al llegar a GOAL
R_ALIVE         =  0.01  # por sobrevivir
R_SMOOTH        =  0.003 # por moverse alineado hacia la meta

P_STUCK         = -0.05
P_LIDAR         = -5.0
P_FALL          = -50.0  # penalizacion terminal por caida
P_SHAKE         = -0.02  # penalizacion por inestabilidad angular
LIDAR_DANGER    =  0.12


# ══════════════════════════════════════════════════════════════════════════════
#  NavigationMonitor — rastrea el progreso hacia la meta
# ══════════════════════════════════════════════════════════════════════════════
class NavigationMonitor:
    """
    Rastrea el progreso del robot a lo largo del camino A*.
    La recompensa es proporcional a cuanto mas cerca esta de la meta.
    """
    def __init__(self):
        self._wps   = [np.array(w) for w in PATH_WAYPOINTS]
        self._goal  = np.array(GOAL_XY)
        self.reset()

    def reset(self, start_xy: np.ndarray = None):
        xy = start_xy if start_xy is not None else np.array(SPAWN_XYZ[:2])
        # Encontrar el waypoint mas cercano al inicio
        dists = [np.linalg.norm(xy - w) for w in self._wps]
        self._wp_idx  = int(np.argmin(dists))
        self._prev_xy = xy.copy()
        self.completed = False

    @property
    def current_target(self) -> np.ndarray:
        """Siguiente waypoint a alcanzar."""
        idx = min(self._wp_idx, len(self._wps) - 1)
        return self._wps[idx]

    @property
    def dist_to_goal(self) -> float:
        return float(np.linalg.norm(self._prev_xy - self._goal))

    def update(self, xy: np.ndarray) -> Tuple[float, bool]:
        """
        Actualiza el monitor y devuelve (reward_step, completed).
        reward_step = R_PROGRESS * metros_avanzados_hacia_meta
        """
        # Progreso: cuanto mas cerca estamos de la meta
        prev_dist = np.linalg.norm(self._prev_xy - self._goal)
        curr_dist = np.linalg.norm(xy           - self._goal)
        progress  = prev_dist - curr_dist   # positivo = nos acercamos
        reward    = R_PROGRESS * progress
        self._prev_xy = xy.copy()

        # Avanzar waypoint si llegamos cerca
        while self._wp_idx < len(self._wps) - 1:
            if np.linalg.norm(xy - self._wps[self._wp_idx]) < GOAL_REACH:
                self._wp_idx += 1
            else:
                break

        # Detectar llegada a la meta
        if curr_dist < GOAL_REACH:
            self.completed = True
            reward += R_COMPLETE

        return reward, self.completed


# ══════════════════════════════════════════════════════════════════════════════
#  AesirNavEnv
# ══════════════════════════════════════════════════════════════════════════════
class AesirNavEnv:
    """
    Entorno de navegacion: el robot va de SPAWN a GOAL por los pallets.
    No necesita tocar todos los pallets, solo llegar a la meta.
    """

    def __init__(self, xml_path=XML_PATH,
                 camera_names=CAMERA_NAMES,
                 image_hw=(CAMERA_H, CAMERA_W),
                 num_lidar=NUM_LIDAR_RAYS,
                 control_dec=CONTROL_DECIMATION,
                 max_steps=EPISODE_MAX_STEPS,
                 render=False):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.image_h, self.image_w = image_hw
        self.renderer     = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)
        self.num_lidar    = num_lidar

        # Actuadores
        self.ids_drive_l  = [self._aid(n) for n in DRIVE_LEFT]
        self.ids_drive_r  = [self._aid(n) for n in DRIVE_RIGHT]
        self.ids_flippers = [self._aid(n) for n in FLIPPERS]
        self.ids_flip_wh  = {self._aid(fn): [self._aid(w) for w in wns]
                              for fn, wns in FLIP_WHEELS.items()}
        self.ids_arm      = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l    = self._aid(FINGER_L)
        self.id_fing_r    = self._aid(FINGER_R)
        self.lidar_spin_id = self._aid(LIDAR_SPIN)

        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)
        self.joint_len    = 2 * len(self._obs_act_ids)

        self.lidar_sensor_adr = []
        for i in range(num_lidar):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid >= 0:
                self.lidar_sensor_adr.append(int(self.model.sensor_adr[sid]))

        self.base_id = max(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "footprint_link"), 1)

        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.act_len     = 14
        self.control_dec = control_dec
        self._dt         = self.model.opt.timestep * control_dec
        self.max_steps   = max_steps
        self._joint_pos  = np.zeros(6, dtype=np.float64)

        self._nav     = NavigationMonitor()
        self._step    = 0
        self._stuck   = 0
        self._last_xy = np.array(SPAWN_XYZ[:2])
        self._fell    = False

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 6.0
            self.viewer.cam.elevation = -25

    def _aid(self, n):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def _read_cameras(self):
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = np.flip(self.renderer.render(), axis=(0, 1))
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self):
        lidar = np.ones(self.num_lidar, dtype=np.float32)
        for i, adr in enumerate(self.lidar_sensor_adr):
            d = float(self.data.sensordata[adr])
            if 0 < d < LIDAR_MAX_RANGE:
                lidar[i] = d / LIDAR_MAX_RANGE
        return lidar

    def _read_joints(self):
        return np.concatenate([
            self.data.qpos[self._qpos_adr],
            self.data.qvel[self._qvel_adr],
        ]).astype(np.float32)

    def _obs(self):
        return {"images": self._read_cameras(),
                "lidar":  self._read_lidar(),
                "joint_states": self._read_joints()}

    def _apply_action(self, action):
        a     = np.clip(action, -1.0, 1.0)
        v_lin = float(a[0]) * MAX_LINEAR_VEL
        omega = float(a[1]) * MAX_ANGULAR_VEL
        vl = float(np.clip((v_lin - omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
                            -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
        vr = float(np.clip((v_lin + omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
                            -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
        for i in self.ids_drive_l: self.data.ctrl[i] = vl
        for i in self.ids_drive_r: self.data.ctrl[i] = vr
        for k, fid in enumerate(self.ids_flippers):
            self.data.ctrl[fid] = float(np.clip(a[2+k] * 3.1416, -3.1416, 3.1416))
            wv = vl if k in (0, 2) else vr
            for wid in self.ids_flip_wh.get(fid, []):
                self.data.ctrl[wid] = float(np.clip(wv, -1.0, 1.0))
        delta = a[6:12] * MAX_JOINT_VEL * self._dt
        self._joint_pos = np.clip(self._joint_pos + delta, -3.1416, 3.1416)
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]
        self.data.ctrl[self.id_fing_l] = float(np.clip((a[12]+1)/2*0.03, 0, 0.03))
        self.data.ctrl[self.id_fing_r] = float(np.clip((a[13]+1)/2*0.03, 0, 0.03))
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL

    def _set_arm_rest(self):
        for jn, ang in ARM_REST.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = ang
        self._joint_pos = np.array([ARM_REST[f"joint_{i+1}"] for i in range(6)])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0]   =  SPAWN_XYZ[0]
        self.data.qpos[1]   =  SPAWN_XYZ[1]
        self.data.qpos[2]   =  SPAWN_XYZ[2]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self._set_arm_rest()
        if self.lidar_spin_id >= 0:
            self.data.ctrl[self.lidar_spin_id] = LIDAR_SPIN_VEL
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            if self.viewer and self.viewer.is_running(): self.viewer.sync()
        self._step    = 0
        self._stuck   = 0
        self._fell    = False
        base_pos      = self.data.xpos[self.base_id]
        self._last_xy = base_pos[:2].copy()
        self._nav.reset(self._last_xy)
        return self._obs()

    def step(self, action):
        self._apply_action(action)
        for _ in range(self.control_dec):
            mujoco.mj_step(self.model, self.data)
            if self.viewer and self.viewer.is_running(): self.viewer.sync()
        self._step += 1

        base_pos = self.data.xpos[self.base_id]
        xy       = base_pos[:2].copy()
        obs      = self._obs()

        # Progreso hacia la meta
        nav_reward, completed = self._nav.update(xy)
        reward = nav_reward + R_ALIVE

        # Suavidad: se mueve alineado hacia la meta
        vel     = xy - self._last_xy
        to_goal = np.array(GOAL_XY) - xy
        if np.linalg.norm(vel) > 1e-4 and np.linalg.norm(to_goal) > 1e-4:
            alignment = float(np.dot(vel, to_goal) /
                              (np.linalg.norm(vel) * np.linalg.norm(to_goal)))
            if alignment > 0.5:
                reward += R_SMOOTH * alignment

        # Stuck
        if np.linalg.norm(xy - self._last_xy) < 0.005:
            self._stuck += 1
            reward += P_STUCK
        else:
            self._stuck = max(0, self._stuck - 1)
        self._last_xy = xy

        # Lidar: cerca de obstaculos
        min_lidar = float(obs["lidar"].min())
        if min_lidar < LIDAR_DANGER:
            reward += P_LIDAR * (LIDAR_DANGER - min_lidar)

        # Inestabilidad angular
        reward += P_SHAKE * float(np.sum(np.square(self.data.qvel[3:6])))

        # Caida
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.20:
            self._fell = True
            reward += P_FALL

        done = self._terminated(completed)
        info = {
            "completed":     completed,
            "dist_to_goal":  self._nav.dist_to_goal,
            "nav_target":    self._nav.current_target,
        }
        return obs, float(reward), done, info

    def _terminated(self, completed: bool) -> bool:
        if completed:             return True
        if self._step >= self.max_steps: return True
        if self._fell:            return True
        if self._stuck >= STUCK_MAX_STEPS: return True
        return False

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except: pass
        try: self.renderer.close()
        except: pass


# Backward-compatible aliases for older tests and scripts.
AesirPalletsEnv = AesirNavEnv


# ══════════════════════════════════════════════════════════════════════════════
#  Redes y PPO (sin cambios respecto a la version anterior)
# ══════════════════════════════════════════════════════════════════════════════
class ImageEncoder(nn.Module):
    def __init__(self, c, h, w, out=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.fc     = nn.Sequential(nn.Linear(flat, out), nn.ReLU(True))
        self.out_dim = out
    def forward(self, x): return self.fc(self.conv(x).flatten(1))

class StateEncoder(nn.Module):
    def __init__(self, in_dim, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.Tanh(),
                                  nn.Linear(128, out), nn.Tanh())
        self.out_dim = out
    def forward(self, x): return self.net(x)

class ConvActorCritic(nn.Module):
    def __init__(self, image_shape, lidar_dim, joint_dim, act_dim,
                 img_feat=256, vec_feat=128, hidden=256, log_std_init=-0.5):
        super().__init__()
        c, h, w = image_shape
        self.img_enc  = ImageEncoder(c, h, w, img_feat)
        self.vec_enc  = StateEncoder(lidar_dim + joint_dim, vec_feat)
        self.trunk    = nn.Sequential(
            nn.Linear(img_feat + vec_feat, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))

    def _fuse(self, im, li, jo):
        return self.trunk(torch.cat(
            [self.img_enc(im), self.vec_enc(torch.cat([li, jo], -1))], -1))

    def forward(self, im, li, jo):
        z   = self._fuse(im, li, jo)
        mu  = torch.tanh(self.actor_mu(z))
        std = torch.clamp(self.log_std, -5, 1).exp().expand_as(mu)
        return mu, std, self.critic(z)

    @torch.no_grad()
    def act(self, obs, device):
        im = obs["images"].unsqueeze(0).to(device)
        li = obs["lidar"].unsqueeze(0).to(device)
        jo = obs["joint_states"].unsqueeze(0).to(device)
        mu, std, val = self(im, li, jo)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        return raw.squeeze(0).cpu().numpy(), float(logp), float(val)

    def evaluate(self, im, li, jo, actions):
        mu, std, val = self(im, li, jo)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, val, entropy


@dataclass
class RolloutBuffer:
    capacity: int; image_shape: tuple; lidar_dim: int; joint_dim: int; act_dim: int
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
        self.images  = np.zeros((self.capacity, c, h, w), np.float32)
        self.lidars  = np.zeros((self.capacity, self.lidar_dim), np.float32)
        self.joints  = np.zeros((self.capacity, self.joint_dim), np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim), np.float32)
        self.logps   = np.zeros(self.capacity, np.float32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.values  = np.zeros(self.capacity, np.float32)
        self.dones   = np.zeros(self.capacity, np.float32)

    def store(self, obs, act, logp, rew, val, done):
        i = self.idx
        self.images[i]  = obs["images"]
        self.lidars[i]  = obs["lidar"]
        self.joints[i]  = obs["joint_states"]
        self.actions[i] = act
        self.logps[i]   = logp
        self.rewards[i] = rew
        self.values[i]  = val
        self.dones[i]   = float(done)
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0; return True
        return False

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        adv = np.zeros_like(self.rewards); gae = 0.0
        for t in reversed(range(self.capacity)):
            nv    = last_val if t == self.capacity-1 else self.values[t+1]
            nt    = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nv * nt - self.values[t]
            gae   = delta + gamma * lam * nt * gae
            adv[t] = gae
        ret = adv + self.values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret


def ppo_update(policy, optimizer, buf, adv, ret,
               epochs, batch_size, clip, vf_c, ent_c, device):
    im  = torch.as_tensor(buf.images,  device=device)
    li  = torch.as_tensor(buf.lidars,  device=device)
    jo  = torch.as_tensor(buf.joints,  device=device)
    ac  = torch.as_tensor(buf.actions, device=device)
    olp = torch.as_tensor(buf.logps,   device=device).unsqueeze(-1)
    a_  = torch.as_tensor(adv,         device=device).unsqueeze(-1)
    r_  = torch.as_tensor(ret,         device=device).unsqueeze(-1)
    stats = {"pi": 0., "v": 0., "ent": 0.}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)),
                                batch_size, drop_last=False):
            lp, val, ent = policy.evaluate(im[idx], li[idx], jo[idx], ac[idx])
            ratio = torch.exp(lp - olp[idx])
            pl    = -torch.min(ratio * a_[idx],
                               torch.clamp(ratio, 1-clip, 1+clip) * a_[idx]).mean()
            vl    = F.smooth_l1_loss(val, r_[idx])
            loss  = pl + vf_c * vl - ent_c * ent
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            stats = {"pi": pl.item(), "v": vl.item(), "ent": ent.item()}
    return stats


def obs_to_tensor(obs):
    return {k: torch.from_numpy(v).float() for k, v in obs.items()}

def save_ckpt(path, policy, optimizer, it, avg_r):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "avg_ep_r": avg_r}, path)


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train(num_iterations=600, steps_per_iter=2048, ppo_epochs=10,
          batch_size=256, gamma=0.99, gae_lambda=0.95, clip=0.20,
          vf_coef=0.5, ent_coef=0.005, lr=3e-4,
          save_every=25, device_str="auto", render=False,
          use_wandb=True, wandb_project="AESIR-NAV", wandb_run=None,
          resume_from=None):

    device = torch.device("cuda" if device_str=="auto" and torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")
    print(f"Camino: {len(PATH_WAYPOINTS)} waypoints | GOAL={GOAL_XY}")

    env = AesirNavEnv(render=render)
    print(f"obs: images={env.image_shape} lidar={env.num_lidar} joints={env.joint_len}")

    policy    = ConvActorCritic(env.image_shape, env.num_lidar, env.joint_len, env.act_len).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    start_it, best_avg = 0, -1e9

    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_it  = ckpt.get("iter", 0)
        best_avg  = ckpt.get("avg_ep_r", -1e9)
        print(f"Resumiendo iter {start_it} (best={best_avg:.1f})")

    buf = RolloutBuffer(steps_per_iter, env.image_shape, env.num_lidar, env.joint_len, env.act_len)

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run, config={
            "n_iterations": num_iterations, "steps_per_iter": steps_per_iter,
            "GOAL_XY": GOAL_XY, "n_waypoints": len(PATH_WAYPOINTS),
            "R_PROGRESS": R_PROGRESS, "R_COMPLETE": R_COMPLETE,
        })

    obs = env.reset()
    ep_r, ep_hist, completed_hist = 0., [], []

    try:
        for it in range(start_it, start_it + num_iterations):
            t0 = time.time()
            for _ in range(steps_per_iter):
                obs_t            = obs_to_tensor(obs)
                act, logp, val   = policy.act(obs_t, device)
                next_obs, rew, done, info = env.step(act)
                buf.store(obs, act, logp, rew, val, done)
                obs   = next_obs
                ep_r += rew
                if done:
                    ep_hist.append(ep_r)
                    completed_hist.append(int(info["completed"]))
                    if len(ep_hist) > 20: ep_hist.pop(0); completed_hist.pop(0)
                    ep_r = 0.; obs = env.reset()

            obs_t = obs_to_tensor(obs)
            with torch.no_grad():
                _, _, lv = policy(obs_t["images"].unsqueeze(0).to(device),
                                   obs_t["lidar"].unsqueeze(0).to(device),
                                   obs_t["joint_states"].unsqueeze(0).to(device))
            adv, ret = buf.compute_gae(float(lv), gamma, gae_lambda)
            stats    = ppo_update(policy, optimizer, buf, adv, ret,
                                   ppo_epochs, batch_size, clip, vf_coef, ent_coef, device)

            avg_r    = float(np.mean(ep_hist)) if ep_hist else float("nan")
            avg_done = float(np.mean(completed_hist)) if completed_hist else 0.
            dt       = time.time() - t0
            print(f"[{it:4d}] avg_r={avg_r:8.1f}  success={avg_done:.2f}  "
                  f"pi={stats['pi']:+.4f}  v={stats['v']:.4f}  ({dt:.1f}s)")

            if use_wandb:
                wandb.log({"iter": it, "avg_ep_reward": avg_r,
                           "success_rate": avg_done,
                           "policy_loss": stats["pi"],
                           "value_loss": stats["v"],
                           "entropy": stats["ent"]},
                          step=(it-start_it+1)*steps_per_iter)

            if (it+1) % save_every == 0:
                p = CHECKPOINT_DIR / f"nav_iter{it+1:05d}.pt"
                save_ckpt(p, policy, optimizer, it+1, avg_r)
                print(f"  ckpt: {p}")

            if ep_hist and avg_r > best_avg:
                best_avg = avg_r
                save_ckpt(CHECKPOINT_DIR / "nav_best.pt", policy, optimizer, it+1, avg_r)
                print(f"  BEST ({avg_r:.1f})")

    finally:
        env.close()
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    train(render=True)