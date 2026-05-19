"""
train_base.py  —  Entrena el Modelo B (base oruga) con PPO.

Usa BaseMuJoCoEnv que tiene:
  - 3 cámaras RGB → CNN encoder (mismo que el entrenamiento completo)
  - 7 rayos lidar → MLP
  - joint_states de base → MLP
  - capa differential drive

Acción (6 valores en [-1, 1]):
  [0] v_lin   [1] ω_ang   [2..5] pos flippers

La red es ConvActorCritic, igual que ppo_conv_train.py.
Los pesos del trunk se pueden luego transferir al entrenamiento conjunto.

Uso:
    cd model_robot
    MUJOCO_GL=egl python3 train_base.py

    # Sin render:
    MUJOCO_GL=egl python3 -c "from train_base import train; train(render=False)"
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

from base_env import BaseMuJoCoEnv
import mujoco
import mujoco.viewer

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# ──────────────────────────── Ruta XML ─────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_FULL    = os.path.join(_HERE, "../models/aesir_complete.xml")
_ROBOT   = os.path.join(_HERE, "../models/aesir_mujoco_robot.xml")
XML_PATH = _FULL if os.path.exists(_FULL) else _ROBOT

CHECKPOINT_DIR = Path("./checkpoints_base")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────── Red (igual que ppo_conv_train) ───────────────
class ImageEncoder(nn.Module):
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

    def forward(self, x): return self.fc(self.conv(x).flatten(1))


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, out_dim), nn.Tanh(),
        )
        self.out_dim = out_dim

    def forward(self, x): return self.net(x)


class ConvActorCritic(nn.Module):
    """Misma arquitectura que ppo_conv_train.py — pesos transferibles."""

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
        self.img_enc  = ImageEncoder(c, h, w, out_dim=img_feat)
        self.vec_enc  = StateEncoder(lidar_dim + joint_dim, out_dim=vec_feat)
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
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], device):
        images = obs["images"].unsqueeze(0).to(device)
        lidar  = obs["lidar"].unsqueeze(0).to(device)
        joints = obs["joint_states"].unsqueeze(0).to(device)
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


# ──────────────────────────── Buffer ───────────────────────────────────────
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
    idx:     int = 0

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

    def compute_gae(self, last_val: float, gamma: float, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv  = last_val if t == self.capacity - 1 else self.values[t + 1]
            nd  = 1.0 - self.dones[t]
            d   = self.rewards[t] + gamma * nv * nd - self.values[t]
            gae = d + gamma * lam * nd * gae
            adv[t] = gae
        ret = adv + self.values
        return (adv - adv.mean()) / (adv.std() + 1e-8), ret


# ──────────────────────────── PPO ──────────────────────────────────────────
def ppo_update(policy, opt, buf, adv, ret,
               epochs, batch, clip, vf_c, ent_c, device):
    images  = torch.as_tensor(buf.images,  dtype=torch.float32, device=device)
    lidar   = torch.as_tensor(buf.lidars,  dtype=torch.float32, device=device)
    joints  = torch.as_tensor(buf.joints,  dtype=torch.float32, device=device)
    actions = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_log = torch.as_tensor(buf.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)), batch, False):
            lp, val, ent = policy.evaluate(images[idx], lidar[idx], joints[idx], actions[idx])
            r    = torch.exp(lp - old_log[idx])
            pl   = -torch.min(r * adv_t[idx],
                              torch.clamp(r, 1 - clip, 1 + clip) * adv_t[idx]).mean()
            vl   = F.smooth_l1_loss(val, ret_t[idx])
            loss = pl + vf_c * vl - ent_c * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            m["pi"] = pl.item(); m["v"] = vl.item(); m["ent"] = ent.item()
    return m


# ──────────────────────────── Helpers ──────────────────────────────────────
def obs_to_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float() for k, v in obs.items()}


def save_checkpoint(path, policy, opt, it, avg):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": opt.state_dict(), "avg_ep_r": avg}, path)


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
          wandb_project:   str   = "AIDL-PPO-AESIR-BASE",
          resume_from:     str   = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Dispositivo: {device}")

    env = BaseMuJoCoEnv(XML_PATH, render=render)
    print(f"act_len     = {env.act_len}  (6 = v_lin, ω, flip×4)")
    print(f"image_shape = {env.image_shape}  (3 cámaras × 3 canales)")
    print(f"lidar_dim   = {env.num_lidar}")
    print(f"joint_dim   = {env.joint_len}  (qpos+qvel de {len(env._obs_act_ids)} actuadores)")
    print(f"XML         = {XML_PATH}")

    policy = ConvActorCritic(
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    start_iter = 0
    best_avg   = -1e9

    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
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
        wandb.init(project=wandb_project, config=dict(
            num_iterations=num_iterations, steps_per_iter=steps_per_iter,
            ppo_epochs=ppo_epochs, batch_size=batch_size, gamma=gamma,
            gae_lambda=gae_lambda, clip_param=clip_param,
            vf_coef=vf_coef, ent_coef=ent_coef, lr=lr,
            act_dim=env.act_len, image_shape=list(env.image_shape),
            lidar_dim=env.num_lidar, joint_dim=env.joint_len,
        ))
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
                nobs, rew, done, _ = env.step(action)
                buf.store(obs, action, logp, rew, val, done)
                obs       = nobs
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
            adv, ret = buf.compute_gae(float(lv.item()), gamma, gae_lambda)
            m = ppo_update(policy, opt, buf, adv, ret,
                           ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)

            avg = float(np.mean(ep_history)) if ep_history else float("nan")
            dt  = time.time() - t0
            print(f"[Iter {it:4d}] avg_ep_r={avg:8.2f}  "
                  f"pi={m['pi']:+.4f}  v={m['v']:.4f}  "
                  f"ent={m['ent']:.3f}  ({dt:.1f}s)")

            if use_wandb:
                wandb.log({"iter": it, "avg_ep_r": avg,
                           "policy_loss": m["pi"], "value_loss": m["v"],
                           "entropy": m["ent"], "iter_time_s": dt},
                          step=(it - start_iter + 1) * steps_per_iter)

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"base_iter{it+1:05d}.pt"
                save_checkpoint(p, policy, opt, it + 1, avg)
                print(f"  ↳ checkpoint: {p}")

            if ep_history and avg > best_avg:
                best_avg = avg
                best_p   = CHECKPOINT_DIR / "base_best.pt"
                save_checkpoint(best_p, policy, opt, it + 1, avg)
                print(f"  ↳ NUEVO MEJOR ({avg:.2f}) → {best_p}")

    finally:
        env.close()
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    train(render=False)
