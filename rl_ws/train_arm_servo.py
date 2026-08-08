"""
train_arm_servo.py — Entrena PPO contra ArmServoEnv (arm_env.py), es decir,
a traves del stack ROS2 real (MoveIt Servo + ros2_control + mujoco_ros_bridge.py),
no MuJoCo directo.

Cada env.step() bloquea SERVO_CONTROL_DT segundos de tiempo REAL (no se puede
acelerar publicando/leyendo topics mas rapido que tiempo real) — esto es
muchisimo mas lento que entrenar con ArmMuJoCoEnv directo (mj_step() en el
mismo proceso). Se eligio steps_per_iter pequeno por default por esta razon.

No hay bridge de camaras/lidar via ROS2, asi que la observacion es solo
joint_states (16,) — por eso la red es un MLP simple (MlpActorCritic), no
ConvActorCritic.

Goal: como nada en el proyecto seteaba un goal antes, este script samplea un
punto aleatorio dentro de una caja relativa a arm_base_link en cada reset()
y lo pasa a env.set_goal() — ajusta GOAL_BOX_LOW/HIGH si quieres otro rango.

Requiere, en otras terminales, ANTES de correr esto:
    ros2 launch robot_moveit_config bringup_viz.launch.py
    cd rl_ws && python3 mujoco_ros_bridge.py

Uso:
    cd rl_ws
    python3 train_arm_servo.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arm_env import ArmServoEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

CHECKPOINT_DIR = Path("./checkpoints_arm_servo")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Caja de muestreo del goal, relativa a arm_base_link (mismo frame que usa
# ArmServoEnv._reward para medir distancia EE->goal via TF).
GOAL_BOX_LOW  = np.array([0.1, -0.3, 0.0])
GOAL_BOX_HIGH = np.array([0.5,  0.3, 0.4])


def sample_goal() -> np.ndarray:
    return np.random.uniform(GOAL_BOX_LOW, GOAL_BOX_HIGH)


# ──────────────────────────── Red (MLP, sin imagenes/lidar) ────────────────
class MlpActorCritic(nn.Module):
    """Misma interfaz que ConvActorCritic (act/evaluate/forward) pero solo MLP."""

    def __init__(self, joint_dim: int, act_dim: int, hidden: int = 128, log_std_init: float = -0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(joint_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,    hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_dim  = act_dim

    def forward(self, joints):
        z       = self.trunk(joints)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, joints: torch.Tensor, device):
        joints = joints.unsqueeze(0).to(device)
        mu, std, value = self(joints)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())

    def evaluate(self, joints, actions):
        mu, std, value = self(joints)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return logp, value, entropy


# ──────────────────────────── Buffer ───────────────────────────────────────
@dataclass
class RolloutBuffer:
    capacity:  int
    joint_dim: int
    act_dim:   int

    joints:  np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    logps:   np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values:  np.ndarray = field(init=False)
    dones:   np.ndarray = field(init=False)
    idx:     int = 0

    def __post_init__(self):
        self.joints  = np.zeros((self.capacity, self.joint_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim),   dtype=np.float32)
        self.logps   = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values  = np.zeros(self.capacity, dtype=np.float32)
        self.dones   = np.zeros(self.capacity, dtype=np.float32)

    def store(self, joints, action, logp, reward, value, done) -> bool:
        i = self.idx
        self.joints[i]  = joints
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


def ppo_update(policy, opt, buf, adv, ret, epochs, batch, clip, vf_c, ent_c, device):
    joints  = torch.as_tensor(buf.joints,  dtype=torch.float32, device=device)
    actions = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_log = torch.as_tensor(buf.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)), batch, False):
            lp, val, ent = policy.evaluate(joints[idx], actions[idx])
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


def save_checkpoint(path, policy, opt, it, avg):
    torch.save({"iter": it, "policy": policy.state_dict(),
                "optimizer": opt.state_dict(), "avg_ep_r": avg}, path)


# ──────────────────────────── Training loop ────────────────────────────────
def train(num_iterations: int   = 200,
          steps_per_iter: int   = 256,   # 256*0.1s ~= 26s de rollout por iteracion
          ppo_epochs:     int   = 10,
          batch_size:     int   = 64,
          gamma:          float = 0.99,
          gae_lambda:     float = 0.95,
          clip_param:     float = 0.2,
          vf_coef:        float = 0.5,
          ent_coef:       float = 0.005,
          lr:             float = 3e-3,
          save_every:     int   = 10,
          device_str:     str   = "auto",
          use_wandb:      bool  = True,
          wandb_project:  str   = "AIDL-PPO-AESIR-ARM-SERVO",
          resume_from:    str   = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"Dispositivo: {device}")

    env = ArmServoEnv()
    print(f"act_len   = {env.act_len}  (3 lin + 3 ang del EE, dedo_izq, dedo_der)")
    print(f"joint_dim = {env.joint_len}")
    print("CONTROL_DT real por step — entrenamiento NO acelerado vs. tiempo real")

    policy = MlpActorCritic(joint_dim=env.joint_len, act_dim=env.act_len).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    start_iter = 0
    best_avg   = -1e9

    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0)
        best_avg   = ckpt.get("avg_ep_r", -1e9)
        print(f"Resumiendo desde iter {start_iter}  (best_avg={best_avg:.2f})")

    buf = RolloutBuffer(capacity=steps_per_iter, joint_dim=env.joint_len, act_dim=env.act_len)

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=wandb_project, config=dict(
            num_iterations=num_iterations, steps_per_iter=steps_per_iter,
            ppo_epochs=ppo_epochs, batch_size=batch_size, gamma=gamma,
            gae_lambda=gae_lambda, clip_param=clip_param,
            vf_coef=vf_coef, ent_coef=ent_coef, lr=lr,
        ))
        wandb.watch(policy, log="gradients", log_freq=100)

    obs        = env.reset()
    env.set_goal(sample_goal())
    ep_reward  = 0.0
    ep_history: List[float] = []

    try:
        for it in range(start_iter, start_iter + num_iterations):
            t0 = time.time()
            for _ in range(steps_per_iter):
                joints_t          = torch.from_numpy(obs["joint_states"]).float()
                action, logp, val = policy.act(joints_t, device)
                nobs, rew, done, _ = env.step(action)
                buf.store(obs["joint_states"], action, logp, rew, val, done)
                obs       = nobs
                ep_reward += rew
                if done:
                    ep_history.append(ep_reward)
                    if len(ep_history) > 20: ep_history.pop(0)
                    ep_reward = 0.0
                    obs = env.reset()
                    env.set_goal(sample_goal())

            with torch.no_grad():
                joints_t = torch.from_numpy(obs["joint_states"]).float().unsqueeze(0).to(device)
                _, _, lv = policy(joints_t)
            adv, ret = buf.compute_gae(float(lv.item()), gamma, gae_lambda)
            m = ppo_update(policy, opt, buf, adv, ret,
                           ppo_epochs, batch_size, clip_param, vf_coef, ent_coef, device)

            avg = float(np.mean(ep_history)) if ep_history else float("nan")
            dt  = time.time() - t0
            print(f"[Iter {it:4d}] avg_ep_r={avg:8.2f}  "
                  f"pi={m['pi']:+.4f}  v={m['v']:.4f}  "
                  f"ent={m['ent']:.3f}  ({dt:.1f}s reales)")

            if use_wandb:
                wandb.log({"iter": it, "avg_ep_r": avg,
                           "policy_loss": m["pi"], "value_loss": m["v"],
                           "entropy": m["ent"], "iter_time_s": dt},
                          step=(it - start_iter + 1) * steps_per_iter)

            if (it + 1) % save_every == 0:
                p = CHECKPOINT_DIR / f"arm_servo_iter{it+1:05d}.pt"
                save_checkpoint(p, policy, opt, it + 1, avg)
                print(f"  -> checkpoint: {p}")

            if ep_history and avg > best_avg:
                best_avg = avg
                best_p   = CHECKPOINT_DIR / "arm_servo_best.pt"
                save_checkpoint(best_p, policy, opt, it + 1, avg)
                print(f"  -> NUEVO MEJOR ({avg:.2f}) -> {best_p}")

    finally:
        env.close()
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    train()
