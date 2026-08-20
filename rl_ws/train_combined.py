"""
train_combined.py — Entrena la base (oruga+flippers, BaseRosEnv) y el brazo
(ArmServoEnv) EN EL MISMO PROCESO, alternando entre iteraciones:

  - "solo": cada uno entrena por separado — rollout completo + PPO update de
    la base, LUEGO rollout completo + PPO update del brazo. Ninguno de los
    dos step()ea mientras el otro entrena.

  - "together": ambos step() se llaman EN PARALELO (dos threads) en cada tick,
    contra el MISMO instante fisico del bridge (mismo mujoco_ros_bridge.py,
    mismo robot) — la base navega mientras el brazo se mueve, cada uno con su
    propia politica/buffer/PPO update independientes.

Por que alternar en vez de ir directo a "together":
    Al principio ambas politicas son ruido. Si desde iter 0 conviven, el
    brazo puede desestabilizar la base (o viceversa) antes de que cualquiera
    aprenda nada util por separado — doble moving-target problem. Los bloques
    "solo" dejan que cada politica converja algo primero; los bloques
    "together" fine-tunean contra la interaccion real (peso/momento del brazo
    moviendose mientras la base navega), que "solo" nunca le enseñaria.

joint_every controla el patron (default=2 -> solo, together, solo, together...
tal como se pidio: "una iteracion split, la siguiente together").

REQUIERE el parche de base_env.py adjunto (close(shutdown_ros=False)) —
sin el, BaseRosEnv.close() mata el contexto rclpy global y se lleva por
delante el nodo del brazo apenas termina el entrenamiento.

Requiere, en otras terminales, ANTES de correr esto:
    cd rl_ws && MUJOCO_GL=glfw python3 mujoco_ros_bridge.py
    ros2 launch robot_moveit_config bringup_viz.launch.py

Uso:
    cd rl_ws && python3 train_combined.py
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from base_env import BaseRosEnv
from arm_env import ArmServoEnv

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

CHECKPOINT_DIR_BASE = Path("./checkpoints_base")
CHECKPOINT_DIR_ARM  = Path("./checkpoints_arm_servo")
CHECKPOINT_DIR_BASE.mkdir(exist_ok=True)
CHECKPOINT_DIR_ARM.mkdir(exist_ok=True)

# Caja de muestreo del goal del brazo, relativa a arm_base_link — igual que en
# train_arm_servo.py.
GOAL_BOX_LOW  = np.array([0.1, -0.3, 0.0])
GOAL_BOX_HIGH = np.array([0.5,  0.3, 0.4])


def sample_goal() -> np.ndarray:
    return np.random.uniform(GOAL_BOX_LOW, GOAL_BOX_HIGH)


# ──────────────────────────── Red (MLP, obs vectorial) ─────────────────────
# Se reusa la MISMA arquitectura para las dos politicas (solo cambian
# obs_dim/act_dim/hidden entre instancias) — identica a la de train_base.py /
# train_arm_servo.py.
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic   = nn.Linear(hidden, 1)
        self.log_std  = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_dim  = act_dim

    def forward(self, obs):
        z       = self.trunk(obs)
        mu      = torch.tanh(self.actor_mu(z))
        value   = self.critic(z)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())

    def evaluate(self, obs, actions):
        mu, std, value = self(obs)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return logp, value, entropy


# ──────────────────────────── Buffer (obs vectorial, generico) ─────────────
@dataclass
class RolloutBuffer:
    capacity: int
    obs_dim:  int
    act_dim:  int

    obs:     np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    logps:   np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values:  np.ndarray = field(init=False)
    dones:   np.ndarray = field(init=False)
    idx:     int = 0

    def __post_init__(self):
        self.obs     = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.logps   = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values  = np.zeros(self.capacity, dtype=np.float32)
        self.dones   = np.zeros(self.capacity, dtype=np.float32)

    def store(self, obs, action, logp, reward, value, done):
        i = self.idx
        self.obs[i]     = obs
        self.actions[i] = action
        self.logps[i]   = logp
        self.rewards[i] = reward
        self.values[i]  = value
        self.dones[i]   = float(done)
        self.idx = (self.idx + 1) % self.capacity

    def compute_gae(self, last_val: float, gamma: float, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv = last_val if t == self.capacity - 1 else self.values[t + 1]
            nd = 1.0 - self.dones[t]
            d  = self.rewards[t] + gamma * nv * nd - self.values[t]
            gae = d + gamma * lam * nd * gae
            adv[t] = gae
        ret = adv + self.values
        return (adv - adv.mean()) / (adv.std() + 1e-8), ret


# ──────────────────────────── PPO (generico, se reusa para las 2 politicas) ─
def ppo_update(policy, opt, buf, adv, ret, epochs, batch, clip, vf_c, ent_c, device):
    obs     = torch.as_tensor(buf.obs,     dtype=torch.float32, device=device)
    actions = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_log = torch.as_tensor(buf.logps,   dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(buf.capacity)), batch, False):
            lp, val, ent = policy.evaluate(obs[idx], actions[idx])
            r  = torch.exp(lp - old_log[idx])
            pl = -torch.min(r * adv_t[idx],
                            torch.clamp(r, 1 - clip, 1 + clip) * adv_t[idx]).mean()
            vl = F.smooth_l1_loss(val, ret_t[idx])
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


# ──────────────────────────── Agent wrapper ─────────────────────────────────
@dataclass
class Agent:
    name:     str            # "base" | "arm" — usado en prints/wandb/checkpoints
    env:      object
    policy:   MLPActorCritic
    opt:      torch.optim.Optimizer
    ckpt_dir: Path
    obs_dim:  int
    act_dim:  int
    best_avg: float = -1e9
    ep_reward: float = 0.0
    ep_history: List[float] = field(default_factory=list)
    obs: Optional[np.ndarray] = None   # ultima obs, SIEMPRE como vector flat


def _extract_obs(agent: Agent, raw_obs) -> np.ndarray:
    """BaseRosEnv.reset()/step() devuelven ya un np.array flat.
    ArmServoEnv devuelve un dict {"joint_states": (16,)} (no hay
    images/lidar via ROS2) — se extrae el vector."""
    if agent.name == "arm":
        return raw_obs["joint_states"]
    return raw_obs


def reset_agent(agent: Agent):
    raw = agent.env.reset()
    if agent.name == "arm":
        agent.env.set_goal(sample_goal())
    agent.obs = _extract_obs(agent, raw)


def step_agent(agent: Agent, buf: RolloutBuffer, device):
    """Un tick de un agente: act -> env.step -> store -> maneja done/reset.
    Pensado para poder correr en su propio thread (ver run_joint_rollout)."""
    action, logp, val = agent.policy.act(agent.obs, device)
    raw_obs, rew, done, _info = agent.env.step(action)
    buf.store(agent.obs, action, logp, rew, val, done)
    agent.obs = _extract_obs(agent, raw_obs)
    agent.ep_reward += rew
    if done:
        agent.ep_history.append(agent.ep_reward)
        if len(agent.ep_history) > 20:
            agent.ep_history.pop(0)
        agent.ep_reward = 0.0
        reset_agent(agent)


# ──────────────────────────── Rollouts ──────────────────────────────────────
def run_solo_rollout(agent: Agent, steps: int, device) -> RolloutBuffer:
    buf = RolloutBuffer(steps, agent.obs_dim, agent.act_dim)
    for _ in range(steps):
        step_agent(agent, buf, device)
    return buf


def run_joint_rollout(base_agent: Agent, arm_agent: Agent, steps: int, device,
                       pool: ThreadPoolExecutor) -> Tuple[RolloutBuffer, RolloutBuffer]:
    """Step "together": cada tick, ambos agentes step()ean EN PARALELO (dos
    threads) contra el mismo instante fisico del bridge. base_env.step()
    duerme ~0.05s, arm_env.step() duerme ~0.1s — al esperar ambos futures
    antes de avanzar, el tick combinado queda acoplado al mas lento (el
    brazo), que es aceptable: estan afectando al MISMO robot fisico, tiene
    sentido que avancen atados al mismo reloj.

    Nota de threading: policy.act() de cada agente es independiente (nn.Module
    propio), y BaseRosEnv/ArmServoEnv usan nodos/topics ROS2 separados, asi
    que no hay estado compartido entre los dos threads — es seguro."""
    buf_base = RolloutBuffer(steps, base_agent.obs_dim, base_agent.act_dim)
    buf_arm  = RolloutBuffer(steps, arm_agent.obs_dim,  arm_agent.act_dim)
    for _ in range(steps):
        fut_base = pool.submit(step_agent, base_agent, buf_base, device)
        fut_arm  = pool.submit(step_agent, arm_agent,  buf_arm,  device)
        fut_base.result()
        fut_arm.result()
    return buf_base, buf_arm


# ──────────────────────────── Update + logging por agente ──────────────────
def update_agent(agent: Agent, buf: RolloutBuffer, gamma, lam, ppo_epochs, batch,
                  clip, vf_c, ent_c, device, it, save_every, use_wandb, tag) -> float:
    with torch.no_grad():
        obs_t = torch.as_tensor(agent.obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, lv = agent.policy(obs_t)
    adv, ret = buf.compute_gae(float(lv.item()), gamma, lam)
    m = ppo_update(agent.policy, agent.opt, buf, adv, ret,
                    ppo_epochs, batch, clip, vf_c, ent_c, device)

    avg = float(np.mean(agent.ep_history)) if agent.ep_history else float("nan")
    print(f"  [{tag}] avg_ep_r={avg:8.2f}  pi={m['pi']:+.4f}  v={m['v']:.4f}  ent={m['ent']:.3f}")

    if use_wandb:
        wandb.log({f"{agent.name}/iter": it, f"{agent.name}/avg_ep_r": avg,
                   f"{agent.name}/policy_loss": m["pi"], f"{agent.name}/value_loss": m["v"],
                   f"{agent.name}/entropy": m["ent"]})

    if (it + 1) % save_every == 0:
        p = agent.ckpt_dir / f"{agent.name}_iter{it+1:05d}.pt"
        save_checkpoint(p, agent.policy, agent.opt, it + 1, avg)
        print(f"    -> checkpoint: {p}")

    if agent.ep_history and avg > agent.best_avg:
        agent.best_avg = avg
        best_p = agent.ckpt_dir / f"{agent.name}_best.pt"
        save_checkpoint(best_p, agent.policy, agent.opt, it + 1, avg)
        print(f"    -> NUEVO MEJOR ({agent.name}): {avg:.2f}")

    return avg


# ──────────────────────────── Training loop ────────────────────────────────
def train(num_iterations:   int   = 500,
          steps_solo_base:  int   = 1024,   # igual que train_base.py
          steps_solo_arm:   int   = 256,    # igual que train_arm_servo.py
          steps_joint:      int   = 256,    # steps por agente en iters "together"
          joint_every:      int   = 2,      # cada joint_every iters, 1 es "together"
          ppo_epochs:       int   = 10,
          batch_base:       int   = 256,
          batch_arm:        int   = 64,
          gamma:            float = 0.99,
          gae_lambda:       float = 0.95,
          clip_param:       float = 0.2,
          vf_coef:          float = 0.5,
          ent_coef:         float = 0.005,
          lr_base:          float = 3e-4,
          lr_arm:           float = 3e-3,
          save_every:       int   = 25,
          device_str:       str   = "auto",
          use_wandb:        bool  = True,
          wandb_project:    str   = "AIDL-PPO-AESIR-COMBINED",
          resume_base:      Optional[str] = None,
          resume_arm:       Optional[str] = None):

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu")
    print(f"Dispositivo: {device}")

    print("[combined] Instanciando BaseRosEnv...")
    base_env = BaseRosEnv()
    print("[combined] Instanciando ArmServoEnv...")
    arm_env = ArmServoEnv()
    print(f"[combined] base: obs_dim={base_env.obs_dim} act_len={base_env.act_len}")
    print(f"[combined] arm:  joint_dim={arm_env.joint_len} act_len={arm_env.act_len}")

    base_policy = MLPActorCritic(base_env.obs_dim, base_env.act_len, hidden=256).to(device)
    arm_policy  = MLPActorCritic(arm_env.joint_len, arm_env.act_len, hidden=128).to(device)
    base_opt = torch.optim.Adam(base_policy.parameters(), lr=lr_base)
    arm_opt  = torch.optim.Adam(arm_policy.parameters(),  lr=lr_arm)

    base_agent = Agent("base", base_env, base_policy, base_opt, CHECKPOINT_DIR_BASE,
                        base_env.obs_dim, base_env.act_len)
    arm_agent  = Agent("arm",  arm_env,  arm_policy,  arm_opt,  CHECKPOINT_DIR_ARM,
                        arm_env.joint_len, arm_env.act_len)

    start_iter = 0
    if resume_base and os.path.isfile(resume_base):
        ckpt = torch.load(resume_base, map_location=device)
        base_policy.load_state_dict(ckpt["policy"])
        base_opt.load_state_dict(ckpt["optimizer"])
        start_iter = max(start_iter, ckpt.get("iter", 0))
        base_agent.best_avg = ckpt.get("avg_ep_r", -1e9)
        print(f"[combined] base resumida desde iter {ckpt.get('iter', 0)}")
    if resume_arm and os.path.isfile(resume_arm):
        ckpt = torch.load(resume_arm, map_location=device)
        arm_policy.load_state_dict(ckpt["policy"])
        arm_opt.load_state_dict(ckpt["optimizer"])
        start_iter = max(start_iter, ckpt.get("iter", 0))
        arm_agent.best_avg = ckpt.get("avg_ep_r", -1e9)
        print(f"[combined] arm resumida desde iter {ckpt.get('iter', 0)}")

    use_wandb = use_wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project=wandb_project, config=dict(
            steps_solo_base=steps_solo_base, steps_solo_arm=steps_solo_arm,
            steps_joint=steps_joint, joint_every=joint_every, ppo_epochs=ppo_epochs,
            batch_base=batch_base, batch_arm=batch_arm, gamma=gamma,
            gae_lambda=gae_lambda, clip_param=clip_param, vf_coef=vf_coef,
            ent_coef=ent_coef, lr_base=lr_base, lr_arm=lr_arm))

    reset_agent(base_agent)
    reset_agent(arm_agent)

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="combined-step")

    try:
        for it in range(start_iter, start_iter + num_iterations):
            t0 = time.time()
            # joint_every=2 -> it par: solo, it impar: together (patron 1:1
            # pedido). joint_every<=0 desactiva "together" del todo.
            joint_iter = joint_every > 0 and ((it - start_iter) % joint_every == joint_every - 1)

            if joint_iter:
                print(f"[Iter {it:4d}] modo=TOGETHER  (steps={steps_joint} c/u, en paralelo)")
                buf_base, buf_arm = run_joint_rollout(base_agent, arm_agent, steps_joint, device, pool)
            else:
                print(f"[Iter {it:4d}] modo=SOLO  (base={steps_solo_base}, arm={steps_solo_arm})")
                buf_base = run_solo_rollout(base_agent, steps_solo_base, device)
                buf_arm  = run_solo_rollout(arm_agent,  steps_solo_arm,  device)

            update_agent(base_agent, buf_base, gamma, gae_lambda, ppo_epochs, batch_base,
                         clip_param, vf_coef, ent_coef, device, it, save_every, use_wandb, "BASE")
            update_agent(arm_agent, buf_arm, gamma, gae_lambda, ppo_epochs, batch_arm,
                         clip_param, vf_coef, ent_coef, device, it, save_every, use_wandb, "ARM ")

            print(f"  ({time.time() - t0:.1f}s reales, iteracion completa)")

    finally:
        pool.shutdown(wait=True)
        # Cerramos los dos envs SIN que ninguno mate el contexto rclpy global
        # por su cuenta, y shutdowneamos rclpy UNA sola vez al final.
        try:
            base_env.close(shutdown_ros=False)
        except TypeError:
            # base_env.py sin el parche (close() viejo, sin el kwarg) -> cae
            # aqui, cierra a la vieja usanza. Aplica el parche adjunto para
            # evitar que esto mate el nodo del brazo antes de cerrarlo.
            print("[combined] AVISO: base_env.py no tiene el parche close(shutdown_ros=...) "
                  "— usando cierre antiguo, puede generar ExternalShutdownException en el brazo.")
            base_env.close()
        arm_env.close()

        import rclpy
        if rclpy.ok():
            rclpy.shutdown()

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train(
        use_wandb=True,
        wandb_project="AIDL-PPO-AESIR-COMBINED",
        resume_base="./checkpoints_base/base_best.pt",
        resume_arm="./checkpoints_arm_servo/arm_servo_best.pt",
    )