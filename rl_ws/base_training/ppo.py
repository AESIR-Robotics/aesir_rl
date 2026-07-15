#!/usr/bin/env python3
"""
ppo.py — El "aprendizaje": red actor-critic + GAE + update de PPO.

Solo algoritmo (torch/numpy), sin MuJoCo ni logica de tarea — la tarea (obs,
reward, terminacion) vive en base_env.py y la fisica en mujoco_sim_base.py.
Los tamaños de red e hiperparametros por default salen de config.py; el loop
de entrenamiento que junta todo es train_fast.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from . import config as C


# ── Red (misma arquitectura que train_base.py -> checkpoints intercambiables) ─
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = C.HIDDEN,
                 log_std_init: float = C.LOG_STD_INIT):
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
        z = self.trunk(obs)
        mu = self.actor_mu(z)          
        value = self.critic(z)
        log_std = torch.clamp(self.log_std, C.LOG_STD_MIN, C.LOG_STD_MAX)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        action = raw.clamp(-1.0, 1.0)
        return (action.cpu().numpy(), raw.cpu().numpy(), logp.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        """Version de UNA obs (envs no vectorizados: train_base/test_base)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw  = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        action = raw.clamp(-1.0, 1.0)
        return (action.squeeze(0).cpu().numpy(), raw.squeeze(0).cpu().numpy(),
                float(logp.item()), float(value.item()))

    def evaluate(self, obs, actions):
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, value, entropy


# ── PPO update (batch plano) ────────────────────────────────────────────────
def ppo_update(policy, opt, obs, actions, old_logp, adv, ret,
               epochs, batch, clip, vf_c, ent_c, device):
    obs     = torch.as_tensor(obs,     dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=device).unsqueeze(-1)
    adv_t   = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(-1)
    ret_t   = torch.as_tensor(ret, dtype=torch.float32, device=device).unsqueeze(-1)
    n = obs.shape[0]

    m = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    n_updates = 0
    for _ in range(epochs):
        for idx in BatchSampler(SubsetRandomSampler(range(n)), batch, False):
            lp, val, ent = policy.evaluate(obs[idx], actions[idx])
            r  = torch.exp(lp - old_logp[idx])
            pl = -torch.min(r * adv_t[idx],
                            torch.clamp(r, 1 - clip, 1 + clip) * adv_t[idx]).mean()
            vl = F.smooth_l1_loss(val, ret_t[idx])
            loss = pl + vf_c * vl - ent_c * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), C.MAX_GRAD_NORM)
            opt.step()
            m["pi"] += pl.item(); m["v"] += vl.item(); m["ent"] += ent.item()
            n_updates += 1
    for k in m:
        m[k] /= max(n_updates, 1)
    return m


def compute_gae(rew, val, done, last_val, gamma, lam):
    """GAE por env (columnas). rew/val/done son (T, N); last_val es (N,)."""
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nextval = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nextval * nonterm - val[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    ret = adv + val
    return adv, ret
