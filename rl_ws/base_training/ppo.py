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
from torch.distributions import Normal, Beta, Bernoulli
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from . import config as C

# La validacion de argumentos/soporte de torch.distributions esta ON por
# default (__debug__) y con la accion hibrida corre 3 veces por llamada; el
# chequeo de soporte hace .all(), que fuerza un sync con la GPU en cada
# act_batch del rollout (medido ~2x mas caro). Los rangos ya los garantiza la
# parametrizacion (softplus+1 para Beta, logits para Bernoulli).
torch.distributions.Distribution.set_default_validate_args(False)

_BETA_EPS = 1e-6   # margen numerico lejos de 0/1 (soporte abierto de Beta)


# ── Red (misma accion hibrida que CNNActorCritic -> checkpoints comparables) ──
class MLPActorCritic(nn.Module):
    """Accion HIBRIDA, identica a CNNActorCritic (ver ppo_cnn_extractor.py para
    el razonamiento completo): [0:2] v,ω Normal; [2:6] flipper×4 Beta en [0,1];
    [6] gate Bernoulli 0.0/1.0."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = C.HIDDEN,
                 log_std_init: float = C.LOG_STD_INIT):
        super().__init__()
        if act_dim != 7:
            raise ValueError(
                f"act_dim={act_dim} -- la accion hibrida asume EXACTO 7: "
                f"[v, ω, flipper×4, gate] (ver config.ACT_DIM)."
            )
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor_vw = nn.Linear(hidden, 2)
        self.log_std_vw = nn.Parameter(torch.full((2,), log_std_init))
        self.actor_flip = nn.Linear(hidden, 8)     # (alpha_raw,beta_raw) x 4
        self.actor_gate = nn.Linear(hidden, 1)
        self.critic   = nn.Linear(hidden, 1)
        self.act_dim  = act_dim

    def forward(self, obs):
        """(batch,obs_dim) -> (params: dict, value (batch,1))."""
        z = self.trunk(obs)
        mu_vw = self.actor_vw(z)
        log_std_vw = torch.clamp(self.log_std_vw, C.LOG_STD_MIN, C.LOG_STD_MAX)
        std_vw = log_std_vw.exp().expand_as(mu_vw)
        raw_flip = self.actor_flip(z)
        alpha = F.softplus(raw_flip[:, 0::2]) + 1.0
        beta_ = F.softplus(raw_flip[:, 1::2]) + 1.0
        gate_logit = self.actor_gate(z).squeeze(-1)
        value = self.critic(z)
        return dict(mu_vw=mu_vw, std_vw=std_vw, alpha=alpha, beta=beta_,
                    gate_logit=gate_logit), value

    def _dists(self, obs):
        params, value = self(obs)
        return (Normal(params["mu_vw"], params["std_vw"]),
                Beta(params["alpha"], params["beta"]),
                Bernoulli(logits=params["gate_logit"]),
                value)

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        d_vw, d_flip, d_gate, value = self._dists(obs)
        raw_vw = d_vw.sample()
        raw_flip = d_flip.sample().clamp(_BETA_EPS, 1.0 - _BETA_EPS)
        raw_gate = d_gate.sample()
        logp = (d_vw.log_prob(raw_vw).sum(-1)
                + d_flip.log_prob(raw_flip).sum(-1)
                + d_gate.log_prob(raw_gate))
        raw = torch.cat([raw_vw, raw_flip, raw_gate.unsqueeze(-1)], dim=-1)
        action = torch.cat([raw_vw.clamp(-1.0, 1.0), raw_flip,
                            raw_gate.unsqueeze(-1)], dim=-1)
        return (action.cpu().numpy(), raw.cpu().numpy(), logp.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        """Version de UNA obs (envs no vectorizados: train_base/test_base)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        d_vw, d_flip, d_gate, value = self._dists(obs)
        raw_vw = d_vw.sample()
        raw_flip = d_flip.sample().clamp(_BETA_EPS, 1.0 - _BETA_EPS)
        raw_gate = d_gate.sample()
        logp = (d_vw.log_prob(raw_vw).sum(-1)
                + d_flip.log_prob(raw_flip).sum(-1)
                + d_gate.log_prob(raw_gate))
        raw = torch.cat([raw_vw, raw_flip, raw_gate.unsqueeze(-1)], dim=-1)
        action = torch.cat([raw_vw.clamp(-1.0, 1.0), raw_flip,
                            raw_gate.unsqueeze(-1)], dim=-1)
        return (action.squeeze(0).cpu().numpy(), raw.squeeze(0).cpu().numpy(),
                float(logp.item()), float(value.item()))

    def evaluate(self, obs, actions):
        d_vw, d_flip, d_gate, value = self._dists(obs)
        raw_flip = actions[:, 2:6].clamp(_BETA_EPS, 1.0 - _BETA_EPS)
        logp = (d_vw.log_prob(actions[:, 0:2]).sum(-1, keepdim=True)
                + d_flip.log_prob(raw_flip).sum(-1, keepdim=True)
                + d_gate.log_prob(actions[:, 6]).unsqueeze(-1))
        entropy = (d_vw.entropy().sum(-1)
                   + d_flip.entropy().sum(-1)
                   + d_gate.entropy()).mean()
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
