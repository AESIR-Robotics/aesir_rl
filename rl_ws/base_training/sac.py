#!/usr/bin/env python3
"""
sac.py — Soft Actor-Critic: red, replay buffer y update. Pipeline PARALELO a
ppo.py; comparten env (mujoco_sim_base), tarea (base_env) y config.

Por que SAC ademas de PPO: FTR-Bench (misma tarea, oruga con flippers) reporta
SAC 100% vs PPO 84%. Aqui la politica SI responde a la señal de flippers (20x
sobre el azar) pero PPO se estanca en ~7-9% de poses correctas -- perfil de
"la señal esta, falta explotarla", que es donde un off-policy con replay tiene
ventaja: reusa las transiciones buenas en vez de tirarlas cada rollout.

ACCION -- la diferencia importante con ppo_cnn_extractor.py:
    SAC necesita muestreo REPARAMETRIZABLE para propagar gradiente del critico
    al actor, asi que se usa la formulacion canonica (gaussiana aplastada con
    tanh). El actor tiene SAC_ACT_DIM=6 dims y `to_env_action` las lleva a las
    7 que espera el env:
        [0:2] v,w      tanh -> [-1,1]              tal cual
        [2:6] flippers tanh -> (a+1)/2 -> [0,1]    (lo que espera flipper_targets)
        [6]   gate     SIEMPRE 1                   (no lo controla la politica)
    Asi base_env NO se toca y los dos algoritmos comparten la misma tarea.
    El buffer guarda la accion CRUDA de 6 dims (tanh, en [-1,1]), que es sobre
    la que el critico y el log-prob estan definidos.

    POR QUE SIN GATE (primer intento: 1.54% de exito vs 37.5% de PPO): con el
    gate umbralizado, cuando valia 0 flipper_targets ignoraba action[2:6] por
    completo -> 5 de las 7 dimensiones no afectaban la transicion. SAC actualiza
    el actor con dQ/da, asi que ese gradiente era ruido puro (el 71% de las
    dims). PPO lo tolera porque nunca deriva respecto a la accion; SAC no.
    La politica no pierde capacidad: "flippers recogidos" se expresa comandando
    theta~0 directamente.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from . import config as C
from .ppo_cnn_extractor import HeightmapCNN

_EPS = 1e-6


# ── Conversion accion cruda (tanh) -> convencion del env ────────────────────
def to_env_action(a: np.ndarray) -> np.ndarray:
    """(...,6) en [-1,1] -> (...,7) en la convencion que espera base_env.
    Ver el docstring del modulo. No modifica el array de entrada."""
    a = np.asarray(a, dtype=np.float32)
    out = np.empty(a.shape[:-1] + (7,), dtype=np.float32)
    out[..., 0:2] = a[..., 0:2]                      # v, w tal cual
    out[..., 2:6] = (a[..., 2:6] + 1.0) * 0.5        # -> [0,1] (flippers)
    out[..., 6] = 1.0                                # gate SIEMPRE ON
    return out


# ── Encoder compartido por actor y criticos (cada uno tiene el suyo) ────────
class _Encoder(nn.Module):
    """obs plana -> features. Mismo troceo que CNNActorCritic (state | heatmap)."""

    def __init__(self, obs_dim: int, map_pixels: int, hidden: int):
        super().__init__()
        self.map_pixels = map_pixels
        self.map_len = map_pixels * map_pixels
        self.state_dim = obs_dim - self.map_len
        if self.state_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} <= map_pixels^2={self.map_len}")
        self.map_encoder = HeightmapCNN(1, input_size=map_pixels, out_dim=128)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
        )
        self.out_dim = 128 + 64

    def forward(self, obs):
        state = obs[:, :self.state_dim]
        img = obs[:, self.state_dim:].view(-1, 1, self.map_pixels, self.map_pixels)
        return torch.cat([self.map_encoder(img), self.state_encoder(state)], dim=1)


class Actor(nn.Module):
    """Gaussiana aplastada con tanh. Devuelve (accion en [-1,1], log_prob)."""

    def __init__(self, obs_dim, act_dim, map_pixels=64, hidden=C.HIDDEN):
        super().__init__()
        self.enc = _Encoder(obs_dim, map_pixels, hidden)
        self.trunk = nn.Sequential(nn.Linear(self.enc.out_dim, hidden), nn.ReLU(inplace=True))
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        z = self.trunk(self.enc(obs))
        mu = self.mu(z)
        log_std = torch.clamp(self.log_std(z), C.SAC_LOG_STD_MIN, C.SAC_LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mu), None
        dist = Normal(mu, std)
        u = dist.rsample()                       # reparametrizado: deja pasar gradiente
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        # correccion de cambio de variable por el tanh (apendice C del paper)
        logp = dist.log_prob(u).sum(-1, keepdim=True)
        logp = logp - torch.log(1.0 - a.pow(2) + _EPS).sum(-1, keepdim=True)
        return a, logp


class Critic(nn.Module):
    """Dos Q independientes (clipped double-Q). Cada una con su propio encoder."""

    def __init__(self, obs_dim, act_dim, map_pixels=64, hidden=C.HIDDEN):
        super().__init__()
        def _q():
            enc = _Encoder(obs_dim, map_pixels, hidden)
            head = nn.Sequential(
                nn.Linear(enc.out_dim + act_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1))
            return nn.ModuleDict(dict(enc=enc, head=head))
        self.q1, self.q2 = _q(), _q()

    @staticmethod
    def _eval(q, obs, act):
        return q["head"](torch.cat([q["enc"](obs), act], dim=1))

    def forward(self, obs, act):
        return self._eval(self.q1, obs, act), self._eval(self.q2, obs, act)


# ── Replay buffer EN GPU ────────────────────────────────────────────────────
class ReplayBuffer:
    """El heatmap (ya normalizado a [0,1] por get_circular_height_heatmap) se
    guarda en uint8: 4x menos memoria y el error de cuantizacion (1/255) es muy
    inferior a la resolucion del octomap. Es lo que hace viable tener el buffer
    en GPU -- en float32 serian 6.6 GB para 200k y en esta maquina solo hay
    ~3 GB de RAM libre."""

    def __init__(self, capacity, obs_dim, act_dim, state_dim, device):
        self.cap, self.device = capacity, device
        self.state_dim, self.map_len = state_dim, obs_dim - state_dim
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=device)
        self.s_state = z(capacity, state_dim, dt=torch.float32)
        self.s_map = z(capacity, self.map_len, dt=torch.uint8)
        self.n_state = z(capacity, state_dim, dt=torch.float32)
        self.n_map = z(capacity, self.map_len, dt=torch.uint8)
        self.act = z(capacity, act_dim, dt=torch.float32)
        self.rew = z(capacity, 1, dt=torch.float32)
        self.done = z(capacity, 1, dt=torch.float32)   # 1 solo si es terminal REAL
        self.ptr, self.full = 0, False

    def _split(self, obs_t):
        return obs_t[:, :self.state_dim], (obs_t[:, self.state_dim:] * 255.0).round().clamp(0, 255).to(torch.uint8)

    def add(self, obs, act, rew, nobs, done):
        """Arrays numpy (B, ...). `done` ya debe venir SIN las truncaciones."""
        b = obs.shape[0]
        idx = (torch.arange(b, device=self.device) + self.ptr) % self.cap
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        n = torch.as_tensor(nobs, dtype=torch.float32, device=self.device)
        self.s_state[idx], self.s_map[idx] = self._split(o)
        self.n_state[idx], self.n_map[idx] = self._split(n)
        self.act[idx] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.rew[idx] = torch.as_tensor(rew, dtype=torch.float32, device=self.device).view(-1, 1)
        self.done[idx] = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        self.ptr = (self.ptr + b) % self.cap
        self.full = self.full or self.ptr < b

    def __len__(self):
        return self.cap if self.full else self.ptr

    def sample(self, batch):
        i = torch.randint(0, len(self), (batch,), device=self.device)
        obs = torch.cat([self.s_state[i], self.s_map[i].float() / 255.0], dim=1)
        nobs = torch.cat([self.n_state[i], self.n_map[i].float() / 255.0], dim=1)
        return obs, self.act[i], self.rew[i], nobs, self.done[i]


# ── Update ──────────────────────────────────────────────────────────────────
def sac_update(actor, critic, critic_targ, log_alpha,
               opt_a, opt_c, opt_alpha, buf, batch, gamma, tau, target_entropy):
    obs, act, rew, nobs, done = buf.sample(batch)
    alpha = log_alpha.exp().detach()

    # critico: target = r + gamma*(1-done)*(min Q_targ(s',a') - alpha*logp(a'|s'))
    with torch.no_grad():
        na, nlogp = actor(nobs)
        tq1, tq2 = critic_targ(nobs, na)
        target = rew + gamma * (1.0 - done) * (torch.min(tq1, tq2) - alpha * nlogp)
    q1, q2 = critic(obs, act)
    loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    opt_c.zero_grad(set_to_none=True); loss_c.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), C.MAX_GRAD_NORM)
    opt_c.step()

    # actor: maximiza min Q - alpha*logp  (los criticos no reciben gradiente aqui)
    for p in critic.parameters():
        p.requires_grad_(False)
    a_pi, logp = actor(obs)
    qa1, qa2 = critic(obs, a_pi)
    loss_a = (alpha * logp - torch.min(qa1, qa2)).mean()
    opt_a.zero_grad(set_to_none=True); loss_a.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), C.MAX_GRAD_NORM)
    opt_a.step()
    for p in critic.parameters():
        p.requires_grad_(True)

    # alpha automatico: empuja la entropia hacia target_entropy
    loss_alpha = -(log_alpha * (logp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad(set_to_none=True); loss_alpha.backward(); opt_alpha.step()

    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_targ.parameters()):
            pt.mul_(1.0 - tau).add_(tau * p)

    return {"q": float(loss_c.item()), "pi": float(loss_a.item()),
            "alpha": float(log_alpha.exp().item()), "entropy": float(-logp.mean().item())}
