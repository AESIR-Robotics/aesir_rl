#!/usr/bin/env python3
"""
net.py — Red actor-critic para la tarea de botones.

Por que Beta y no una Normal recortada
======================================
Las 7 dimensiones de la accion estan ACOTADAS en [-1,1] (twist cartesiano
unitless + garra). Con una Normal hay que recortar la muestra al rango, pero el
log_prob se calcula sobre la muestra SIN recortar: la probabilidad que PPO usa
en el ratio no es la de la accion que el env ejecuto. Ese desajuste es el que la
base ya resolvio usando Beta para sus dims acotadas (los flippers) -- ver
base_training/ppo.py y ppo_cnn_extractor.py. `train_arm_servo.py` todavia tiene
la version con Normal+clip; aqui no se repite.

Beta tiene soporte exacto en (0,1), asi que se mapea a [-1,1] con a = 2b - 1 y
no hay nada que recortar. La parametrizacion softplus(x)+1 mantiene alpha,beta
>= 1 (siempre unimodal, sin masa en los extremos). En la init, los pesos de
salida son pequeños -> alpha ~ beta ~ softplus(0)+1 = 1.69 -> Beta simetrica
centrada en 0.5 -> accion 0.0: el brazo arranca quieto, no sacudiendose.

El resto (update de PPO, GAE, normalizacion del valor) se REUSA tal cual de
base_training/ppo.py: el algoritmo es el mismo que el de la base, lo unico que
cambia es la distribucion de la accion y el tamaño de la observacion.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from . import config as C

torch.distributions.Distribution.set_default_validate_args(False)

_BETA_EPS = 1e-6   # margen numerico lejos de 0/1 (soporte abierto de Beta)


def _to_env(b: torch.Tensor) -> torch.Tensor:
    """Beta en (0,1) -> accion del env en (-1,1)."""
    return 2.0 * b - 1.0


def _to_beta(a: torch.Tensor) -> torch.Tensor:
    """Accion del env en [-1,1] -> soporte de la Beta en (0,1)."""
    return ((a + 1.0) * 0.5).clamp(_BETA_EPS, 1.0 - _BETA_EPS)


class BetaActorCritic(nn.Module):
    """Trunk Tanh compartido -> cabeza Beta (7 dims) + critico.

    El critico predice en espacio NORMALIZADO (ver RunningMeanStd en
    base_training/ppo.py): hay que denormalize() antes de GAE y normalize() el
    retorno antes de usarlo como objetivo. Sin eso, el gradiente del critico
    ahoga al de la politica a traves del trunk compartido -- esta medido en el
    repo, no es teoria.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = C.HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 2 * act_dim)   # (alpha_raw, beta_raw) x act_dim
        self.critic = nn.Linear(hidden, 1)
        self.act_dim = act_dim

        # Salida pequeña -> Beta casi simetrica -> accion ~0 en la iteracion 0.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        z = self.trunk(obs)
        raw = self.actor(z)
        alpha = F.softplus(raw[:, 0::2]) + 1.0
        beta_ = F.softplus(raw[:, 1::2]) + 1.0
        return alpha, beta_, self.critic(z)

    def _dist(self, obs):
        alpha, beta_, value = self(obs)
        return Beta(alpha, beta_), value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        dist, value = self._dist(obs)
        b = dist.sample().clamp(_BETA_EPS, 1.0 - _BETA_EPS)
        logp = dist.log_prob(b).sum(-1)
        action = _to_env(b)
        return (action.cpu().numpy(), action.cpu().numpy(),
                logp.cpu().numpy(), value.squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device, deterministic: bool = False):
        """Una sola obs. deterministic=True usa la MEDIA de la Beta
        (alpha/(alpha+beta)) en vez de muestrear -- para evaluar/ver la politica
        sin el ruido de exploracion."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        dist, value = self._dist(obs)
        b = (dist.mean if deterministic else dist.sample()).clamp(_BETA_EPS, 1.0 - _BETA_EPS)
        logp = dist.log_prob(b).sum(-1)
        action = _to_env(b).squeeze(0)
        return (action.cpu().numpy(), action.cpu().numpy(),
                float(logp.item()), float(value.item()))

    def evaluate(self, obs, actions):
        """Interfaz que espera base_training.ppo.ppo_update."""
        dist, value = self._dist(obs)
        b = _to_beta(actions)
        logp = dist.log_prob(b).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, value, entropy
