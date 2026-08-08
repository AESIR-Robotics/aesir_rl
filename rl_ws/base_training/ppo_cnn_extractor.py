r"""
ppo_cnn_extractor.py
=====================
CNNActorCritic -- reemplazo DIRECTO de MLPActorCritic (definida en ppo.py),
con la MISMA interfaz publica externa que necesita ppo_update()/train_fast.py:

    act_batch(obs_np, device) -> (action, raw, logp, val) numpy
    act(obs_np, device)       -> (action, raw, logp, val) de UNA obs
    evaluate(obs, actions)    -> (logp, value, entropy)
        logp    (batch,1)  (sum(-1, keepdim=True))
        value   (batch,1)  (sin squeeze)
        entropy escalar    (.mean() ya aplicado)

Distribucion de accion HIBRIDA (ver config.ACT_DIM), no una Gaussiana unica:
    [0:2] v, ω        -- Normal, recortada a [-1,1] (igual que siempre)
    [2:6] flipper×4   -- Beta, soporte YA en [0,1] (sin necesidad de recortar
                         -- antes esto era Normal+clip, lo que genera una
                         inconsistencia entre la accion cruda que ve el
                         log-prob y la accion recortada que ejecuta el
                         entorno; con Beta el soporte ya es exacto)

Hubo una 7a dimension (gate Bernoulli: mandaba los flippers a reposo ignorando
[2:6]). Medido que costaba 18 pp de exito en steps1m -> eliminada.
Ver docs/gate_flippers.md.

forward() ya NO regresa (mu,std,value) -- regresa (params: dict, value); el
unico llamador externo (train_fast.py, bootstrap del ultimo value para GAE)
solo necesita `value`, ver el `_, lv = policy(obs)` ahi.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

from . import config as C

_BETA_EPS = 1e-6   # margen numerico lejos de 0/1 (soporte abierto de Beta)


# ----------------------------------------------------------------------
# 1. El CNN que procesa la parte de imagen del vector de obs
# ----------------------------------------------------------------------
class HeightmapCNN(nn.Module):
    """(batch, 1, H, W) -> (batch, out_dim)"""

    def __init__(self, in_channels=1, input_size=64, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            flat_dim = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.conv(x))


# ----------------------------------------------------------------------
# 2. Actor-Critic completo -- accion hibrida (Normal + Beta)
# ----------------------------------------------------------------------
class CNNActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 map_pixels: int = 64, map_feat_dim: int = 128,
                 state_feat_dim: int = 64, hidden: int = C.HIDDEN,
                 log_std_init: float = C.LOG_STD_INIT):
        super().__init__()

        self.map_pixels = map_pixels
        self.map_len = map_pixels * map_pixels
        self.state_dim = obs_dim - self.map_len
        if self.state_dim <= 0:
            raise ValueError(
                f"obs_dim={obs_dim} <= map_pixels^2={self.map_len}. "
                f"OBS_DIM en config.py debe ser state_dim + H*W, no solo H*W."
            )
        if act_dim != 6:
            raise ValueError(
                f"act_dim={act_dim} -- se esperan 6: [v, ω, flipper×4]. "
                f"Ver config.ACT_DIM."
            )
        self.act_dim = act_dim

        self.map_encoder = HeightmapCNN(in_channels=1, input_size=map_pixels, out_dim=map_feat_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, state_feat_dim), nn.Tanh(),
        )
        self.shared = nn.Sequential(
            nn.Linear(map_feat_dim + state_feat_dim, hidden), nn.Tanh(),
        )

        self.actor_vw = nn.Linear(hidden, 2)               # mu de [v, ω]
        self.log_std_vw = nn.Parameter(torch.full((2,), log_std_init))
        self.actor_flip = nn.Linear(hidden, 8)              # (alpha_raw,beta_raw) x 4 flippers
        self.critic = nn.Linear(hidden, 1)

    def _split(self, obs: torch.Tensor):
        """obs: (batch, obs_dim) -> (state (batch,state_dim), map_img (batch,1,H,W))"""
        state = obs[:, :self.state_dim]
        map_img = obs[:, self.state_dim:].view(-1, 1, self.map_pixels, self.map_pixels)
        return state, map_img

    def forward(self, obs):
        """(batch, obs_dim) -> (params: dict, value (batch,1)).
        params = {mu_vw, std_vw, alpha, beta} -- ver _dists()."""
        state, map_img = self._split(obs)
        feat = torch.cat([self.map_encoder(map_img), self.state_encoder(state)], dim=1)
        z = self.shared(feat)

        mu_vw = self.actor_vw(z)                                        # (batch,2)
        log_std_vw = torch.clamp(self.log_std_vw, C.LOG_STD_MIN, C.LOG_STD_MAX)
        std_vw = log_std_vw.exp().expand_as(mu_vw)

        raw_flip = self.actor_flip(z)                                   # (batch,8)
        alpha = F.softplus(raw_flip[:, 0::2]) + 1.0                     # (batch,4)
        beta_ = F.softplus(raw_flip[:, 1::2]) + 1.0                     # (batch,4)

        value = self.critic(z)                                          # (batch,1)

        return dict(mu_vw=mu_vw, std_vw=std_vw, alpha=alpha, beta=beta_), value

    def _dists(self, obs):
        params, value = self(obs)
        d_vw = Normal(params["mu_vw"], params["std_vw"])
        d_flip = Beta(params["alpha"], params["beta"])
        return d_vw, d_flip, value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        """Rollout vectorizado (VecMujocoEnv, N envs)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        d_vw, d_flip, value = self._dists(obs)

        raw_vw = d_vw.sample()                                          # (batch,2)
        raw_flip = d_flip.sample().clamp(_BETA_EPS, 1.0 - _BETA_EPS)     # (batch,4)

        logp = d_vw.log_prob(raw_vw).sum(-1) + d_flip.log_prob(raw_flip).sum(-1)
        raw = torch.cat([raw_vw, raw_flip], dim=-1)
        action = torch.cat([raw_vw.clamp(-1.0, 1.0), raw_flip], dim=-1)

        return (action.cpu().numpy(), raw.cpu().numpy(), logp.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        """Version de UNA obs (envs no vectorizados: train_base/test_base)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        d_vw, d_flip, value = self._dists(obs)

        raw_vw = d_vw.sample()
        raw_flip = d_flip.sample().clamp(_BETA_EPS, 1.0 - _BETA_EPS)

        logp = d_vw.log_prob(raw_vw).sum(-1) + d_flip.log_prob(raw_flip).sum(-1)
        raw = torch.cat([raw_vw, raw_flip], dim=-1)
        action = torch.cat([raw_vw.clamp(-1.0, 1.0), raw_flip], dim=-1)

        return (action.squeeze(0).cpu().numpy(), raw.squeeze(0).cpu().numpy(),
                float(logp.item()), float(value.item()))

    def evaluate(self, obs, actions):
        """(logp, value, entropy) -- MISMAS formas que antes, ver ppo_update():
            logp    (batch,1)
            value   (batch,1)
            entropy escalar (mean ya aplicado, suma de las 2 distribuciones)
        `actions` = el tensor `raw` guardado por act_batch/act (6 columnas)."""
        d_vw, d_flip, value = self._dists(obs)

        raw_vw = actions[:, 0:2]
        raw_flip = actions[:, 2:6].clamp(_BETA_EPS, 1.0 - _BETA_EPS)

        logp = (d_vw.log_prob(raw_vw).sum(-1, keepdim=True)
                + d_flip.log_prob(raw_flip).sum(-1, keepdim=True))
        entropy = (d_vw.entropy().sum(-1) + d_flip.entropy().sum(-1)).mean()

        return logp, value, entropy
