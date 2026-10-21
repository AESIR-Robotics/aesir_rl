"""
ppo_cnn_extractor_legacy.py — Legacy base actor-critic (`actor_mu`).

Network of the base checkpoints trained on the server (commit 50788fee): a
single Normal head over all actions (7 = v, w, flipper x4, gate) instead of
the current hybrid Normal + Beta head (ppo_cnn_extractor.CNNActorCritic).
Kept only to load and fine-tune those checkpoints (fsm_training/policies.py).

Same public contract as the current networks, as used by ppo.ppo_update():
    forward(obs)              -> (mu, std, value (B,1))
    act_batch(obs_np, device) -> (action, raw, logp, value) numpy
    act(obs_np, device)       -> same for a single observation
    evaluate(obs, actions)    -> (logp (B,1), value (B,1), entropy scalar)

The observation is [state | heatmap.ravel()]; the heatmap tail is reshaped
to (1, H, W) and encoded by a small CNN. log_std is clamped with the
workspace's LOG_STD_MIN/LOG_STD_MAX.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from . import config as _C


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


class CNNActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 map_pixels: int = 64, map_feat_dim: int = 128,
                 state_feat_dim: int = 64, hidden: int = _C.HIDDEN,
                 log_std_init: float = _C.LOG_STD_INIT):
        super().__init__()
        self.map_pixels = map_pixels
        self.map_len = map_pixels * map_pixels
        self.state_dim = obs_dim - self.map_len
        if self.state_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} <= map_pixels^2={self.map_len}: "
                             f"OBS_DIM must be state_dim + H*W")
        self.act_dim = act_dim

        self.map_encoder = HeightmapCNN(in_channels=1, input_size=map_pixels, out_dim=map_feat_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, state_feat_dim), nn.Tanh(),
        )
        self.shared = nn.Sequential(
            nn.Linear(map_feat_dim + state_feat_dim, hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def _split(self, obs: torch.Tensor):
        state = obs[:, :self.state_dim]
        map_img = obs[:, self.state_dim:].view(-1, 1, self.map_pixels, self.map_pixels)
        return state, map_img

    def forward(self, obs):
        state, map_img = self._split(obs)
        feat = torch.cat([self.map_encoder(map_img), self.state_encoder(state)], dim=1)
        z = self.shared(feat)
        mu = self.actor_mu(z)                                 # no tanh
        value = self.critic(z)                                # (batch, 1)
        log_std = torch.clamp(self.log_std, _C.LOG_STD_MIN, _C.LOG_STD_MAX)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        action = raw.clamp(-1.0, 1.0)
        return (action.cpu().numpy(), raw.cpu().numpy(), logp.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw = dist.sample()
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
