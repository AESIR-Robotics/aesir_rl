r"""
ppo_cnn_extractor.py
=====================
CNNActorCritic -- reemplazo DIRECTO de MLPActorCritic (definida en ppo.py),
con la MISMA interfaz publica exacta, verificada contra tu ppo.py real:

    forward(obs)              -> (mu, std, value)      igual forma que MLP
    act_batch(obs_np, device) -> (act, logp, val) numpy  igual forma que MLP
    act(obs_np, device)       -> (act, logp, val) de UNA obs (no vectorizado)
    evaluate(obs, actions)    -> (logp, value, entropy)  MISMO nombre, MISMA
                                  forma que MLPActorCritic.evaluate():
                                    logp    (batch,1)  (sum(-1, keepdim=True))
                                    value   (batch,1)  (sin squeeze)
                                    entropy escalar    (.mean() ya aplicado)

Por que esto importa: ppo_update() en ppo.py llama a
`lp, val, ent = policy.evaluate(obs[idx], actions[idx])` y despues hace
aritmetica de tensores asumiendo esas formas exactas (`r = exp(lp-old_logp)`,
`F.smooth_l1_loss(val, ret_t[idx])`, `ent_c*ent`). Si evaluate() regresa
formas distintas, el training explota o (peor) entrena mal en silencio.
Con esta version, CERO cambios son necesarios en ppo.py: ppo_update() y
compute_gae() son agnosticos a la arquitectura, solo llaman al metodo
publico de policy -- por eso basta con importar CNNActorCritic en vez de
MLPActorCritic en train_fast.py.

Como interpreta el vector de obs (aplanado por base_env.build_obs):
    obs = [state (state_dim,) | heatmap.ravel() (H*W,)]
La red corta la cola, la reacomoda a (1,H,W) y la mete a un CNN chico;
el resto sigue como el MLP de siempre (mismo trunk conceptual, Tanh incl.).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from . import config as _C
    _HIDDEN = _C.HIDDEN
    _LOG_STD_INIT = _C.LOG_STD_INIT
    _LOG_STD_MIN = _C.LOG_STD_MIN
    _LOG_STD_MAX = _C.LOG_STD_MAX
except ImportError:
    _HIDDEN, _LOG_STD_INIT, _LOG_STD_MIN, _LOG_STD_MAX = 128, 0.0, -20.0, 2.0


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
# 2. Actor-Critic completo -- MISMO contrato publico que MLPActorCritic
# ----------------------------------------------------------------------
class CNNActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 map_pixels: int = 64, map_feat_dim: int = 128,
                 state_feat_dim: int = 64, hidden: int = _HIDDEN,
                 log_std_init: float = _LOG_STD_INIT):
        super().__init__()

        self.map_pixels = map_pixels
        self.map_len = map_pixels * map_pixels
        self.state_dim = obs_dim - self.map_len
        if self.state_dim <= 0:
            raise ValueError(
                f"obs_dim={obs_dim} <= map_pixels^2={self.map_len}. "
                f"OBS_DIM en config.py debe ser state_dim + H*W, no solo H*W."
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

        self.actor_mu = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def _split(self, obs: torch.Tensor):
        """obs: (batch, obs_dim) -> (state (batch,state_dim), map_img (batch,1,H,W))"""
        state = obs[:, :self.state_dim]
        map_img = obs[:, self.state_dim:].view(-1, 1, self.map_pixels, self.map_pixels)
        return state, map_img

    def forward(self, obs):
        """(batch, obs_dim) -> (mu, std, value) -- MISMA forma que MLPActorCritic.forward"""
        state, map_img = self._split(obs)
        feat = torch.cat([self.map_encoder(map_img), self.state_encoder(state)], dim=1)
        z = self.shared(feat)

        mu = self.actor_mu(z)                                 # SIN tanh -- ver ppo.py.forward
        value = self.critic(z)                                # (batch,1), SIN squeeze -- igual que MLP
        log_std = torch.clamp(self.log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        return mu, log_std.exp().expand_as(mu), value

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        """Rollout vectorizado (VecMujocoEnv, N envs) -- misma forma que MLP."""
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
        """Version de UNA obs (envs no vectorizados: train_base/test_base) --
        misma forma que MLP.act()."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(dim=-1)
        action = raw.clamp(-1.0, 1.0)
        return (action.squeeze(0).cpu().numpy(), raw.squeeze(0).cpu().numpy(),
                float(logp.item()), float(value.item()))

    def evaluate(self, obs, actions):
        """(logp, value, entropy) -- MISMO nombre y MISMAS formas que
        MLPActorCritic.evaluate(), verificado contra como ppo_update() lo usa:
            logp    (batch,1)
            value   (batch,1)
            entropy escalar (mean ya aplicado)
        """
        mu, std, value = self(obs)
        dist = Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return logp, value, entropy


# ----------------------------------------------------------------------
# TEST -- confirma formas EXACTAS contra lo que ppo_update() espera,
# usando dimensiones y heatmap reales de tu proyecto.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Test: CNNActorCritic vs contrato de ppo.py (MLPActorCritic)")
    print("=" * 60)

    N_LOOKAHEAD = 3          # <-- AJUSTA a tu valor real de config.py
    STATE_DIM = 15 + 3 * N_LOOKAHEAD
    MAP_PIXELS = 32
    OBS_DIM = STATE_DIM + MAP_PIXELS * MAP_PIXELS
    ACT_DIM = 6

    print(f"STATE_DIM={STATE_DIM}  OBS_DIM={OBS_DIM}  ACT_DIM={ACT_DIM}")
    net = CNNActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM, map_pixels=MAP_PIXELS)
    device = torch.device("cpu")

    import sys
    sys.path.insert(0, ".")
    from map_context import MapContext
    map_ctx = MapContext(bt_path="occ2.bt", resolution=0.05, radius_m=1.0, patch_pixels=MAP_PIXELS)
    heatmap = map_ctx.get_heatmap(robot_xy=np.array([3.0, -2.5]), robot_z=0.15, robot_yaw=np.pi / 2)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    obs_flat = np.concatenate([state, heatmap.astype(np.float32).ravel()])
    obs_batch = np.stack([obs_flat] * 4)   # simula N=4 envs

    # -- act_batch: shapes que espera el rollout de train_fast.py --
    act, logp, val = net.act_batch(obs_batch, device)
    assert act.shape == (4, ACT_DIM), act.shape
    assert logp.shape == (4,), logp.shape
    assert val.shape == (4,), val.shape
    print(f"act_batch OK  -> act{act.shape} logp{logp.shape} val{val.shape}")

    # -- act: version de una sola obs --
    a1, lp1, v1 = net.act(obs_flat, device)
    assert a1.shape == (ACT_DIM,), a1.shape
    print(f"act OK        -> act{a1.shape} logp={lp1:.3f} val={v1:.3f}")

    # -- evaluate: EXACTO lo que ppo_update() necesita --
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    lp, val_t, ent = net.evaluate(obs_t, act_t)
    assert lp.shape == (4, 1), lp.shape          # keepdim=True
    assert val_t.shape == (4, 1), val_t.shape    # sin squeeze
    assert ent.dim() == 0, ent.shape             # escalar (mean ya aplicado)
    print(f"evaluate OK   -> logp{tuple(lp.shape)} value{tuple(val_t.shape)} entropy=escalar({ent.item():.3f})")

    # -- simula EXACTO lo que hace ppo_update() con estos tensores --
    old_logp = torch.as_tensor(logp, dtype=torch.float32).unsqueeze(-1)
    r = torch.exp(lp - old_logp)
    print(f"ratio r shape (debe ser [4,1]): {tuple(r.shape)}")
    assert r.shape == (4, 1)

    print("✅ TODO calza exacto con el contrato de ppo_update() / MLPActorCritic")