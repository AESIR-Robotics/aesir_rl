"""
policies.py — Loading and running the base and arm policies for the FSM.

Shared by the evaluation runner (tests/test_fsm.py) and the trainer
(fsm_training/train_fsm.py), so both convert actions exactly the same way.

Base: two checkpoint architectures
  - LEGACY (`actor_mu`, 7 actions = v, w, 4 flippers, gate). All base
    checkpoints trained on the server (checkpoints_base/remote_2026100*/).
    Normal distribution over the 7 dims; the network and its evaluate() come
    from base_training/ppo_cnn_extractor_legacy.py. Server env semantics
    (config_legacy): gate >= 0 -> flippers = clip(a * FLIPPER_MAX);
    gate < 0 -> flippers at rest. Translated to the current 6-dim env action
    ([v, w] in [-1,1] + flippers in [0,1]).
  - CURRENT (`actor_vw`/`actor_flip`, 6 actions, Normal + Beta).

For both, PPO must store the RAW sample returned by act_batch() (second
element): evaluate() computes the log-prob of that sample, not of the clipped
or converted action the env executes.

Arm: BetaActorCritic(28, 7) from boton_training/net.py. Its act_batch returns
the action in [-1,1] and evaluate() maps it back to (0,1), so the action is
stored.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

import rl_ws.base_training.config as C
from rl_ws.base_training import config_legacy as LC
from rl_ws.base_training.ppo import RunningMeanStd
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic
from rl_ws.base_training.ppo_cnn_extractor_legacy import CNNActorCritic as LegacyCNN
from rl_ws.boton_training.net import BetaActorCritic


def legacy_to_env(a7: np.ndarray) -> np.ndarray:
    """(B,7) or (7,) legacy action -> 6-dim action of the current env.
    Valid for both the mean and the raw Normal sample: the conversion is
    deterministic, so the raw sample's log-prob stays correct for PPO."""
    a = np.atleast_2d(np.asarray(a7, dtype=np.float64))
    rad = np.clip(a[:, 2:6] * LC.FLIPPER_MAX, LC.FLIPPER_MIN_RAD, LC.FLIPPER_MAX_RAD)
    rad = np.where(a[:, 6:7] >= 0.0, rad, LC.FLIPPER_HOME_RAD)
    u = np.clip((rad - C.FLIPPER_MIN_RAD) / (C.FLIPPER_MAX_RAD - C.FLIPPER_MIN_RAD), 0.0, 1.0)
    out = np.concatenate([np.clip(a[:, :2], -1.0, 1.0), u], axis=1).astype(np.float32)
    return out[0] if np.ndim(a7) == 1 else out


def _rms_from(d: Optional[dict]) -> Optional[RunningMeanStd]:
    if d is None:
        return None
    r = RunningMeanStd()
    r.load_state_dict(d)
    return r


class BaseAgent:
    """Wraps the base network (legacy or current) behind a common API."""

    def __init__(self, state: dict, device: torch.device, ret_rms: Optional[dict] = None,
                 log_std: Optional[float] = None):
        self.device = device
        self.legacy = "actor_mu.weight" in state
        if self.legacy:
            act_dim = int(state["log_std"].shape[0])
            if act_dim != 7:
                raise ValueError(f"legacy base with {act_dim} actions: only 7 (with gate) is supported")
            self.net = LegacyCNN(C.OBS_DIM, 7, map_pixels=C.HEATMAP_PIXELS)
        else:
            self.net = CNNActorCritic(C.OBS_DIM, C.ACT_DIM, map_pixels=C.HEATMAP_PIXELS)
        self.net.load_state_dict(state)
        self.arch = "legacy7" if self.legacy else "current6"
        self.store_dim = 7 if self.legacy else 6
        self.ret_rms = _rms_from(ret_rms)
        if log_std is not None:
            # Reset the exploration log-std. Needed for legacy checkpoints: the
            # server clamped at LOG_STD_MAX=1.0 and they arrive with log_std=1.0
            # in all 7 dims (std=2.7 on actions in [-1,1], almost bang-bang
            # noise); the workspace clamps at 0.0, so 1.0 would stay pinned at
            # the limit with no gradient.
            p = self.net.log_std if self.legacy else self.net.log_std_vw
            with torch.no_grad():
                p.fill_(float(log_std))
        self.net.to(device)

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Training rollout -> (env_action (B,6), raw_sample (B,store_dim), logp (B,), norm_value (B,))."""
        act, raw, logp, val = self.net.act_batch(obs, self.device)
        env_act = legacy_to_env(raw) if self.legacy else act
        return env_act, raw, logp, val

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for evaluation."""
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        if self.legacy:
            out = legacy_to_env(self.net(x)[0].cpu().numpy())
        else:
            params, _ = self.net(x)
            flip = params["alpha"] / (params["alpha"] + params["beta"])     # Beta mean
            out = torch.cat([params["mu_vw"].clamp(-1, 1), flip], -1).cpu().numpy()
        return out[0] if np.ndim(obs) == 1 else out

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        out = self.net(x)
        v = out[-1] if self.legacy else out[1]
        return v.squeeze(-1).cpu().numpy()


class ArmAgent:
    def __init__(self, state: dict, device: torch.device, ret_rms: Optional[dict] = None,
                 obs_dim: int = 28, act_dim: int = 7):
        if (obs_dim, act_dim) != (28, 7):
            raise ValueError(f"not a BotonArmEnv policy (obs {obs_dim}, act {act_dim})")
        self.device = device
        self.net = BetaActorCritic(28, 7)
        self.net.load_state_dict(state)
        self.net.to(device)
        self.ret_rms = _rms_from(ret_rms)
        self.store_dim = 7

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray):
        act, _, logp, val = self.net.act_batch(obs, self.device)
        return act, act, logp, val

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        dist, _ = self.net._dist(x)
        b = dist.mean.clamp(1e-6, 1.0 - 1e-6)
        out = (2.0 * b - 1.0).cpu().numpy().astype(np.float32)
        return out[0] if np.ndim(obs) == 1 else out

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        return self.net(x)[2].squeeze(-1).cpu().numpy()


def is_fsm_checkpoint(ck: dict) -> bool:
    return isinstance(ck, dict) and "base" in ck and "arm" in ck


def load_base(path: str, device: torch.device, log_std: Optional[float] = None,
              verbose: bool = True) -> Tuple[BaseAgent, dict]:
    """Accepts a base checkpoint (train_fast) or an FSM checkpoint (train_fsm)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    src = ck["base"] if is_fsm_checkpoint(ck) else ck
    agent = BaseAgent(src["policy"], device, src.get("ret_rms"), log_std=log_std)
    if verbose:
        print(f"[base] {agent.arch} iter={ck.get('iter', '?')} "
              f"avg_ep_r={src.get('avg_ep_r', '?')} ({'FSM' if is_fsm_checkpoint(ck) else 'base'})")
    return agent, src


def load_arm(path: str, device: torch.device, verbose: bool = True) -> Tuple[ArmAgent, dict]:
    """Accepts a button-arm checkpoint (train_boton) or an FSM checkpoint (train_fsm)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    src = ck["arm"] if is_fsm_checkpoint(ck) else ck
    agent = ArmAgent(src["policy"], device, src.get("ret_rms"),
                     int(src.get("obs_dim", 28)), int(src.get("act_dim", 7)))
    if verbose:
        print(f"[arm] iter={ck.get('iter', '?')} buttons={src.get('targets', '?')} "
              f"({'FSM' if is_fsm_checkpoint(ck) else 'boton'})")
    return agent, src
