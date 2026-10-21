"""
config_legacy.py — Action semantics of the legacy base checkpoints.

The base checkpoints trained on the server (commit 50788fee,
checkpoints_base/remote_2026100*/) use 7 actions: [v, w, flipper x4, gate].
These constants describe how that action maps to flipper angles, so the
legacy policy can drive the current 6-action env (see
fsm_training/policies.legacy_to_env):

  gate >= 0 -> flipper angle = clip(a * FLIPPER_MAX, FLIPPER_MIN_RAD, FLIPPER_MAX_RAD)
  gate <  0 -> flippers at FLIPPER_HOME_RAD
"""

ACT_DIM = 7                   # [v, w, flipper x4, gate]
FLIPPER_MAX = 3.1416          # rad at action = 1
FLIPPER_MIN_RAD = -1.3
FLIPPER_MAX_RAD = 3.14159
FLIPPER_HOME_RAD = 0.0        # rest pose when the gate is off
