"""
Aesir-specific adapter.

Aesir has 3 drive wheels per side (vel_drive_l_1/2/3 and vel_drive_r_1/2/3) plus
the canonical aliases wheel_left_act/wheel_right_act used by the base
MujocoAdapter (which only drive a single wheel each). For correct propulsion
all 3 wheels on each side must receive the same velocity command, so we
override _cache_indices and _apply_base.

Track width and wheel radius defaults match the geometry baked into
aesir_scene.xml (~0.40 m and 0.10 m respectively).
"""
import numpy as np
import mujoco
from workspace.src.mujoco_package.adapters.mujoco_adapter import MujocoAdapter


class AesirMujocoAdapter(MujocoAdapter):
    LEFT_WHEEL_ACT_NAMES  = ["vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3"]
    RIGHT_WHEEL_ACT_NAMES = ["vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3"]

    def __init__(self, model, data, track_width=0.40, wheel_radius=0.10):
        super().__init__(model, data,
                         track_width=track_width,
                         wheel_radius=wheel_radius)

    def _cache_indices(self):
        """Cache ctrl indices, storing lists for the left/right wheel groups."""
        def act_id(name):
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if aid < 0:
                raise ValueError(f"Actuator '{name}' not found in model")
            return aid

        def joint_addr(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            return self.model.jnt_qposadr[jid]

        self.ctrl_idx = {
            "wheels_left":  [act_id(n) for n in self.LEFT_WHEEL_ACT_NAMES],
            "wheels_right": [act_id(n) for n in self.RIGHT_WHEEL_ACT_NAMES],
            **{n: act_id(f"{n}_act") for n in self.ARM_NAMES},
            **{n: act_id(f"{n}_act") for n in self.FLIPPER_NAMES},
        }
        self.arm_qpos = {n: joint_addr(n) for n in self.ARM_NAMES}

    def _apply_base(self, twist):
        """Differential drive: send the same wheel-spin command to every wheel
        on each side."""
        v, w = twist.linear_x, twist.angular_z
        omega_right = (v + w * self.track_width / 2.0) / self.wheel_radius
        omega_left  = (v - w * self.track_width / 2.0) / self.wheel_radius
        for idx in self.ctrl_idx["wheels_right"]:
            self.data.ctrl[idx] = omega_right
        for idx in self.ctrl_idx["wheels_left"]:
            self.data.ctrl[idx] = omega_left
