import numpy as np
import mujoco
from adapters.action_adapter import ActionAdapter, RobotAction


class MujocoAdapter(ActionAdapter):
    def __init__(self, model, data, track_width=1.1, wheel_radius=0.15):
        self.model        = model
        self.data         = data
        self.track_width  = track_width
        self.wheel_radius = wheel_radius
        self._cache_indices()

    def _cache_indices(self):
        """Caches ctrl indices once at initialization (avoids looking them up every step)."""
        def act_id(name):
            return mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )

        def joint_addr(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        self.ctrl_idx = {
            "wheel_left":  act_id("wheel_left_act"),
            "wheel_right": act_id("wheel_right_act"),
            **{n: act_id(f"{n}_act") for n in self.ARM_NAMES},
            **{n: act_id(f"{n}_act") for n in self.FLIPPER_NAMES},
        }
        self.arm_qpos = {n: joint_addr(n) for n in self.ARM_NAMES}

    def apply(self, action: RobotAction):
        self._apply_base(action.twist)
        self._apply_flippers(action.flippers)
        self._apply_arm(action.arm)

    def _apply_base(self, twist):
        v, w = twist.linear_x, twist.angular_z
        self.data.ctrl[self.ctrl_idx["wheel_right"]] = \
            (v + w * self.track_width / 2) / self.wheel_radius
        self.data.ctrl[self.ctrl_idx["wheel_left"]] = \
            (v - w * self.track_width / 2) / self.wheel_radius

    def _apply_flippers(self, flippers):
        for name, pos in zip(flippers.joint_names, flippers.position):
            self.data.ctrl[self.ctrl_idx[name]] = pos

    def _apply_arm(self, arm):
        for name, delta in zip(arm.joint_names, arm.position):
            current = self.data.qpos[self.arm_qpos[name]]
            target  = np.clip(current + delta, -np.pi, np.pi)
            self.data.ctrl[self.ctrl_idx[name]] = target
