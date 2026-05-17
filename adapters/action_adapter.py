from dataclasses import dataclass
import numpy as np

@dataclass
class TwistCmd:
    linear_x:  float
    angular_z: float

@dataclass
class JointCmd:
    joint_names: list
    position:    np.ndarray
    velocity:    np.ndarray

@dataclass
class RobotAction:
    twist:    TwistCmd
    arm:      JointCmd
    flippers: JointCmd


class ActionAdapter:
    # Real physical limits of the robot
    MAX_LINEAR_VEL  = 0.5   # m/s
    MAX_ANGULAR_VEL = 1.0   # rad/s
    MAX_FLIPPER_POS = np.pi # rad
    ARM_DELTA_MAX   = 0.1   # rad per step

    ARM_NAMES     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    FLIPPER_NAMES = ["flipper_0", "flipper_1", "flipper_2", "flipper_3"]

    def from_policy_output(self, raw: np.ndarray) -> RobotAction:
        """
        raw: np.ndarray shape (12,), values in [-1, 1]

        Layout:
          raw[0]    -> linear_x
          raw[1]    -> angular_z
          raw[2:6]  -> flippers (absolute position)
          raw[6:12] -> arm (position delta)
        """
        twist = TwistCmd(
            linear_x  = float(raw[0]) * self.MAX_LINEAR_VEL,
            angular_z = float(raw[1]) * self.MAX_ANGULAR_VEL,
        )

        flippers = JointCmd(
            joint_names = self.FLIPPER_NAMES,
            position    = raw[2:6] * self.MAX_FLIPPER_POS,  # [-pi, pi]
            velocity    = np.full(4, 1.5),                  # fixed (temporary)
        )

        arm = JointCmd(
            joint_names = self.ARM_NAMES,
            position    = raw[6:12] * self.ARM_DELTA_MAX,   # delta
            velocity    = np.full(6, 0.5),
        )

        return RobotAction(twist=twist, arm=arm, flippers=flippers)

    def apply(self, action: RobotAction):
        raise NotImplementedError
