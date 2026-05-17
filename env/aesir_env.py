"""
Aesir environment: MuJoCo + AesirMujocoAdapter + door-traversal reward.

State (74 dims):
    qpos       (29) - generalized coordinates
    qvel       (28) - generalized velocities
    sensordata (17) - 1 long-range rangefinder + 16 short-range LiDAR rays

Action (12 dims, in [-1, 1]):
    Routed through ActionAdapter.from_policy_output() then applied via
    AesirMujocoAdapter.apply().
"""
import os
import numpy as np
import mujoco
import mediapy as media

from adapters.aesir_adapter import AesirMujocoAdapter


MUJOCO_STEPS = 5

# Door geometry baked into aesir_scene.xml
DOOR_X       = 3.0    # door hinge position along +X
CROSSED_X    = 4.0    # robot considered "past the door" beyond this
FALLEN_Z     = 0.05   # if base drops below this, robot has fallen

# LiDAR clipping (rangefinder returns -1 on miss)
LIDAR_MAX = 10.0

# Index of door-hinge joint in qpos
DOOR_HINGE_QPOS = 27


def _yaw_from_quat(qw, qx, qy, qz):
    """Yaw from a MuJoCo (w,x,y,z) quaternion."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class Env:
    """RL environment for Aesir robot navigating through a hinged door."""

    def __init__(self,
                 scene_path: str = None,
                 render_camera: str = "cam_chase"):
        if scene_path is None:
            scene_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "aesir_scene.xml",
            )
        self.model    = mujoco.MjModel.from_xml_path(scene_path)
        self.data     = mujoco.MjData(self.model)
        # Renderer is created lazily on first render so the Env can be built
        # even on hosts without an OpenGL context.
        self.renderer = None

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Adapter handles all ctrl writes from now on
        self.adapter = AesirMujocoAdapter(self.model, self.data)

        # Simulation params (kept consistent with the ANYmal base script)
        self.FRAMERATE = 60
        self.DURATION  = 8
        self.TIMESTEP  = 0.002
        self.model.opt.timestep = self.TIMESTEP

        # Rendering
        self.render_camera_name = render_camera
        self.render_camera_id   = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, render_camera
        )
        if self.render_camera_id < 0:
            raise ValueError(f"Camera '{render_camera}' not in scene")

        self.frames = []
        self.done   = False

        # Sizes useful to the trainer
        self.obs_len = self.model.nq + self.model.nv + self.model.nsensordata
        self.act_len = 12

        # Internal trackers for the reward
        self._prev_door_dist = None
        self._max_x          = 0.0

    # ------------------------------------------------------------------ #
    #                              helpers                               #
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        """Concatenate qpos, qvel and lidar readings into one flat vector."""
        sensor = self.data.sensordata.copy()
        # rangefinder returns -1 when no hit -> replace with LIDAR_MAX
        sensor[sensor < 0] = LIDAR_MAX
        sensor = np.clip(sensor, 0.0, LIDAR_MAX)
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            sensor,
        ]).astype(np.float64)

    def _door_dist(self) -> float:
        """Horizontal distance from the base to the door plane (x=DOOR_X)."""
        return abs(DOOR_X - self.data.qpos[0])

    def _render_one_frame(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
        self.renderer.update_scene(self.data, camera=self.render_camera_id)
        pixels = self.renderer.render()
        self.frames.append(pixels.copy())

    # ------------------------------------------------------------------ #
    #                              gym API                               #
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._prev_door_dist = self._door_dist()
        self._max_x          = float(self.data.qpos[0])
        self.frames.clear()
        self.done = False
        return self._get_state()

    def step(self, action: np.ndarray, render: bool = False):
        """
        Args:
            action: array of shape (12,) with values in [-1, 1]
            render: capture frames at FRAMERATE
        """
        self.done = False
        reward   = 0.0

        # ---- action via adapter ----------------------------------------
        # action may arrive as (1, 12) from compute_action; flatten
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        robot_action = self.adapter.from_policy_output(action)
        self.adapter.apply(robot_action)

        # ---- physics + reward shaping ----------------------------------
        for _ in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)

            x      = self.data.qpos[0]
            vx     = self.data.qvel[0]
            cur_dd = self._door_dist()
            door_ang = abs(self.data.qpos[DOOR_HINGE_QPOS])

            # Progress toward the door (BEFORE crossing)
            if x < DOOR_X:
                progress = (self._prev_door_dist - cur_dd)
                reward  += 5.0 * progress
            else:
                # After crossing, reward forward velocity instead
                reward  += 2.0 * max(vx, 0.0)

            # Encourage opening the door
            reward += 0.5 * door_ang

            # Update trackers
            self._prev_door_dist = cur_dd
            if x > self._max_x:
                self._max_x = x

            # Capture video frames
            if render and (len(self.frames) < self.data.time * self.FRAMERATE):
                self._render_one_frame()

        # ---- per-step (not per substep) shaping ------------------------
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = _yaw_from_quat(qw, qx, qy, qz)
        # Heading: face +X
        reward -= 0.02 * (1.0 - np.cos(yaw))
        # Light energy / shake penalty
        reward -= 1e-5 * np.sum(np.square(self.data.qvel))

        # ---- terminal conditions ---------------------------------------
        x = float(self.data.qpos[0])
        z = float(self.data.qpos[2])
        qw, qx, qy, qz = self.data.qpos[3:7]
        # Tilt: angle between body +Z and world +Z
        # body_z_world = R . [0,0,1] = (2(xz+wy), 2(yz-wx), 1-2(x^2+y^2))
        body_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_ok  = body_z_z > 0.3   # less than ~72 deg from upright

        if self.data.time > self.DURATION:
            self.done = True
        if z < FALLEN_Z or not tilt_ok:
            self.done = True
            reward  -= 100.0
        if x > CROSSED_X:
            self.done = True
            reward  += 200.0    # success bonus

        return self._get_state(), float(reward), self.done

    def close(self, episode, ep_reward):
        path = f"./video_{episode}_{ep_reward:.2f}.mp4"
        if len(self.frames) > 0:
            media.write_video(path, self.frames, fps=self.FRAMERATE)
        return path
