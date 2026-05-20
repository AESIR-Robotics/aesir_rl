"""
arm_env.py  —  Env MuJoCo para Modelo A (brazo 6-DOF + garra).

Observaciones — mismo formato dict que los demás envs:
  images      : 3 cámaras RGB con flip → (9, H, W)
  lidar       : 7 rangefinders normalizados → (7,)
  joint_states: qpos+qvel del brazo + dedos → (16,)

Acciones (8 valores en [-1, 1]):
  [0..5]  velocidades articulares joint_1..6  (integradas a posición)
  [6]     dedo izquierdo
  [7]     dedo derecho

La base permanece quieta. El integrador de velocidad es equivalente
a MoveIt Servo en deployment real.
"""
from __future__ import annotations

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple

# ──────────────────────────── Constantes ───────────────────────────────────
CAMERA_NAMES        = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W  = 84, 84
NUM_LIDAR           = 7
LIDAR_MAX           = 15.0
LIDAR_SPIN_VEL      = 20.0

ARM_JOINTS  = ["pos_joint_1", "pos_joint_2", "pos_joint_3",
               "pos_joint_4", "pos_joint_5", "pos_joint_6"]
FINGER_L    = "pos_left_finger"
FINGER_R    = "pos_right_finger"
LIDAR_SPIN  = "vel_lidar_spin"

# Actuadores observados en joint_states (brazo + dedos)
OBS_ACTUATORS = ARM_JOINTS + [FINGER_L, FINGER_R]

MAX_JOINT_VEL = 1.0      # rad/s por articulación
JOINT_RANGE   = 3.1416

REST_ANGLES = {
    "joint_1": -0.314, "joint_2": -3.14, "joint_3": 3.14,
    "joint_4":  0.0,   "joint_5":  0.0,  "joint_6": 0.0,
}

CONTROL_DECIMATION  = 10
EPISODE_MAX_STEPS   = 500


class ArmMuJoCoEnv:
    """
    Env para Modelo A.

    Observación: dict {"images": (9,H,W), "lidar": (7,), "joint_states": (16,)}
    — mismo formato que BaseMuJoCoEnv y AesirMuJoCoEnv.

    Acción: (8,) en [-1,1]
      [0..5] vel articular joint_1..6  [6] dedo izq  [7] dedo der
    """

    def __init__(self,
                 xml_path: str,
                 camera_names: List[str] = CAMERA_NAMES,
                 image_hw: Tuple[int, int] = (CAMERA_H, CAMERA_W),
                 control_decimation: int = CONTROL_DECIMATION,
                 max_steps: int = EPISODE_MAX_STEPS,
                 render: bool = False):

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.max_steps          = max_steps
        self.control_decimation = control_decimation
        self._step_count        = 0
        self._dt                = self.model.opt.timestep * control_decimation

        # ── cámaras ────────────────────────────────────────────────────────
        self.image_h, self.image_w = image_hw
        self.renderer     = mujoco.Renderer(self.model,
                                            height=self.image_h,
                                            width=self.image_w)
        self.camera_names = list(camera_names)
        self.num_cameras  = len(self.camera_names)

        # ── lidar ──────────────────────────────────────────────────────────
        self.num_lidar = NUM_LIDAR
        self.lidar_adr = []
        for i in range(NUM_LIDAR):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            if sid < 0:
                raise ValueError(f"Sensor lidar_{i} no encontrado")
            self.lidar_adr.append(int(self.model.sensor_adr[sid]))
        self.id_lidar_spin = self._aid(LIDAR_SPIN)

        # ── actuadores del brazo ───────────────────────────────────────────
        self.ids_arm   = [self._aid(n) for n in ARM_JOINTS]
        self.id_fing_l = self._aid(FINGER_L)
        self.id_fing_r = self._aid(FINGER_R)

        # ── joint_states: qpos+qvel de brazo + dedos ──────────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)

        # ── end-effector body ──────────────────────────────────────────────
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "logitech_gripper_assembly"
        )
        if self.ee_id < 0:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")

        # ── base body ──────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── tamaños expuestos — mismos atributos que los demás envs ────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (NUM_LIDAR,)
        self.joint_len   = 2 * len(self._obs_act_ids)
        self.joint_shape = (self.joint_len,)
        self.act_len     = 8

        # ── integrador de velocidad del brazo ─────────────────────────────
        self._joint_pos = np.zeros(6, dtype=np.float64)

        # goal externo (opcional, para reward)
        self.goal_pos: Optional[np.ndarray] = None

        # ── viewer ─────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 2.0
            self.viewer.cam.elevation = -20

    # ── utilidad ───────────────────────────────────────────────────────────
    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuador no encontrado: '{name}'")
        return i

    # ── Integrador de velocidad (≡ MoveIt Servo en sim) ───────────────────
    def _apply_action(self, action: np.ndarray):
        a     = np.clip(action, -1.0, 1.0)
        delta = a[:6] * MAX_JOINT_VEL * self._dt
        self._joint_pos = np.clip(self._joint_pos + delta, -JOINT_RANGE, JOINT_RANGE)
        for k, aid in enumerate(self.ids_arm):
            self.data.ctrl[aid] = self._joint_pos[k]
        self.data.ctrl[self.id_fing_l] = float(np.clip((a[6] + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_fing_r] = float(np.clip((a[7] + 1.0) / 2.0 * 0.03, 0.0, 0.03))
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

    # ── Observaciones ──────────────────────────────────────────────────────
    def _read_cameras(self) -> np.ndarray:
        """→ (9, H, W) float32 en [0,1].
        np.flip(..., axis=(0,1)) equivale a cv2.flip(img, -1).
        """
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            img = np.flip(img, axis=(0, 1))
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        """→ (7,) float32 en [0,1]."""
        lidar = np.empty(NUM_LIDAR, dtype=np.float32)
        for i, adr in enumerate(self.lidar_adr):
            d = float(self.data.sensordata[adr])
            if d <= 0.0 or d >= LIDAR_MAX:
                d = LIDAR_MAX
            lidar[i] = d / LIDAR_MAX
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        """→ (16,) float32: qpos+qvel de joint_1..6 + dedos."""
        qpos = self.data.qpos[self._qpos_adr]
        qvel = self.data.qvel[self._qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    # ── Reward ────────────────────────────────────────────────────────────
    def _reward(self) -> float:
        # Minimizar distancia EE → goal si existe
        if self.goal_pos is not None and self.ee_id >= 0:
            dist = float(np.linalg.norm(self.data.xpos[self.ee_id] - self.goal_pos))
            rew  = -2.0 * dist + 0.01
        else:
            rew = 0.01

        # action_cost sobre todos los actuadores del brazo
        action_cost = 1e-3 * float(np.square(self.data.ctrl[self._obs_act_ids]).mean())
        return rew - action_cost

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        return self._step_count >= self.max_steps

    # ── API pública ────────────────────────────────────────────────────────
    def reset(self, keep_base: bool = True) -> Dict[str, np.ndarray]:
        if keep_base:
            base_qpos = self.data.qpos[:7].copy()
            base_qvel = self.data.qvel[:6].copy()
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = base_qpos
            self.data.qvel[:6] = base_qvel
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0] = -1.5
            self.data.qpos[1] =  3.5
            self.data.qpos[2] =  0.2

        for nombre, angulo in REST_ANGLES.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angulo

        self._joint_pos = np.array([REST_ANGLES[f"joint_{i+1}"] for i in range(6)])
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

        return self._observation()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        self._step_count += 1
        obs  = self._observation()
        rew  = self._reward()
        done = self._terminated()
        return obs, rew, done, {}

    def set_goal(self, goal_xyz: np.ndarray):
        self.goal_pos = np.array(goal_xyz, dtype=np.float64)

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass