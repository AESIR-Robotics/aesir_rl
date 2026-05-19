"""
base_env.py  —  Env MuJoCo para Modelo B (base oruga).

Observaciones — mismo formato que AesirMuJoCoEnv:
  images      : 3 cámaras RGB (cam_gripper, cam_oakd, cam_back) → (9, H, W)
  lidar       : 7 rangefinders normalizados                      → (7,)
  joint_states: qpos+qvel de los actuadores de base              → (N,)

Acciones (6 valores en [-1, 1]):
  [0]    v_lin       → capa differential drive → vel_drive_l/r_*
  [1]    ω_ang       → capa differential drive
  [2..5] flipper_1..4  posición objetivo (±π)

Las rueditas de los flippers (vel_flip*) se sincronizan automáticamente
con la velocidad del tracker al que pertenecen (izq: flippers 0,2 — der: 1,3).
vel_lidar_spin se mantiene constante, no es parte de la acción.
"""
from __future__ import annotations

import numpy as np
import mujoco
from typing import Dict, List, Tuple

# ──────────────────────────── Constantes ───────────────────────────────────
CAMERA_NAMES        = ["cam_gripper", "cam_oakd", "cam_back"]
CAMERA_H, CAMERA_W  = 84, 84
NUM_LIDAR           = 7
LIDAR_MAX           = 15.0
LIDAR_SPIN_VEL      = 20.0

TRACK_HALF_WIDTH    = 0.21    # m — ajustar midiendo en el XML
WHEEL_RADIUS        = 0.05    # m
MAX_WHEEL_VEL       = 20.0    # rad/s  (= ctrlrange de vel_drive_*)
MAX_LINEAR_VEL      = 1.5     # m/s
MAX_ANGULAR_VEL     = 2.0     # rad/s

DRIVE_LEFT   = ["vel_drive_l_1", "vel_drive_l_2", "vel_drive_l_3"]
DRIVE_RIGHT  = ["vel_drive_r_1", "vel_drive_r_2", "vel_drive_r_3"]
FLIPPERS     = ["pos_flipper_1", "pos_flipper_2", "pos_flipper_3", "pos_flipper_4"]
FLIP_WHEELS  = {
    "pos_flipper_1": ["vel_flip1_back", "vel_flip1_front"],
    "pos_flipper_2": ["vel_flip2_back", "vel_flip2_front"],
    "pos_flipper_3": ["vel_flip3_back", "vel_flip3_front"],
    "pos_flipper_4": ["vel_flip4_back", "vel_flip4_front"],
}
LIDAR_SPIN   = "vel_lidar_spin"

# Actuadores expuestos en joint_states (base + flippers, sin rueditas auxiliares)
OBS_ACTUATORS = DRIVE_LEFT + DRIVE_RIGHT + FLIPPERS

CONTROL_DECIMATION  = 10
EPISODE_MAX_STEPS   = 1000


class BaseMuJoCoEnv:
    """
    Env para Modelo B.

    La observación tiene el mismo formato dict que AesirMuJoCoEnv:
      {"images": ndarray(9,H,W), "lidar": ndarray(7,), "joint_states": ndarray(N,)}
    Así la misma red ConvActorCritic sirve aquí y en el entrenamiento conjunto.

    Acción: (6,) en [-1, 1]
      [0] v_lin  [1] ω_ang  [2..5] flipper_1..4
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
        self._stuck_counter     = 0
        self._last_x            = 0.0

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
                raise ValueError(f"Sensor lidar_{i} no encontrado en el modelo")
            self.lidar_adr.append(int(self.model.sensor_adr[sid]))
        self.id_lidar_spin = self._aid(LIDAR_SPIN)

        # ── actuadores de base ─────────────────────────────────────────────
        self.ids_drive_l  = [self._aid(n) for n in DRIVE_LEFT]
        self.ids_drive_r  = [self._aid(n) for n in DRIVE_RIGHT]
        self.ids_flippers = [self._aid(n) for n in FLIPPERS]
        self.ids_flip_wh  = {
            self._aid(fname): [self._aid(w) for w in wnames]
            for fname, wnames in FLIP_WHEELS.items()
        }

        # ── joint_states: qpos+qvel de actuadores de base ─────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)

        # ── base body ──────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── tamaños expuestos — mismos atributos que AesirMuJoCoEnv ────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (NUM_LIDAR,)
        self.joint_len   = 2 * len(self._obs_act_ids)
        self.joint_shape = (self.joint_len,)
        self.act_len     = 6

        # ── viewer ─────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 4.0
            self.viewer.cam.elevation = -20

    # ── utilidad ───────────────────────────────────────────────────────────
    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuador no encontrado: '{name}'")
        return i

    # ── Capa differential drive ────────────────────────────────────────────
    def _apply_action(self, action: np.ndarray):
        a     = np.clip(action, -1.0, 1.0)
        v_lin = float(a[0]) * MAX_LINEAR_VEL
        omega = float(a[1]) * MAX_ANGULAR_VEL

        vl = float(np.clip(
            (v_lin - omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
            -MAX_WHEEL_VEL, MAX_WHEEL_VEL
        ))
        vr = float(np.clip(
            (v_lin + omega * TRACK_HALF_WIDTH) / WHEEL_RADIUS,
            -MAX_WHEEL_VEL, MAX_WHEEL_VEL
        ))

        for i in self.ids_drive_l: self.data.ctrl[i] = vl
        for i in self.ids_drive_r: self.data.ctrl[i] = vr

        for k, fid in enumerate(self.ids_flippers):
            fp = float(np.clip(a[2 + k] * 3.1416, -3.1416, 3.1416))
            self.data.ctrl[fid] = fp
            wvel = vl if k in (0, 2) else vr
            for wid in self.ids_flip_wh.get(fid, []):
                self.data.ctrl[wid] = float(np.clip(wvel, -1.0, 1.0))

        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

    # ── Observaciones ──────────────────────────────────────────────────────
    def _read_cameras(self) -> np.ndarray:
        """→ (9, H, W) float32 en [0, 1]."""
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        """→ (7,) float32 en [0, 1]."""
        lidar = np.empty(NUM_LIDAR, dtype=np.float32)
        for i, adr in enumerate(self.lidar_adr):
            d = float(self.data.sensordata[adr])
            if d <= 0.0 or d >= LIDAR_MAX:
                d = LIDAR_MAX
            lidar[i] = d / LIDAR_MAX
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        """→ (2*N_obs_act,) float32: qpos ‖ qvel de actuadores de base."""
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
    def _reward(self, obs: Dict[str, np.ndarray]) -> float:
        x  = float(self.data.xpos[self.base_id, 0])
        dx = x - self._last_x
        self._last_x = x

        if abs(dx) < 0.005:
            self._stuck_counter += 1
            pen_inactiv = 0.05
        else:
            self._stuck_counter = 0
            pen_inactiv = 0.0

        min_lidar    = float(obs["lidar"].min())
        obstacle_pen = max(0.0, 0.1 - min_lidar) * 5.0
        ctrl_vals    = np.concatenate([
            self.data.ctrl[self.ids_drive_l + self.ids_drive_r]
        ])
        action_cost = 1e-3 * float(np.square(ctrl_vals).mean())
        alive_bonus = 0.01

        return (10.0 * dx) + alive_bonus - obstacle_pen - action_cost - pen_inactiv

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        if self._step_count >= self.max_steps: return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.2: return True
        if self._stuck_counter > 50: return True
        return False

    # ── API pública ────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = -1.5
        self.data.qpos[1] =  3.5
        self.data.qpos[2] =  0.2
        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._step_count    = 0
        self._stuck_counter = 0
        self._last_x        = float(self.data.xpos[self.base_id, 0])
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
        rew  = self._reward(obs)
        done = self._terminated()
        return obs, rew, done, {}

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass
