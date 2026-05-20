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

TRACK_HALF_WIDTH    = 0.21
WHEEL_RADIUS        = 0.05
MAX_WHEEL_VEL       = 20.0
MAX_LINEAR_VEL      = 1.5
MAX_ANGULAR_VEL     = 2.0

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

OBS_ACTUATORS = DRIVE_LEFT + DRIVE_RIGHT + FLIPPERS

CONTROL_DECIMATION  = 10
EPISODE_MAX_STEPS   = 1000

# Ruta de checkpoints en orden de visita
RUTA_PALLETS = [
    "fatal_pallet 9",   # Spawn
    "fatal_pallet 8",
    "fatal_pallet 7",
    "fatal_pallet 6",
    "fatal_pallet 5",
    "fatal_pallet 4",
    "fatal_pallet 3",
    "fatal_pallet 2",
    "fatal_pallet 1",   # Fin rampa izquierda
    "fatal_pallet 18",  # Inicio curva de conexión
    "fatal_pallet 17",
    "fatal_pallet 16",
    "fatal_pallet 15",
    "fatal_pallet 14",
    "fatal_pallet 13",
    "fatal_pallet 12",
    "fatal_pallet 11",  # Meta final
]

ARM_REST_POSITIONS = {
    "joint_1": -0.0,
    "joint_2": -3.14,
    "joint_3":  3.14,
    "joint_4": -1.57295,
    "joint_5": -1.57295,
    "joint_6":  1.57295,
}

TIP_NAMES = [
    "wheel_flip1_front",
    "wheel_flip2_front",
    "wheel_flip3_front",
    "wheel_flip4_front",
]


class BaseMuJoCoEnv:

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

        # Solo añadir al array si el geom EXISTE, y guardar el nombre en paralelo
        # para que current_pallet_idx siempre apunte al nombre correcto.
        self.pallet_names: List[str] = []
        self.pallet_geom_ids: List[int] = []
        for name in RUTA_PALLETS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.pallet_names.append(name)
                self.pallet_geom_ids.append(gid)
            else:
                print(f"[base_env] Aviso: geom '{name}' no encontrado en el modelo — se omite de la ruta")

        # Tips de flippers (para colisiones opcionales)
        self.tip_ids: List[int] = []
        for name in TIP_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                print(f"[base_env] Aviso: body '{name}' no encontrado")
            self.tip_ids.append(bid)

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

        # ── joint_states ───────────────────────────────────────────────────
        self._obs_act_ids = np.array([self._aid(n) for n in OBS_ACTUATORS], dtype=np.int32)
        _jnt_ids          = [int(self.model.actuator_trnid[i, 0]) for i in self._obs_act_ids]
        self._qpos_adr    = np.array([self.model.jnt_qposadr[j] for j in _jnt_ids], dtype=np.int32)
        self._qvel_adr    = np.array([self.model.jnt_dofadr[j]  for j in _jnt_ids], dtype=np.int32)

        # ── base body ──────────────────────────────────────────────────────
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            self.base_id = 1

        # ── tamaños expuestos ──────────────────────────────────────────────
        self.image_shape = (3 * self.num_cameras, self.image_h, self.image_w)
        self.lidar_shape = (NUM_LIDAR,)
        self.joint_len   = 2 * len(self._obs_act_ids)
        self.joint_shape = (self.joint_len,)
        self.act_len     = 6

        # antes del primer reset()
        self._reset_state()

        # ── viewer ─────────────────────────────────────────────────────────
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = 4.0
            self.viewer.cam.elevation = -20

    def _reset_state(self):
        """Inicializa todas las variables de estado. Llamado en __init__ y reset()."""
        self._step_count          = 0
        self._stuck_counter       = 0
        self._last_base_xy        = np.zeros(2)
        self._last_flipper_action = np.zeros(4, dtype=np.float32)
        self.current_pallet_idx   = 0
        self.last_dist_to_target  = 0.0

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
            # FIX Bug 3: izq = flippers 0,2 — der = flippers 1,3
            wvel = vl if k in (0, 2) else vr
            for wid in self.ids_flip_wh.get(fid, []):
                self.data.ctrl[wid] = wvel

        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

    # ── Observaciones ──────────────────────────────────────────────────────
    def _read_cameras(self) -> np.ndarray:
        frames = []
        for cam in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = np.flip(self.renderer.render(), axis=(0, 1))
            frames.append(img.astype(np.float32) / 255.0)
        return np.transpose(np.concatenate(frames, axis=-1), (2, 0, 1))

    def _read_lidar(self) -> np.ndarray:
        lidar = np.empty(NUM_LIDAR, dtype=np.float32)
        for i, adr in enumerate(self.lidar_adr):
            d = float(self.data.sensordata[adr])
            if d <= 0.0 or d >= LIDAR_MAX:
                d = LIDAR_MAX
            lidar[i] = d / LIDAR_MAX
        return lidar

    def _read_joint_state(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self._qpos_adr],
            self.data.qvel[self._qvel_adr],
        ]).astype(np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "images":       self._read_cameras(),
            "lidar":        self._read_lidar(),
            "joint_states": self._read_joint_state(),
        }

    # ── Reward ────────────────────────────────────────────────────────────
    def _reward(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
        base_pos = self.data.xpos[self.base_id]
        base_xy  = base_pos[:2].copy()
        base_z   = float(base_pos[2])

        # 1. Caída letal
        if base_z < 0.10:
            return -100.0

        # 2. Inactividad
        move_dist = float(np.linalg.norm(base_xy - self._last_base_xy))
        self._last_base_xy = base_xy.copy()
        if move_dist < 0.005:
            self._stuck_counter += 1
            penalty_stuck = min(2.0, 0.01 * self._stuck_counter)
        else:
            self._stuck_counter = 0
            penalty_stuck = 0.0

        # 3. Obstáculos lidar
        min_lidar    = float(obs["lidar"].min())
        obstacle_pen = max(0.0, 0.0001 - min_lidar) * 5.0

        # 4. Costo de energía
        action_cost = 1e-9 * float(np.square(self.data.ctrl[self._obs_act_ids]).mean())

        # 5. Movimiento errático de flippers
        current_flipper = action[2:6].astype(np.float32)
        flipper_pen = 0.2 * float(np.square(current_flipper - self._last_flipper_action).mean())
        self._last_flipper_action = current_flipper.copy()

        # 6. Inclinación
        zmat      = self.data.xmat[self.base_id].reshape(3, 3)
        tilt_pen  = max(0.0, 0.65 - float(zmat[2, 2])) * 5.0

        # 7. Progreso hacia pallet actual
        progress_reward = 0.0
        pallet_bonus    = 0.0

        if self.current_pallet_idx < len(self.pallet_geom_ids):
            target_gid = self.pallet_geom_ids[self.current_pallet_idx]
            target_pos = self.data.geom_xpos[target_gid][:2]
            dist       = float(np.linalg.norm(base_xy - target_pos))

            delta_dist           = self.last_dist_to_target - dist
            proximity_multiplier = float(np.exp(-dist))
            progress_reward      = delta_dist * (50 + 100.0 * proximity_multiplier)
            self.last_dist_to_target = dist

            if dist < 0.45:
                pallet_name = self.pallet_names[self.current_pallet_idx]
                pallet_bonus = 50.0
                #print(f"[base_env] ✅ Pallet alcanzado: {pallet_name} "
                #      f"(idx {self.current_pallet_idx + 1}/{len(self.pallet_geom_ids)})")
                self.current_pallet_idx += 1
                #print(f"[base_env] Próximo objetivo: "
                #      f"{self.pallet_names[self.current_pallet_idx] if self.current_pallet_idx < len(self.pallet_names) else 'Ninguno, pista completada'}")

                # Actualizar distancia al siguiente pallet
                if self.current_pallet_idx < len(self.pallet_geom_ids):
                    next_pos = self.data.geom_xpos[
                        self.pallet_geom_ids[self.current_pallet_idx]
                    ][:2]
                    self.last_dist_to_target = float(np.linalg.norm(base_xy - next_pos))
                else:
                    self.last_dist_to_target = 0.0

                # Devolver solo el bonus más penalizaciones al pisar el pallet
                return pallet_bonus - penalty_stuck - obstacle_pen - action_cost - flipper_pen - tilt_pen

        return progress_reward + pallet_bonus - penalty_stuck - obstacle_pen - action_cost - flipper_pen - tilt_pen

    # ── Terminación ───────────────────────────────────────────────────────
    def _terminated(self) -> bool:
        if self._step_count >= self.max_steps:
            #print("[base_env] ⏰ Episodio terminado por límite de pasos")
            return True
        zmat = self.data.xmat[self.base_id].reshape(3, 3)
        if float(zmat[2, 2]) < 0.20:
            #print("[base_env] 🛑 Episodio terminado por caída (base demasiado inclinado)")
            return True
        if float(self.data.xpos[self.base_id, 2]) < 0.10:
            #print("[base_env] 🛑 Episodio terminado por caída (base demasiado bajo)")
            return True
        if self.current_pallet_idx >= len(self.pallet_geom_ids):
            print("[base_env] 🏆 ¡Pista completada!")
            return True
        return False

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = -1.5
        self.data.qpos[1] =  3.5
        self.data.qpos[2] =  0.2

        # Poner brazo en posición de reposo
        for joint_name, target_angle in ARM_REST_POSITIONS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = target_angle
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pos_{joint_name}"
            )
            if act_id >= 0:
                self.data.ctrl[act_id] = target_angle

        self.data.ctrl[self.id_lidar_spin] = LIDAR_SPIN_VEL

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # FIX Bug 4: un solo bloque de inicialización de estado
        self._reset_state()
        self._last_base_xy = self.data.xpos[self.base_id, :2].copy()

        # Distancia inicial al primer pallet
        if len(self.pallet_geom_ids) > 0:
            first_pos = self.data.geom_xpos[self.pallet_geom_ids[0]][:2]
            self.last_dist_to_target = float(
                np.linalg.norm(self._last_base_xy - first_pos)
            )

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
        rew  = self._reward(obs, action)
        done = self._terminated()
        return obs, rew, done, {}

    def close(self):
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
        try: self.renderer.close()
        except Exception: pass