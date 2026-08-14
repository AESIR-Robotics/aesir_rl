#!/usr/bin/env python3
"""
mujoco_sim_base.py — Backend de entrenamiento RAPIDO: MuJoCo DIRECTO (sin ROS).

BaseMujocoEnv es dueño de su propio (MjModel, MjData) y avanza la fisica sin
tiempo real ni bridge — corre a la velocidad de la CPU (~2000 pasos/s), no a los
~40/s del lazo ROS. La accion [v, ω, flipper×4] se aplica por la MISMA interfaz de actuadores que el
bridge — cinematica diferencial + rampas AVR446 (robot_control.RampController) —
asi que la respuesta dinamica es identica y la politica es desplegable sin gap.
Todas las constantes (rutas, decimacion, brazo, escalas) vienen de config.py.

VecMujocoEnv corre N envs en THREADS. MuJoCo libera el GIL en mj_step, asi que
los threads dan paralelismo real (medido ~5x con 6 envs) sin la complejidad de
multiprocessing (memoria compartida, sin pickling, sin fork).

Uso tipico (desde train_fast.py):
    venv = VecMujocoEnv(n_envs=6)
    obs = venv.reset()                       # (N, OBS_DIM)
    obs, rew, done, info = venv.step(acts)   # acts (N, ACT_DIM)
"""
from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import cv2
except Exception:  # pragma: no cover - fallback en entornos sin OpenCV
    cv2 = None

import rl_ws.base_training.config as C
import rl_ws.base_training.base_env as W
from rl_ws.base_training.robot_control import RampController
from rl_ws.base_training.map_context import MapContext
from rl_ws.global_navigator import (
    build_platform_zone, plan_platform_route, plan_platform_route_with_obstacle,
    plan_route, plan_track_route, GlobalNavigator, quat_to_yaw, quat_to_grav_body,
)

CHECKOUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _quat_upright(q) -> float:
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


class BaseMujocoEnv:
    """Un env MuJoCo directo. Misma tarea/obs/reward/accion que BaseRosEnv.

    `track` = entrada de C.TRACK_DEFS (kind/xml/bt/spawn_z/settle_steps[/nav_json]).
      kind="platform": mision procedural (spawn/meta random + obstaculo virtual).
      kind="pallets":  mision clasica del JSON (A* sobre pallets, spawn/meta fijos)
                       — la implementacion original de la pista de pallets."""

    def __init__(self, waypoints, model: mujoco.MjModel = None,
                 goal_xy=None, control_decimation: int = C.CONTROL_DECIMATION,
                 max_steps: int = C.EPISODE_MAX_STEPS,
                 map_ctx: Optional[MapContext] = None,
                 platform_zone=None,
                 track: dict = None):
        # Un modelo por env (mismo XML). VecMujocoEnv le pasa una COPIA propia
        # de MjModel a cada uno (ver VecMujocoEnv.__init__) porque el
        # obstaculo virtual fisico reescribe geom_pos/geom_size por env en
        # cada reset() -- si el MjModel viniera compartido por referencia,
        # los envs se pisarian el mismo geom entre threads.
        self.track = track if track is not None else C.TRACK_DEFS[C.ACTIVE_TRACKS[0]]
        self.track_name = self.track.get("name", "?")
        self._kind = self.track.get("kind", "platform")
        self._spawn_z = float(self.track.get("spawn_z", C.SPAWN_Z))
        self._settle = int(self.track.get("settle_steps", C.SPAWN_SETTLE_STEPS))
        self._fall_z_min = float(self.track.get("fall_z_min", C.FALL_Z_MIN))
        self._finish_dist = float(self.track.get("finish_dist", C.FINISH_DIST))
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(self.track["xml"])
        self.data  = mujoco.MjData(self.model)
        self.decim = control_decimation
        self.max_steps = max_steps
        self._dt = float(self.model.opt.timestep)

        self.map_ctx = map_ctx
        self._platform_zone = platform_zone
        if self._platform_zone is None and self._kind == "platform":
            self._platform_zone = build_platform_zone()

        m = self.model
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

        # Control de tracks (velocidad) + flippers (posicion) con las MISMAS
        # rampas AVR446 que el bridge de despliegue -> misma respuesta dinamica.
        self.ctrl = RampController(m, self.data)
        # Actuadores de posicion del brazo (parado en reposo, no lo controla RL).
        self._a_arm = [aid(n) for n in C.ARM_ACTUATORS]

        # qpos/qvel adr: chasis (freejoint), flippers, brazo.
        base_j = jid("base_freejoint")
        self._base_qadr = int(m.jnt_qposadr[base_j])
        self._base_vadr = int(m.jnt_dofadr[base_j])
        self._flip_qadr = [int(m.jnt_qposadr[jid(n)]) for n in C.FLIPPER_JOINTS]
        self._flip_vadr = [int(m.jnt_dofadr[jid(n)])  for n in C.FLIPPER_JOINTS]
        self._arm_qadr  = [int(m.jnt_qposadr[jid(n)]) for n in C.ARM_JOINTS]
        self._arm_vadr  = [int(m.jnt_dofadr[jid(n)])  for n in C.ARM_JOINTS]

        # Geoms para deteccion de contacto robot<->piso (igual que el bridge).
        # "maze_floor" es el piso caminable de la pista maze (a diferencia del
        # resto de pistas, donde el unico plane es el piso fatal debajo de la
        # plataforma/pallets) 
        self._floor_gids = {g for g in range(m.ngeom)
                            if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE
                            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") != "maze_floor"}
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "footprint_link")
        self._robot_gids = {g for g in range(m.ngeom)
                            if int(m.body_rootid[m.geom_bodyid[g]]) == root}

        # Geom del obstaculo virtual FISICO (plataform.xml)
        self._obstacle_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "virtual_obstacle")
        self._obstacle_gids = {self._obstacle_gid} if self._obstacle_gid >= 0 else set()
        # + los postes "fatal_stick N" de la pista de pallets (18 cajas finas de
        # 4.8x10x90 cm que hay que esquivar). ANTES no estaban aqui: como
        # `virtual_obstacle` solo existe en plataform.xml, en pallets el set
        # quedaba VACIO y chocar con un poste costaba CERO -- ni penalizacion ni
        # terminacion, solo el atasco indirecto a los 300 pasos. Con WP_BONUS=200
        # y GOAL_BONUS=1000 en juego, eso hacia rentable rozarlos en vez de
        # rodearlos. Solo existen en aesir_pallets.xml, asi que no hace falta
        # condicionar por pista: en las demas el conjunto sale vacio igual.
        self._obstacle_gids |= {
            g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("fatal_stick")}

        # Mision + navegador segun el tipo de pista.
        if self._kind in ("pallets", "maze"):
            # MISMO planificador para las dos (global_navigator.TrackMap): A*
            # consciente de la huella sobre (x, y, rumbo). Lo unico que cambia es
            # de donde salen las cajas, y de eso se encarga el extractor que
            # elige get_track_map segun el `kind`.
            #
            # La meta se SORTEA en zona alcanzable y el episodio termina al
            # llegar -> el siguiente reset sortea otra. Para pallets esto es un
            # cambio: antes la mision era FIJA (mismos 74 waypoints, 18.1 m,
            # identicos en cada episodio), y esa era la razon de que nunca
            # despegara del 0% entrenando en multipista -- sin rutas cortas no
            # hay de donde bootstrapear el GOAL_BONUS. Ver docs/sigma_estado.md.
            spawn = self.track.get("spawn_xy", C.START_XY)
            waypoints, goal_xy = plan_track_route(
                spawn, goal_xy, track=self.track,
                start_yaw=float(self.track.get("spawn_yaw", 0.0)))
            self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
            self._fixed_waypoints = waypoints
            # El navegador de pallets necesita el JSON para los bordes/obstaculos
            # del vortex (repulsion fina); el maze no tiene JSON.
            self.nav = GlobalNavigator(self.track.get("nav_json"), waypoints=waypoints,
                                       n_lookahead=C.N_LOOKAHEAD)
        else:
            # Plataforma: goal_xy inicial solo para la primera construccion --
            # el primer reset() ya sortea uno random y replanifica.
            if goal_xy is None:
                goal_xy = C.GOAL_XY or (0.0, 0.0)
            self.goal_xy = np.array(goal_xy, dtype=np.float64)
            if waypoints is None:
                waypoints = plan_platform_route(C.START_XY, tuple(self.goal_xy))
            self.nav = GlobalNavigator(None, waypoints=waypoints,
                                       n_lookahead=C.N_LOOKAHEAD,
                                       edges_zone=self._platform_zone)

        self._rs = W.RewardState()
        self._ep_steps = 0
        self._heatmap_out_dir = os.path.join(CHECKOUT_ROOT, "heatmap_debug")
        self._heatmap_counter = 0
        self._heatmap_video_path = os.path.join(self._heatmap_out_dir, "heatmap_live.mp4")
        self._heatmap_video_writer = None
        self._heatmap_video_fps = 6
        os.makedirs(self._heatmap_out_dir, exist_ok=True)

    # ── Feedback desde mjData (mismo dict `fb` que el backend ROS) ───────────
    def _feedback(self) -> dict:
        a, v = self._base_qadr, self._base_vadr
        pos  = self.data.qpos[a:a + 3]
        quat = self.data.qpos[a + 3:a + 7]

        lin_world = self.data.qvel[v:v + 3].copy()
        ang_body  = self.data.qvel[v + 3:v + 6].copy()
        quat_inv  = np.zeros(4); mujoco.mju_negQuat(quat_inv, quat)
        lin_body  = np.zeros(3); mujoco.mju_rotVecQuat(lin_body, lin_world, quat_inv)

        flip_qpos = np.array([self.data.qpos[i] for i in self._flip_qadr], dtype=np.float32)
        flip_qvel = np.array([self.data.qvel[i] for i in self._flip_vadr], dtype=np.float32)

        return dict(
            xy=pos[:2].copy().astype(np.float64), z=float(pos[2]),
            yaw=quat_to_yaw(quat), upright=_quat_upright(quat),
            grav_body=quat_to_grav_body(quat),
            twist=np.array([lin_body[0], lin_body[1], ang_body[2]], dtype=np.float32),
            flip_qpos=flip_qpos, flip_qvel=flip_qvel,
            floor_contact=self._count_floor_contacts(),
            obstacle_contact=self._count_obstacle_contacts(),
            fall_z_min=self._fall_z_min, finish_dist=self._finish_dist,
        )

    def _count_floor_contacts(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self._floor_gids and g2 in self._robot_gids) or
                    (g2 in self._floor_gids and g1 in self._robot_gids)):
                n += 1
        return n

    def _count_obstacle_contacts(self) -> int:
        """Contactos robot<->caja fisica del obstaculo virtual (no letal, a
        diferencia de _count_floor_contacts: solo alimenta la penalizacion de
        colision en compute_reward, no termina el episodio)."""
        if not self._obstacle_gids:
            return 0
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self._obstacle_gids and g2 in self._robot_gids) or
                    (g2 in self._obstacle_gids and g1 in self._robot_gids)):
                n += 1
        return n

    # ── Fijar objetivos de la accion (las rampas los persiguen en apply) ────
    def _apply_action(self, action: np.ndarray):
        self.ctrl.set_base_twist(float(action[0]) * C.V_MAX_MPS, float(action[1]) * C.W_MAX_RADPS)
        flip_rad = W.flipper_targets(action)
        if flip_rad is not None:
            self.ctrl.set_flippers(flip_rad)

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        a = self._base_qadr
        if self._kind in ("pallets", "maze"):
            # Spawn FIJO (entrada del laberinto / START_XY de pallets, los mismos
            # puntos que usa el bridge ROS). Lo que cambia entre episodios es la
            # meta, no el punto de partida.
            spawn_xy = np.array(self.track.get("spawn_xy", C.START_XY), dtype=float)
            spawn_yaw = float(self.track.get("spawn_yaw", 0.0))
        else:
            # Plataforma: spawn random en ±SPAWN_XY_RANGE con yaw random.
            spawn_xy = np.random.uniform(-C.SPAWN_XY_RANGE, C.SPAWN_XY_RANGE, 2)
            spawn_yaw = np.random.uniform(-np.pi, np.pi)
        self.data.qpos[a + 0] = spawn_xy[0]
        self.data.qpos[a + 1] = spawn_xy[1]
        self.data.qpos[a + 2] = self._spawn_z          # altura segun el techo de la pista
        self.data.qpos[a + 3:a + 7] = [np.cos(spawn_yaw / 2.0), 0.0, 0.0, np.sin(spawn_yaw / 2.0)]
        # Brazo en reposo (qpos + ctrl del actuador de posicion).
        for name, qadr, act in zip(C.ARM_JOINTS, self._arm_qadr, self._a_arm):
            self.data.qpos[qadr] = C.ARM_REST_POSE[name]
            self.data.ctrl[act]  = C.ARM_REST_POSE[name]
        mujoco.mj_forward(self.model, self.data)
        self.ctrl.sync_from_data()          # rampas al estado nuevo (flippers en 0, vel 0)
        self._grav_comp_arm()
        for _ in range(self._settle):
            self.ctrl.apply(self._dt)
            mujoco.mj_step(self.model, self.data)

        self._ep_steps = 0
        fb = self._feedback()

        if self._kind == "platform":
            # Nueva mision cada episodio: goal random + ruta que RODEA el
            # obstaculo virtual (fisico, reposicionado abajo).
            goal_xy = np.random.uniform(-C.GOAL_XY_RANGE, C.GOAL_XY_RANGE, 2)
            self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
            if C.USE_VIRTUAL_OBSTACLE:
                # El obstaculo se coloca primero (random sobre el paso directo) y
                # luego se planea TENIENDOLO EN CUENTA. La ruta la da el mismo
                # planificador que maze/pallets, con la caja como OVERLAY
                # dinamico: no se puede hornear en las mascaras porque cambia en
                # cada episodio y rehacerlas cuesta ~1 s.
                _, vobs = plan_platform_route_with_obstacle(tuple(fb["xy"]), tuple(goal_xy))
                waypoints, goal_xy = plan_track_route(
                    fb["xy"], goal_xy, track=self.track, start_yaw=float(fb["yaw"]),
                    overlay=([vobs.bounds()] if vobs is not None else []))
            else:
                waypoints, vobs = plan_platform_route(tuple(fb["xy"]), tuple(goal_xy)), None
            self.nav.replan(waypoints, obstacles=[vobs] if vobs is not None else [])

            if self._obstacle_gid >= 0:
                if vobs is not None:
                    self.model.geom_pos[self._obstacle_gid] = [
                        vobs.x, vobs.y, C.PLATFORM_SURFACE_Z + C.VIRTUAL_OBSTACLE_HEIGHT_HALF]
                    self.model.geom_size[self._obstacle_gid] = [
                        vobs.hx, vobs.hy, C.VIRTUAL_OBSTACLE_HEIGHT_HALF]
                else:
                    self.model.geom_pos[self._obstacle_gid, 2] = -5.0   # fuera de juego
                mujoco.mj_forward(self.model, self.data)
        elif self._kind in ("pallets", "maze"):
            # Meta NUEVA en cada episodio, sorteada en zona libre y ALCANZABLE
            # desde donde esta el robot; la ruta la da el A* consciente de la
            # huella, no la recta (que atravesaria paredes o el vacio entre
            # tarimas). Guardar la meta es obligatorio: de ella dependen la
            # condicion de exito y GOAL_BONUS.
            waypoints, goal_xy = plan_track_route(fb["xy"], track=self.track,
                                                  start_yaw=float(fb["yaw"]))
            self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
            self.nav.replan(waypoints, obstacles=[])

        self.nav.reset(fb["xy"])
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        self._rs.reset(fb["xy"], float(np.linalg.norm(fb["xy"] - np.asarray(guidance["target"]))), fb["yaw"])
        heatmap = self._get_heatmap(fb)
        return W.build_obs(guidance, fb, heatmap)

    def _grav_comp_arm(self):
        # Compensa gravedad en los DOFs del brazo (como el bridge) para que el
        # brazo parado no cargue ni derive.
        self.data.qfrc_applied[self._arm_vadr] = self.data.qfrc_bias[self._arm_vadr]

    def _get_heatmap(self, fb: dict):
        """Minimapa por altura real, recalculado con la posicion ACTUAL del
        robot (fb["xy"], fb["z"], fb["yaw"]) -- None si no hay map_ctx."""
        if self.map_ctx is None:
            return None
        return self.map_ctx.get_heatmap(robot_xy=fb["xy"], robot_z=fb["z"], robot_yaw=fb["yaw"])

    def _close_heatmap_video(self):
        """Finaliza el writer de video si está abierto."""
        if self._heatmap_video_writer is not None:
            try:
                self._heatmap_video_writer.release()
            except Exception:
                pass
            self._heatmap_video_writer = None

    def _append_heatmap_to_video(self, heatmap: np.ndarray):
        """Agrega el heatmap actual a un MP4 en tiempo real si es posible."""
        if heatmap is None or cv2 is None:
            return
        try:
            frame = np.asarray(heatmap, dtype=np.float32)
            frame = np.clip(frame, 0.0, 1.0)
            gray = np.uint8(frame * 255.0)
            frame_bgr = cv2.merge([gray, gray, gray])
            if self._heatmap_video_writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._heatmap_video_writer = cv2.VideoWriter(self._heatmap_video_path, fourcc,
                                                             self._heatmap_video_fps, (w, h))
                if not self._heatmap_video_writer.isOpened():
                    self._heatmap_video_writer = None
                    return
            self._heatmap_video_writer.write(frame_bgr)
        except Exception:
            # No interrumpir el entrenamiento si el backend de video no está disponible.
            self._close_heatmap_video()

    def _save_heatmap_debug(self, heatmap: np.ndarray, fb: dict):
        """Guarda una imagen del heatmap actual en disco para inspeccionarlo."""
        if heatmap is None:
            return
        out_dir = self._heatmap_out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._heatmap_counter += 1
        out_path = os.path.join(out_dir, f"heatmap_step_{self._heatmap_counter:05d}.png")
        dpi = 240
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(heatmap, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"step={self._heatmap_counter}  xy=({fb['xy'][0]:.2f},{fb['xy'][1]:.2f})",
                     fontsize=10, pad=4)
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, format="png")
        plt.close(fig)
        self._append_heatmap_to_video(heatmap)

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        a_exec = np.clip(action, -1.0, 1.0)
        self._apply_action(a_exec)
        # grav-comp del brazo una vez por control-step (qfrc_applied persiste).
        self._grav_comp_arm()
        # Rampas AVR446 por substep (igual que el bridge) + fisica.
        for _ in range(self.decim):
            self.ctrl.apply(self._dt)
            mujoco.mj_step(self.model, self.data)

        self._ep_steps += 1
        fb = self._feedback()
        fb["flipper_edge"] = W.flipper_edge(self.map_ctx, fb)
        guidance = self.nav.step(fb["xy"], fb["yaw"])
        reward = W.compute_reward(fb, guidance, a_exec, self._rs, self.goal_xy)
        done, reached, reason = W.terminated(fb, self.goal_xy, self._ep_steps, self.max_steps, self._rs)
        heatmap = self._get_heatmap(fb)
        if C.SAVE_HEATMAP_DEBUG:
            self._save_heatmap_debug(heatmap, fb)
        obs = W.build_obs(guidance, fb, heatmap)
        info = {"wp": guidance["wp"], "reached": reached, "reason": reason,
                "dist_goal": float(np.linalg.norm(fb["xy"] - self.goal_xy)),
                "track": self.track_name}
        return obs, reward, done, info


# ──────────────────────────── Vec-env por threads ──────────────────────────
class VecMujocoEnv:
    """N BaseMujocoEnv en paralelo (threads; mj_step libera el GIL). Con
    auto-reset: cuando un env termina, su siguiente obs es la del episodio nuevo
    (convencion gym/SB3); el reward/done terminal se devuelven igual para el
    buffer de PPO."""

    def __init__(self, n_envs: int = C.N_ENVS, control_decimation: int = C.CONTROL_DECIMATION,
                 max_steps: int = C.EPISODE_MAX_STEPS, verbose: bool = True,
                 tracks: Optional[List[str]] = None):
        """tracks: nombres de C.TRACK_DEFS a entrenar (default C.ACTIVE_TRACKS).
        Con VARIAS pistas se reparten entre los envs round-robin: cada env vive
        toda la corrida en SU pista (cargar el XML es caro; episodios nuevos ya
        randomizan spawn/meta/obstaculo dentro de la pista)."""
        self.n = n_envs
        self.obs_dim = C.OBS_DIM
        self.act_dim = C.ACT_DIM
        self.verbose = verbose
        self.tracks = list(tracks) if tracks else list(C.ACTIVE_TRACKS)
        for t in self.tracks:
            if t not in C.TRACK_DEFS:
                raise ValueError(f"pista desconocida '{t}' (validas: {sorted(C.TRACK_DEFS)})")

        # Recursos por pista, compartidos (read-only) entre los envs de esa
        # pista: MjModel base (cada env recibe su copia), MapContext (heatmap)
        # y zona segura de plataforma.
        self._track_res: Dict[str, dict] = {}
        for t in self.tracks:
            td = dict(C.TRACK_DEFS[t]); td["name"] = t
            res = dict(td=td)
            res["base_model"] = mujoco.MjModel.from_xml_path(td["xml"])
            res["map_ctx"] = None
            if getattr(C, "USE_HEATMAP", False):
                res["map_ctx"] = MapContext(
                    bt_path=td["bt"], resolution=C.OCTOMAP_RESOLUTION,
                    radius_m=C.HEATMAP_RADIUS_M, patch_pixels=C.HEATMAP_PIXELS,
                    z_range_m=C.HEATMAP_Z_RANGE_M)
            res["platform_zone"] = build_platform_zone() if td["kind"] == "platform" else None
            self._track_res[t] = res

        env_tracks = [self.tracks[i % len(self.tracks)] for i in range(n_envs)]
        if verbose:
            counts = {t: env_tracks.count(t) for t in self.tracks}
            print(f"[vec] n_envs={n_envs} | pistas={counts} | "
                  f"flippers={'ON' if C.CONTROL_FLIPPERS else 'OFF'}")

        self.envs = []
        for i, t in enumerate(env_tracks):
            res = self._track_res[t]
            self.envs.append(BaseMujocoEnv(
                None, model=copy.deepcopy(res["base_model"]),
                control_decimation=control_decimation, max_steps=max_steps,
                map_ctx=res["map_ctx"], platform_zone=res["platform_zone"],
                track=res["td"]))
        self._pool = ThreadPoolExecutor(max_workers=n_envs)
        self._ep_return = np.zeros(n_envs, dtype=np.float64)
        self._ep_history: List[float] = []
        self._success_history: List[bool] = []
        # Success rate por pista: con varias pistas activas la tasa global
        # esconde que una pista facil compense a una dificil (o al reves).
        # Ventana propia por pista, no compartida.
        self._success_by_track: Dict[str, List[bool]] = {t: [] for t in self.tracks}
        self._n_reached = 0

    def reset(self) -> np.ndarray:
        obs = list(self._pool.map(lambda e: e.reset(), self.envs))
        self._ep_return[:] = 0.0
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        def _one(i):
            return self.envs[i].step(actions[i])
        results = list(self._pool.map(_one, range(self.n)))

        obs   = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        rews  = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        infos: List[dict] = []
        for i, (o, r, d, inf) in enumerate(results):
            self._ep_return[i] += r
            if d:
                self._ep_history.append(float(self._ep_return[i]))
                if len(self._ep_history) > 100:
                    self._ep_history.pop(0)
                # Success rate movil: mismo criterio que avg_return (ultimos 100
                # episodios TERMINADOS, no por iteracion) -- con ~17 episodios
                # por iteracion la tasa cruda salta demasiado para leerla.
                won = bool(inf.get("reached"))
                self._success_history.append(won)
                if len(self._success_history) > 100:
                    self._success_history.pop(0)
                by_tr = self._success_by_track.get(inf.get("track"))
                if by_tr is not None:
                    by_tr.append(won)
                    if len(by_tr) > 100:
                        by_tr.pop(0)
                if inf.get("reached"):
                    self._n_reached += 1
                if self.verbose and inf.get("reason"):
                    tag = "🏁" if inf.get("reached") else "🛑"
                    print(f"[vec] env{i}[{inf.get('track','?')}] {tag} {inf['reason']}  "
                          f"ret={self._ep_return[i]:.1f}  wp={inf.get('wp')}")
                self._ep_return[i] = 0.0
                inf["terminal_obs"] = o
                reason = inf.get("reason") or ""
                inf["truncated"] = reason.startswith("limite") or reason.startswith("atascado")
                o = self.envs[i].reset()      # auto-reset
            obs[i]  = o
            rews[i] = r
            dones[i] = float(d)
            infos.append(inf)
        return obs, rews, dones, infos

    def avg_return(self) -> float:
        return float(np.mean(self._ep_history)) if self._ep_history else float("nan")

    def success_rate(self) -> float:
        """Fraccion de los ultimos 100 episodios TERMINADOS que llegaron a la
        meta. Es la metrica de tarea: n_reached es acumulado (siempre sube, no
        dice si esta mejorando) y avg_return mezcla progreso con castigos."""
        return float(np.mean(self._success_history)) if self._success_history else float("nan")

    def success_rate_by_track(self) -> Dict[str, float]:
        """{pista: success rate de SUS ultimos 100 episodios}. nan mientras esa
        pista no haya terminado ninguno todavia."""
        return {t: (float(np.mean(h)) if h else float("nan"))
                for t, h in self._success_by_track.items()}

    @property
    def n_reached(self) -> int:
        return self._n_reached

    def close(self):
        for env in self.envs:
            env._close_heatmap_video()
        self._pool.shutdown(wait=False)