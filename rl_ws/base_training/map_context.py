"""
map_context.py — Contexto de mapa ESTATICO, compartido entre todos los envs
(mismo patron que `shared_model` en mujoco_sim_base.py: se carga UNA vez y
se pasa de solo-lectura a cada BaseMujocoEnv).

Envuelve elevation_map.py para exponer una sola llamada barata que hay que
correr EN CADA STEP: get_heatmap(robot_xy, robot_z, robot_yaw) -> (H,W)
numpy array listo para meter al CNN. La parte pesada (leer el .bt,
extractPointCloud, colapsar a grilla de elevacion) se hace una unica vez
en __init__.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Permitir importar elevation_map.py cuando el script se ejecuta desde la raíz del proyecto
# o desde rl_ws/base_training.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RL_WS = os.path.abspath(os.path.join(_HERE, ".."))
if _RL_WS not in sys.path:
    sys.path.insert(0, _RL_WS)

try:
    from rl_ws.elevation_map import (
        octree_to_global_elevation, get_circular_height_heatmap, flipper_climb_edge,
    )
except ModuleNotFoundError:
    from elevation_map import (
        octree_to_global_elevation, get_circular_height_heatmap, flipper_climb_edge,
    )


class MapContext:
    """
    Uso:
        map_ctx = MapContext(
            bt_path=C.OCTOMAP_BT_PATH,
            resolution=C.OCTOMAP_RESOLUTION,
            radius_m=C.HEATMAP_RADIUS_M,
            patch_pixels=C.HEATMAP_PIXELS,
        )
        # ... se pasa el MISMO map_ctx a todos los BaseMujocoEnv (solo lectura) ...

        heatmap = map_ctx.get_heatmap(robot_xy=fb["xy"], robot_z=fb["z"], robot_yaw=fb["yaw"])
    """

    def __init__(self, bt_path: str, resolution: float,
                 radius_m: float = 1.0, patch_pixels: int = 64,
                 shape: str = "square", margin: float = 0.5):
        self.resolution = resolution
        self.radius_m = radius_m
        self.patch_pixels = patch_pixels
        self.shape = shape

        # Carga pesada -- UNA sola vez para todos los envs.
        (self.elevation, self.origin, self.res,
         self.z_min, self.z_max) = octree_to_global_elevation(
            bt_path, resolution=resolution, margin=margin
        )

        print(f"[map_context] Cargado {bt_path}: grilla {self.elevation.shape}, "
              f"z_min={self.z_min:.3f}  z_max={self.z_max:.3f}")

    def get_heatmap(self, robot_xy, robot_z: float, robot_yaw: float) -> np.ndarray:
        """
        Barato -- solo numpy (recorte + rotacion), no vuelve a tocar el
        octree. Esto es lo que se llama EN CADA STEP del env.
        """
        return get_circular_height_heatmap(
            self.elevation, self.origin, self.res,
            robot_xy=robot_xy, robot_z=robot_z, robot_yaw=robot_yaw,
            z_min=self.z_min, z_max=self.z_max,
            radius_m=self.radius_m, patch_pixels=self.patch_pixels,
            shape=self.shape,
        )

    def get_flipper_climb_edges(self, robot_xy, robot_yaw: float,
                                mounts_xy, axis_sign, look_ahead_m: float,
                                min_step_m: float, max_climb_m: float,
                                n_samples: int = 8):
        """Busca el borde real (escalon) en la direccion de extension de cada
        flipper -- ver elevation_map.flipper_climb_edge. Solo geometria pura
        (usa self.elevation cruda, NO el heatmap normalizado); el criterio de
        "la punta lo libra" vive en base_env.flipper_terrain_bonus.
        Devuelve (found, d_edge, h_edge, climbable), cada uno (4,)."""
        return flipper_climb_edge(
            self.elevation, self.origin, self.res, self.z_min,
            robot_xy=robot_xy, robot_yaw=robot_yaw,
            mounts_xy=mounts_xy, axis_sign=axis_sign, look_ahead_m=look_ahead_m,
            min_step_m=min_step_m, max_climb_m=max_climb_m, n_samples=n_samples,
        )