"""
legacy_heatmap.py — Heatmap with the server encoding (commit 50788fee).

The legacy base checkpoints (checkpoints_base/remote_2026100*/, `actor_mu`
network with 7 actions) were trained on the server, whose heatmap normalized
height with the track octomap's GLOBAL extremes (z_min/z_max), piecewise
around robot_z. The current workspace uses a FIXED robot-relative window
(±HEATMAP_Z_RANGE_M), see elevation_map.height_to_grayscale.

Measured on flat (octomap z_min=0.025, z_max=0.075, robot_z ~0.12): the floor
is ~0.87 grey with the legacy encoding and ~0.53 with the current one, so a
legacy base sees an out-of-distribution map under the current encoding.

Only the heatmap is reproduced (same egocentric crop, different
normalization); the state vector (vortex guidance, lookahead, ...) still comes
from the current navigator. Used only when requested explicitly
(heatmap_encoding="legacy" in MissionEnv).
"""
from __future__ import annotations

import numpy as np

from rl_ws.base_training.map_context import MapContext
from rl_ws.elevation_map import get_egocentric_patch


def legacy_height_to_grayscale(height_patch, z_min, z_max, robot_z, eps=1e-6):
    """elevation_map.height_to_grayscale as of 50788fee: global z_min -> white
    (1.0), robot_z -> 0.5, global z_max -> black (0.0), normalized piecewise so
    that robot_z maps exactly to 0.5."""
    h = np.asarray(height_patch, dtype=np.float32)
    norm = np.empty_like(h)
    below = h <= robot_z
    denom_low = max(robot_z - z_min, eps)
    denom_high = max(z_max - robot_z, eps)
    norm[below] = 0.5 * (h[below] - z_min) / denom_low
    norm[~below] = 0.5 + 0.5 * (h[~below] - robot_z) / denom_high
    return (1.0 - np.clip(norm, 0.0, 1.0)).astype(np.float32)


class LegacyHeatmapMapContext(MapContext):
    """MapContext whose get_heatmap() returns the 50788fee encoding. Other
    queries (flipper terrain edges for the reward) are unchanged."""

    def get_heatmap(self, robot_xy, robot_z: float, robot_yaw: float) -> np.ndarray:
        patch = get_egocentric_patch(
            self.elevation_filled, self.origin, self.res, robot_xy, robot_yaw,
            patch_size_m=2.0 * self.radius_m, patch_pixels=self.patch_pixels,
            fill_value=self.z_min)
        if self.shape == "circle":
            raise NotImplementedError("the server trained with shape='square'")
        return legacy_height_to_grayscale(patch, self.z_min, self.z_max, robot_z)

    @classmethod
    def from_map_context(cls, ctx: MapContext) -> "LegacyHeatmapMapContext":
        """Reuse the already loaded grid (avoids re-reading the octree)."""
        new = cls.__new__(cls)
        new.__dict__.update(ctx.__dict__)
        return new
