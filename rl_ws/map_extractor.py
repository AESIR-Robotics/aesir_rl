from __future__ import annotations

import os
import json
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_FULL = os.path.join(_HERE, "../models/aesir_complete.xml")
_PALLETS = os.path.join(_HERE, "../models/aesir_pallets.xml")
_ROBOT = os.path.join(_HERE, "../models/aesir_mujoco_robot.xml")
DEFAULT_XML = (_FULL if os.path.exists(_FULL) else
               _PALLETS if os.path.exists(_PALLETS) else _ROBOT)

ROBOT_RESET_XY = np.array([-1.5, 3.5])
DEFAULT_MIN_VIS = 0.06

ROBOT_BODIES = {
    "base_link", "tracked_1", "tracked_2",
    "flipper_1_1", "flipper_2_1", "flipper_3_1", "flipper_4_1",
    "footprint_link", "sensor_lidar", "camera_back_link",
    "wheel_l_1", "wheel_l_2", "wheel_l_3",
    "wheel_r_1", "wheel_r_2", "wheel_r_3",
    "wheel_flip1_back", "wheel_flip1_front",
    "wheel_flip2_back", "wheel_flip2_front",
    "wheel_flip3_back", "wheel_flip3_front",
    "wheel_flip4_back", "wheel_flip4_front",
}

ARM_BODIES = {
    "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
    "logitech_gripper_assembly", "left_finger_link", "right_finger_link",
}

STICK_SUBSTR = "fatal_stick"
MUERTE_SUBSTR = "muerte_"
FATAL_SUBSTR = "fatal_"
FLOOR_KEYWORDS = {"floor", "ground", "plane", "suelo", "piso", "terrain", "table"}

THIN_THRESHOLD = 0.15
MAX_ROBOT_HEIGHT = 0.4

MJGEOM_PLANE = 0
MJGEOM_SPHERE = 2
MJGEOM_CAPSULE = 3
MJGEOM_CYLINDER = 5
MJGEOM_BOX = 6
MJGEOM_MESH = 7

GEOM_TYPE_STR = {
    MJGEOM_PLANE: "plane", MJGEOM_SPHERE: "sphere",
    MJGEOM_CAPSULE: "capsule", MJGEOM_CYLINDER: "cylinder",
    MJGEOM_BOX: "box", MJGEOM_MESH: "mesh",
}

STYLE = {
    "floor": dict(fc="#dfe6e9", ec="#b2bec3", alpha=0.40, zorder=0, lw=0.5, label="Floor"),
    "obstacle": dict(fc="#2d3436", ec="#000000", alpha=0.85, zorder=3, lw=0.8, label="Obstacle"),
    "stick": dict(fc="#e17055", ec="#d63031", alpha=0.95, zorder=5, lw=1.5, label="Collision Stick"),
    "muerte": dict(fc="#d63031", ec="#c0392b", alpha=0.15, zorder=1, lw=0.5, label="Penalty Zone"),
    "fatal": dict(fc="#00b894", ec="#00cec9", alpha=0.80, zorder=4, lw=0.8, label="Pallet Target"),
    "robot": dict(fc="#0984e3", ec="#2980b9", alpha=0.90, zorder=6, lw=0.8, label="Robot Base"),
    "arm": dict(fc="#6c5ce7", ec="#5541d7", alpha=0.70, zorder=6, lw=0.8, label="Robot Arm"),
    "mesh": dict(fc="#fdcb6e", ec="#e17055", alpha=0.50, zorder=3, lw=0.8, label="Mesh BBox"),
    "unknown": dict(fc="#b2bec3", ec="#636e72", alpha=0.30, zorder=1, lw=0.5, label="Unclassified"),
}

def classify_geom(geom_name: str, body_name: str) -> str:
    gn = (geom_name or "").lower()
    bn = (body_name or "").lower()

    if STICK_SUBSTR in gn: return "stick"
    if MUERTE_SUBSTR in gn: return "muerte"
    if FATAL_SUBSTR in gn: return "fatal"
    if any(kw in gn for kw in FLOOR_KEYWORDS) or any(kw in bn for kw in FLOOR_KEYWORDS): 
        return "floor"
    if body_name in ROBOT_BODIES: return "robot"
    if body_name in ARM_BODIES: return "arm"
    if gn == "" and bn == "": return "unknown"
    return "obstacle"

def geom_max_xy(gtype: int, size: np.ndarray, rot_mat: np.ndarray) -> float:
    if gtype == MJGEOM_BOX:
        return max(float(size[0]), float(size[1])) * 2.0
    if gtype in (MJGEOM_CYLINDER, MJGEOM_CAPSULE):
        r = float(size[0])
        h = float(size[1])
        z_axis_world = abs(rot_mat[2, 2])
        if z_axis_world > 0.9:
            return r * 2.0
        return max(r, h) * 2.0
    if gtype == MJGEOM_SPHERE:
        return float(size[0]) * 2.0
    return 1.0

def get_mesh_bbox(model: mujoco.MjModel, gid: int) -> np.ndarray:
    mesh_id = model.geom_dataid[gid]
    if mesh_id < 0:
        return np.array([0.1, 0.1, 0.1])
    
    vertex_adr = model.mesh_vertadr[mesh_id]
    vertex_num = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[vertex_adr : vertex_adr + vertex_num]
    
    if len(vertices) == 0:
        return np.array([0.1, 0.1, 0.1])
        
    scale = model.geom_size[gid]
    scaled_vertices = vertices * scale
    
    min_bounds = np.min(scaled_vertices, axis=0)
    max_bounds = np.max(scaled_vertices, axis=0)
    return (max_bounds - min_bounds) / 2.0

def safe_extract_number(name: str) -> int:
    numbers = re.findall(r'\d+', name)
    return int(numbers[-1]) if numbers else 999

def box_corners_2d(center_xy: np.ndarray,
                   half_sizes: np.ndarray,
                   rot_mat: np.ndarray,
                   min_vis: float = 0.0) -> np.ndarray:
    hx = max(float(half_sizes[0]), min_vis / 2.0)
    hy = max(float(half_sizes[1]), min_vis / 2.0)
    R2 = rot_mat[:2, :2]
    ax = R2 @ np.array([hx, 0.0])
    ay = R2 @ np.array([0.0, hy])
    return np.array([
        center_xy + ax + ay,
        center_xy - ax + ay,
        center_xy - ax - ay,
        center_xy + ax - ay,
    ])

def extract_geoms(model: mujoco.MjModel,
                  data: mujoco.MjData,
                  only_collision: bool = False) -> list[dict]:
    result = []
    for gid in range(model.ngeom):
        g_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        body_id = int(model.geom_bodyid[gid])
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

        is_collision = int(model.geom_contype[gid]) != 0
        if only_collision and not is_collision:
            continue

        gtype = int(model.geom_type[gid])
        gtype_str = GEOM_TYPE_STR.get(gtype, f"type{gtype}")
        center = data.geom_xpos[gid].copy()
        rot_mat = data.geom_xmat[gid].reshape(3, 3)
        
        if gtype == MJGEOM_MESH:
            size = get_mesh_bbox(model, gid)
        else:
            size = model.geom_size[gid].copy()
            
        category = classify_geom(g_name, b_name)
        
        # Filter obstacles positioned above the robot's driving clearance height
        if category in ("obstacle", "stick") and is_collision:
            if gtype == MJGEOM_BOX and (center[2] - size[2]) > MAX_ROBOT_HEIGHT:
                continue
            elif gtype in (MJGEOM_CYLINDER, MJGEOM_CAPSULE) and abs(rot_mat[2,2]) > 0.9 and (center[2] - size[1]) > MAX_ROBOT_HEIGHT:
                continue
            elif center[2] > MAX_ROBOT_HEIGHT and gtype not in (MJGEOM_BOX, MJGEOM_CYLINDER, MJGEOM_CAPSULE):
                continue

        is_thin = (category in ("obstacle", "stick") and
                   geom_max_xy(gtype, size, rot_mat) < THIN_THRESHOLD)

        entry = dict(
            geom_id=gid, name=g_name, body_name=b_name,
            category=category, gtype=gtype, gtype_str=gtype_str,
            center_xy=center[:2].copy(), center_z=float(center[2]),
            size=size, rot_mat=rot_mat,
            is_collision=is_collision, is_thin=is_thin,
        )
        result.append(entry)
    return result

def _make_patch(g: dict, style: dict, min_vis: float):
    gtype = g["gtype"]
    c_xy = g["center_xy"]
    size = g["size"]
    rot_mat = g["rot_mat"]
    kw = dict(fc=style["fc"], ec=style["ec"], alpha=style["alpha"], zorder=style["zorder"], lw=style["lw"])

    if gtype == MJGEOM_PLANE:
        hx = float(size[0]) if float(size[0]) > 0 else 60.0
        hy = float(size[1]) if float(size[1]) > 0 else 60.0
        return Polygon(box_corners_2d(c_xy, np.array([hx, hy, 0.0]), rot_mat), closed=True, **kw)
    if gtype in (MJGEOM_BOX, MJGEOM_MESH):
        return Polygon(box_corners_2d(c_xy, size, rot_mat, min_vis), closed=True, **kw)
    if gtype in (MJGEOM_CYLINDER, MJGEOM_CAPSULE):
        if abs(rot_mat[2, 2]) > 0.9:
            return mpatches.Circle(c_xy, max(float(size[0]), min_vis / 2.0), **kw)
        return Polygon(box_corners_2d(c_xy, np.array([size[1], size[0], 0.0]), rot_mat, min_vis), closed=True, **kw)
    if gtype == MJGEOM_SPHERE:
        return mpatches.Circle(c_xy, max(float(size[0]), min_vis / 2.0), **kw)
    return None

def build_map_figure(geoms: list[dict], title: str = "Bird's Eye View - Aesir Track", min_vis: float = DEFAULT_MIN_VIS) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f4f8")
    ax.grid(True, linestyle="--", alpha=0.35, color="#a0aec0", zorder=0)
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    legend_done = set()
    for g in geoms:
        cat = g["category"]
        style = STYLE.get(cat, STYLE["unknown"])
        patch = _make_patch(g, style, min_vis)
        if patch is not None:
            if cat not in legend_done:
                patch.set_label(style["label"])
                legend_done.add(cat)
            ax.add_patch(patch)

        if cat in ("fatal", "stick", "muerte"):
            txt = g["name"].replace("fatal_pallet ", "P").replace("fatal_stick ", "S").replace("fatal_", "").replace("muerte_", "!")
            ax.annotate(txt, xy=g["center_xy"], fontsize=6.5, ha="center", va="center", fontweight="bold",
                        color="white" if cat in ("fatal", "stick") else "#2d3436", zorder=style["zorder"] + 2)

    ax.plot(*ROBOT_RESET_XY, marker="*", color="#e84393", markersize=16, zorder=10, linestyle="None", label="Agent Reset")
    
    xs = [g["center_xy"][0] for g in geoms if g["category"] not in {"floor", "muerte", "robot", "arm"}]
    ys = [g["center_xy"][1] for g in geoms if g["category"] not in {"floor", "muerte", "robot", "arm"}]
    if xs:
        ax.set_xlim(min(xs) - 2.0, max(xs) + 2.0)
        ax.set_ylim(min(ys) - 2.0, max(ys) + 2.0)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#a0aec0")
    fig.tight_layout()
    return fig

def export_json(geoms: list[dict], path: str):
    obstacles, sticks, pallets = [], [], []
    for g in geoms:
        if not g["is_collision"]:
            continue
        entry = {
            "name": g["name"],
            "gtype": g["gtype_str"],
            "center_xy": g["center_xy"].tolist(),
            "center_z": g["center_z"],
            "size": g["size"].tolist(),
            "rot_mat": g["rot_mat"].tolist(),
        }
        if g["category"] == "obstacle": obstacles.append(entry)
        elif g["category"] == "stick": sticks.append(entry)
        elif g["category"] == "fatal" and "pallet" in g["name"].lower(): pallets.append(entry)

    pallets.sort(key=lambda p: safe_extract_number(p["name"]))
    output = {
        "obstacles": obstacles,
        "sticks": sticks,
        "pallets": pallets,
        "waypoints": [p["center_xy"] for p in pallets],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="ENV MAP")
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--save", default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--only-collision", action="store_true")
    parser.add_argument("--min-vis", type=float, default=DEFAULT_MIN_VIS)
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        print(f"Error: XML path not found: {args.xml}")
        return

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    geoms = extract_geoms(model, data, only_collision=args.only_collision)

    if args.export_json:
        export_json(geoms, args.export_json)

    fig = build_map_figure(geoms, min_vis=args.min_vis)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()