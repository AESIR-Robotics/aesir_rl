import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_WS_ROOT = os.path.dirname(_THIS_DIR)
if _RL_WS_ROOT not in sys.path:
    sys.path.insert(0, _RL_WS_ROOT)

from ppo_pallets_train import SPAWN_XYZ, GOAL_XY
from global_navigator import plan_route, box_corners_2d

ROBOT_RADIUS = 0.20  
GAP_BRIDGE_DISTANCE = 0.15  

STYLE_OBSTACLE = dict(fc="#2f3542", ec="#000000", alpha=0.90, zorder=2, lw=0.8)
STYLE_STICK = dict(fc="#e17055", ec="#d63031", alpha=0.95, zorder=2, lw=1.2)
STYLE_REAL_PALLET = dict(fc="#ced6e0", ec="#a4b0be", alpha=0.50, zorder=1, lw=0.8)
STYLE_SAFE_PALLET = dict(fc="#00b894", ec="#00cec9", alpha=0.85, zorder=3, lw=0.8)

def main():
    parser = argparse.ArgumentParser(description="Static Navigation Map and Path Visualizer")
    parser.add_argument("--json", default="obstacles.json")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found. Generate the JSON map first.")
        return

    with open(args.json, "r") as f:
        data = json.load(f)

    print("Computing A* path with exact curvature simplification...")
    waypoints = plan_route(args.json, SPAWN_XYZ[:2], GOAL_XY)
    wps_arr = np.array(waypoints)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#1e222b")
    ax.grid(True, linestyle=":", alpha=0.2, color="#ffffff", zorder=0)
    ax.set_xlabel("X [m]", fontsize=11, color="#ffffff")
    ax.set_ylabel("Y [m]", fontsize=11, color="#ffffff")
    ax.set_title("Global Navigation Map - Path Verification", fontsize=13, fontweight="bold", color="#ffffff", pad=14)
    ax.tick_params(colors="#ffffff")

    for o in data["obstacles"]:
        corners = box_corners_2d(np.array(o["center_xy"]), np.array(o["size"]), np.array(o["rot_mat"]))
        ax.add_patch(MPolygon(corners, closed=True, **STYLE_OBSTACLE))

    for s in data["sticks"]:
        corners = box_corners_2d(np.array(s["center_xy"]), np.array(s["size"]), np.array(s["rot_mat"]))
        ax.add_patch(MPolygon(corners, closed=True, **STYLE_STICK))

    pallets_poly = []
    for p in data["pallets"]:
        corners = box_corners_2d(np.array(p["center_xy"]), np.array(p["size"]), np.array(p["rot_mat"]))
        poly = ShapelyPolygon(corners)
        pallets_real_patch = MPolygon(corners, closed=True, **STYLE_REAL_PALLET)
        ax.add_patch(pallets_real_patch)
        pallets_poly.append(poly)
        
        name = p["name"].split()[-1]
        ax.annotate(f"P{name}", xy=p["center_xy"], fontsize=7, ha="center", va="center", color="#ffffff", alpha=0.3, zorder=5)

    merged_pallets = unary_union(pallets_poly)
    dilated = merged_pallets.buffer(GAP_BRIDGE_DISTANCE)
    closed = dilated.buffer(-GAP_BRIDGE_DISTANCE)
    safe_zone_union = closed.buffer(-ROBOT_RADIUS)

    if safe_zone_union.geom_type == 'Polygon':
        geoms_to_plot = [safe_zone_union]
    else:
        geoms_to_plot = list(safe_zone_union.geoms)

    legend_safe = False
    for poly in geoms_to_plot:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        patch = MPolygon(np.column_stack((x, y)), closed=True, **STYLE_SAFE_PALLET)
        if not legend_safe:
            patch.set_label("Safe Drivable Zone (Teal)")
            legend_safe = True
        ax.add_patch(patch)

    # Plot continuous trajectory line beneath markers
    ax.plot(wps_arr[:, 0], wps_arr[:, 1], color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.5, zorder=4)

    # Explicit numerical waypoint markers plotting loop
    legend_wp = False
    for idx, (x, y) in enumerate(waypoints):
        is_start = (idx == 0)
        is_goal = (idx == len(waypoints) - 1)
        
        if is_start:
            color_node = "#e84393"
            label_node = "Spawn Point (WP0)"
        elif is_goal:
            color_node = "#ff4757"
            label_node = f"Goal Node (WP{idx})"
        else:
            color_node = "#ff9f43"
            label_node = "Route Waypoint"

        # Base circle marker representing spatial waypoint location
        if not legend_wp or is_start or is_goal:
            ax.plot(x, y, marker="o", color=color_node, markersize=14, zorder=6, linestyle="None", label=label_node)
            if not is_start and not is_goal:
                legend_wp = True
        else:
            ax.plot(x, y, marker="o", color=color_node, markersize=14, zorder=6, linestyle="None")

        # Overlay exact text index inside the circle marker
        ax.annotate(str(idx), xy=(x, y), fontsize=7, ha="center", va="center", color="#ffffff", fontweight="bold", zorder=7)

    handles, labels = ax.get_legend_handles_labels()
    obstacle_patch = MPolygon([[0,0]], **STYLE_OBSTACLE, label="Rigid Obstacles")
    stick_patch = MPolygon([[0,0]], **STYLE_STICK, label="Collision Sticks")
    real_pallet_patch = MPolygon([[0,0]], **STYLE_REAL_PALLET, label="Eroded Border (Grey)")
    
    handles.extend([obstacle_patch, stick_patch, real_pallet_patch])
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9, facecolor="#1e222b", edgecolor="#a4b0be", labelcolor="#ffffff")

    all_pts = wps_arr.tolist() + [SPAWN_XYZ[:2], GOAL_XY]
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Navigation map figure saved to: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()