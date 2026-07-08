import json
import os
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import unary_union

ROBOT_RADIUS = 0.26         
GAP_BRIDGE_DISTANCE = 0.15  
GRID_RESOLUTION = 0.05 
REACH_DIST = 0.30           
MAX_GUIDE_DIST = 5.0

def box_corners_2d(center_xy: np.ndarray, half_sizes: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    hx, hy = half_sizes[0], half_sizes[1]
    R2 = np.array(rot_mat)[:2, :2]
    ax = R2 @ np.array([hx, 0.0])
    ay = R2 @ np.array([0.0, hy])
    return np.array([
        center_xy + ax + ay,
        center_xy - ax + ay,
        center_xy - ax - ay,
        center_xy + ax - ay,
    ])

class Obstacle2D:
    def __init__(self, name: str, cx: float, cy: float, hx: float, hy: float):
        self.name = name
        self.x = cx
        self.y = cy
        self.hx = hx
        self.hy = hy
        
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x - self.hx, self.y - self.hy, self.x + self.hx, self.y + self.hy)

    @classmethod
    def from_entry(cls, e: dict):
        cx, cy = e["center_xy"]
        s = e["size"]
        g = e["gtype"]
        return cls(e["name"], cx, cy, s[0], s[1] if g == "box" else s[0])

def _aabb_dist(b1: tuple, b2: tuple) -> float:
    dx = max(0., max(b1[0], b2[0]) - min(b1[2], b2[2]))
    dy = max(0., max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return np.sqrt(dx*dx + dy*dy)

def vortex_apf(robot_xy: np.ndarray, target_xy: np.ndarray, obstacles: list[Obstacle2D], 
               safe: float = 0.5, k: float = 0.35, rh: float = 0.28) -> np.ndarray:
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx, dy = tx - rx, ty - ry
    dt = np.hypot(dx, dy)
    
    if dt < 0.15:
        return target_xy.copy()
        
    sc = min(1., dt)
    adx = (dx / dt) * sc
    ady = (dy / dt) * sc
    
    rb = (rx - rh, ry - rh, rx + rh, ry + rh)
    rpx = rpy = 0.
    
    for o in obstacles:
        do = _aabb_dist(rb, o.bounds())
        if 0.01 < do < safe:
            m = k * (1. / do - 1. / safe) / (do**2) * min(1., dt / safe)
            d = np.hypot(rx - o.x, ry - o.y) + 1e-9
            rx_ = (rx - o.x) / d * m
            ry_ = (ry - o.y) / d * m
            rpx += rx_ - ry_
            rpy += ry_ + rx_
            
    vx = adx + rpx
    vy = ady + rpy
    return np.array([rx + vx, ry + vy])

def _plan_segment(json_path: str, start_xy: tuple, goal_xy: tuple) -> list[tuple]:
    with open(json_path, "r") as f:
        data = json.load(f)
        
    pallets_poly = []
    for p in data["pallets"]:
        corners = box_corners_2d(np.array(p["center_xy"]), np.array(p["size"]), np.array(p["rot_mat"]))
        pallets_poly.append(ShapelyPolygon(corners))
        
    merged_pallets = unary_union(pallets_poly)
    dilated = merged_pallets.buffer(GAP_BRIDGE_DISTANCE)
    closed = dilated.buffer(-GAP_BRIDGE_DISTANCE)
    safe_zone_union = closed.buffer(-ROBOT_RADIUS)
    
    all_pts = [np.array(p["center_xy"]) for p in data["pallets"]] + [np.array(start_xy), np.array(goal_xy)]
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    
    margin = 1.5
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    
    width = int(np.ceil((max_x - min_x) / GRID_RESOLUTION))
    height = int(np.ceil((max_y - min_y) / GRID_RESOLUTION))
    
    grid = np.ones((width, height), dtype=bool)
    for i in range(width):
        for j in range(height):
            x = min_x + i * GRID_RESOLUTION
            y = min_y + j * GRID_RESOLUTION
            if safe_zone_union.contains(ShapelyPoint(x, y)):
                grid[i, j] = False
                
    def w_to_g(x, y):
        return int(np.clip(round((x - min_x) / GRID_RESOLUTION), 0, width - 1)), int(np.clip(round((y - min_y) / GRID_RESOLUTION), 0, height - 1))
    def g_to_w(i, j):
        return min_x + i * GRID_RESOLUTION, min_y + j * GRID_RESOLUTION
        
    start_node = w_to_g(*start_xy)
    goal_node = w_to_g(*goal_xy)
    grid[start_node[0], start_node[1]] = False
    grid[goal_node[0], goal_node[1]] = False
    
    moves = [(1,0,1.), (-1,0,1.), (0,1,1.), (0,-1,1.), (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]
    open_set = {start_node: 0.0}
    closed_set = set()
    parent = {}
    g_score = {start_node: 0.0}
    
    while open_set:
        curr = min(open_set, key=lambda o: g_score[o] + np.hypot(o[0]-goal_node[0], o[1]-goal_node[1]))
        if curr == goal_node:
            path = []
            while curr in parent:
                path.append(g_to_w(*curr))
                curr = parent[curr]
            path.append(start_xy)
            path = path[::-1]
            
            # Subsampling reduction optimization
            simplified = [path[0]]
            for i in range(1, len(path) - 1):
                p_prev, p_curr, p_next = np.array(simplified[-1]), np.array(path[i]), np.array(path[i+1])
                v1 = (p_curr - p_prev) / (np.linalg.norm(p_curr - p_prev) + 1e-6)
                v2 = (p_next - p_curr) / (np.linalg.norm(p_next - p_curr) + 1e-6)
                if np.dot(v1, v2) < 0.99: 
                    simplified.append(path[i])
            simplified.append(path[-1])
            return simplified
            
        open_set.pop(curr)
        closed_set.add(curr)
        
        for dx, dy, cost in moves:
            neighbor = (curr[0] + dx, curr[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height) or grid[neighbor[0], neighbor[1]] or neighbor in closed_set:
                continue
            tg = g_score[curr] + cost * GRID_RESOLUTION
            if tg < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tg
                parent[neighbor] = curr
                open_set[neighbor] = tg
                
    return [start_xy, goal_xy]

def plan_route(json_path: str, start_xy: tuple, goal_xy: tuple) -> list[tuple]:
    return _plan_segment(json_path, start_xy, goal_xy)

class GlobalNavigator:
    def __init__(self, json_path: str, waypoints: list):
        with open(json_path, "r") as f:
            data = json.load(f)
        self._wps = [np.array(w) for w in waypoints]
        self._vo = [Obstacle2D.from_entry(o) for o in data["obstacles"] if o["name"] != "col_manija"]
        self._vo += [Obstacle2D.from_entry(s) for s in data["sticks"]]
        self._wi = 0

    def reset(self, robot_xy: np.ndarray):
        self._wi = 0

    def step(self, robot_xy: np.ndarray, robot_yaw: float) -> dict:
        if self._wi < len(self._wps):
            wp = self._wps[self._wi]
            if np.linalg.norm(robot_xy - wp) < REACH_DIST:
                self._wi += 1
                
        target = self._wps[min(self._wi, len(self._wps) - 1)]
        vortex_pt = vortex_apf(robot_xy, target, self._vo)
        
        dx, dy = float(vortex_pt[0] - robot_xy[0]), float(vortex_pt[1] - robot_xy[1])
        d = np.hypot(dx, dy)
        if d < 1e-5:
            obs = np.array([0., 0., 1.], dtype=np.float32)
        else:
            angle = np.arctan2(dy, dx) - robot_yaw
            obs = np.array([min(d / MAX_GUIDE_DIST, 1.), np.sin(angle), np.cos(angle)], dtype=np.float32)
            
        return {"obs": obs, "target": target, "vortex": vortex_pt, "wp": self._wi}

def quat_to_yaw(xquat: np.ndarray) -> float:
    w, x, y, z = xquat
    return np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))