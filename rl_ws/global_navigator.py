"""
global_navigator.py — Navegacion global compartida: planeacion A* sobre la zona
segura de pallets + seguimiento con vortex APF (atraccion al waypoint, repulsion
de obstaculos/bordes con componente tangencial anti-minimo-local) + lookahead.

Lo usan los DOS backends (mujoco_sim_base.py directo y base_ros_env.py sobre el
bridge ROS2) y los scripts de visualizacion/tests. Todos los parametros
ajustables viven en base_training/config.py.
"""
import json
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import unary_union, nearest_points

# Parametros desde config.py (import dual: este modulo se usa tanto como
# `rl_ws.global_navigator` — sys.path en aesir_rl/ — como `global_navigator`
# a secas — sys.path en rl_ws/, p.ej. los scripts de tests/).
try:
    from rl_ws.base_training.config import (
        ROBOT_RADIUS, GAP_BRIDGE_DISTANCE, GRID_RESOLUTION,
        CORNER_DOT_THRESHOLD, MAX_WAYPOINT_DIST,
        REACH_DIST, MAX_GUIDE_DIST, N_LOOKAHEAD, LOOKAHEAD_STEP,
        ATT_GAIN, ATT_RANGE, REP_GAIN, REP_RANGE, ROBOT_HALF, SWIRL, MIN_PROGRESS,
        PLATFORM_HALF_EXTENT, VIRTUAL_OBSTACLE_HALF_SIZE,
        VIRTUAL_OBSTACLE_MIN_HALF_SIZE, VIRTUAL_OBSTACLE_OFFSET_FRAC,
    )
except ImportError:
    from base_training.config import (
        ROBOT_RADIUS, GAP_BRIDGE_DISTANCE, GRID_RESOLUTION,
        CORNER_DOT_THRESHOLD, MAX_WAYPOINT_DIST,
        REACH_DIST, MAX_GUIDE_DIST, N_LOOKAHEAD, LOOKAHEAD_STEP,
        ATT_GAIN, ATT_RANGE, REP_GAIN, REP_RANGE, ROBOT_HALF, SWIRL, MIN_PROGRESS,
        PLATFORM_HALF_EXTENT, VIRTUAL_OBSTACLE_HALF_SIZE,
        VIRTUAL_OBSTACLE_MIN_HALF_SIZE, VIRTUAL_OBSTACLE_OFFSET_FRAC,
    )

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

def _nearest_point_on_box(p: np.ndarray, bounds: tuple) -> np.ndarray:
    """Punto mas cercano del AABB `bounds` a `p` (clamp por eje). Si p esta
    AFUERA es el punto real de la superficie -- funciona bien tambien en las
    esquinas, a diferencia de usar el centro de la caja como origen de la
    repulsion (ver _repulsion_vec). Si p esta ADENTRO coincide con p mismo
    (caso degenerado, lo resuelve el fallback en vortex_apf)."""
    x0, y0, x1, y1 = bounds
    return np.array([min(max(p[0], x0), x1), min(max(p[1], y0), y1)])

def _repulsion_vec(robot: np.ndarray, source_xy: np.ndarray, gap: float,
                   rep_gain: float, rep_range: float, swirl: float) -> np.ndarray:
    """Repulsion de UNA fuente: caida LINEAL acotada (sin singularidad 1/do^2,
    sin piso ni tope: la magnitud vive en [0, rep_gain] por construccion) mas
    la componente tangencial 'vortex' (rota la radial 90deg) que evita minimos
    locales. gap = hueco cuerpo-fuente; direccion = alejarse de source_xy."""
    if gap >= rep_range:
        return np.zeros(2)
    strength = rep_gain * (1.0 - max(gap, 0.0) / rep_range)   # [0, rep_gain]
    away = robot - source_xy
    n = np.hypot(*away) + 1e-9
    radial = away / n * strength
    tangential = np.array([-radial[1], radial[0]]) * swirl
    return radial + tangential


def _edge_repulsion(robot: np.ndarray, zone, rep_gain: float, rep_range: float,
                    rh: float, swirl: float) -> np.ndarray:
    """Repulsion de los BORDES del contorno transitable de pallets ('zone',
    shapely Polygon/MultiPolygon): empuja al robot HACIA ADENTRO para que no se
    salga por las orillas. gap = distancia cuerpo-borde (dist al borde - rh),
    misma caida lineal acotada que un obstaculo."""
    pt = ShapelyPoint(robot[0], robot[1])
    boundary = zone.boundary
    dist = pt.distance(boundary)
    gap = dist - rh
    if gap >= rep_range:
        return np.zeros(2)
    near = nearest_points(boundary, pt)[0]
    near = np.array([near.x, near.y])
    # Direccion para MANTENER DENTRO: si el robot esta dentro, alejarse del
    # borde (hacia el interior); si ya salio, volver hacia el borde (adentro).
    toward_in = (robot - near) if zone.contains(pt) else (near - robot)
    n = np.hypot(*toward_in) + 1e-9
    strength = rep_gain * (1.0 - max(gap, 0.0) / rep_range)
    radial = toward_in / n * strength
    tangential = np.array([-radial[1], radial[0]]) * swirl
    return radial + tangential


def vortex_apf(robot_xy: np.ndarray, target_xy: np.ndarray, obstacles: list[Obstacle2D],
               edges=None,
               att_gain: float = ATT_GAIN, att_range: float = ATT_RANGE,
               rep_gain: float = REP_GAIN, rep_range: float = REP_RANGE,
               rh: float = ROBOT_HALF, swirl: float = SWIRL,
               min_progress: float = MIN_PROGRESS) -> np.ndarray:
    """Campo vortex: ATRACCION al waypoint + REPULSION (radial + swirl) de los
    obstaculos-caja (sticks, puerta) y de los BORDES de los pallets ('edges').

    Parametros (todos con default arriba, controlables por el que llame):
      att_gain / att_range  grado y distancia de la atraccion.
      rep_gain / rep_range  grado y distancia de la repulsion (compartidos por
                            obstaculos y bordes de pallets).
      rh                    media anchura del robot (su cuerpo, no un punto).
      swirl                 peso de la componente tangencial (remolino vortex).
      min_progress          fraccion de la atraccion que SIEMPRE sobrevive a la
                            repulsion (garantia anti-minimo-local, ver abajo).
      edges                 contorno transitable de pallets (shapely); si se
                            pasa, sus bordes repelen hacia adentro.

    Devuelve el punto-guia (robot + atraccion + repulsion). La repulsion es
    acotada por construccion (caida lineal), asi que no hace falta ningun tope
    ni piso ad-hoc."""
    robot  = np.array([float(robot_xy[0]), float(robot_xy[1])])
    target = np.array([float(target_xy[0]), float(target_xy[1])])
    rx, ry = robot

    # ── Atraccion al waypoint: unidad hacia el target, magnitud que crece de 0
    #    a att_gain sobre att_range (pull suave al llegar).
    to_t = target - robot
    dt = float(np.hypot(*to_t))
    att = (to_t / dt) * att_gain * min(dt / att_range, 1.0) if dt > 1e-9 else np.zeros(2)

    # ── Repulsion de obstaculos-caja (sticks + puerta): gap = hueco entre la
    #    caja del robot (media anchura rh) y la caja del obstaculo. La
    #    DIRECCION se toma desde el punto mas cercano del BORDE del obstaculo
    #    (no desde su centro): usar el centro sesga la direccion hacia el
    #    centro de la caja cuando el robot se acerca a un lado plano lejos de
    #    una esquina (empuja tambien "hacia el costado equivocado" ademas de
    #    alejar), lo que puede desviarlo justo hacia una esquina en vez de
    #    alejarlo limpiamente del lado que esta cruzando.
    rep = np.zeros(2)
    rb = (rx - rh, ry - rh, rx + rh, ry + rh)
    for o in obstacles:
        bounds = o.bounds()
        gap = _aabb_dist(rb, bounds)
        nearest = _nearest_point_on_box(robot, bounds)
        # robot con el centro DENTRO de la caja (nearest == robot, direccion
        # degenerada) -- fallback al centro de la caja para tener con que salir.
        source = nearest if np.hypot(*(robot - nearest)) > 1e-9 else np.array([o.x, o.y])
        rep += _repulsion_vec(robot, source, gap, rep_gain, rep_range, swirl)

    # ── Repulsion de los bordes de los pallets (mantener dentro).
    if edges is not None:
        rep += _edge_repulsion(robot, edges, rep_gain, rep_range, rh, swirl)

    attr_mag = float(np.hypot(*att))
    if attr_mag > 1e-9:
        u = att / attr_mag
        along = float(np.dot(rep, u))
        floor = -(1.0 - min_progress) * attr_mag
        if along < floor:
            rep = rep + (floor - along) * u

    return robot + att + rep

def _rel_obs(rel_xy: np.ndarray, robot_yaw: float) -> np.ndarray:
    """Codifica un punto RELATIVO al robot (vector robot->punto en frame mapa)
    como (dist_norm, sin(ang), cos(ang)) en el frame del robot — misma
    representacion que usa la guia inmediata."""
    d = float(np.hypot(rel_xy[0], rel_xy[1]))
    if d < 1e-5:
        return np.array([0., 0., 1.], dtype=np.float32)
    ang = np.arctan2(rel_xy[1], rel_xy[0]) - robot_yaw
    return np.array([min(d / MAX_GUIDE_DIST, 1.), np.sin(ang), np.cos(ang)], dtype=np.float32)


def build_platform_zone(half_extent: float = PLATFORM_HALF_EXTENT,
                        robot_radius: float = ROBOT_RADIUS) -> ShapelyPolygon:
    he = float(half_extent)
    square = ShapelyPolygon([(-he, -he), (he, -he), (he, he), (-he, he)])
    return square.buffer(-robot_radius)


def plan_platform_route(start_xy: tuple, goal_xy: tuple,
                        max_wp_dist: float = MAX_WAYPOINT_DIST) -> list[tuple]:
    p_start = np.array(start_xy, dtype=float)
    p_end = np.array(goal_xy, dtype=float)
    segment_vector = p_end - p_start
    segment_dist = float(np.linalg.norm(segment_vector))

    waypoints = [tuple(p_start)]
    if segment_dist > max_wp_dist:
        num_segments = int(np.ceil(segment_dist / max_wp_dist))
        for j in range(1, num_segments):
            alpha = j / num_segments
            waypoints.append(tuple(p_start + alpha * segment_vector))
    waypoints.append(tuple(p_end))
    return waypoints


def _seg_hits_box(a, b, lo, hi) -> bool:
    """True si el segmento a->b cruza la caja AABB [lo,hi] (Liang-Barsky)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(2):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
        else:
            inv = 1.0 / d[k]
            ta, tb = (lo[k] - a[k]) * inv, (hi[k] - a[k]) * inv
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


def _resample_polyline(points, max_wp_dist: float) -> list[tuple]:
    """Densifica una polilinea insertando puntos para que ningun tramo supere
    max_wp_dist (conserva los vertices originales)."""
    pts = [np.asarray(p, float) for p in points]
    out = [tuple(pts[0])]
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        d = float(np.hypot(*seg))
        if d > max_wp_dist:
            nseg = int(np.ceil(d / max_wp_dist))
            for j in range(1, nseg):
                out.append(tuple(pts[i] + seg * (j / nseg)))
        out.append(tuple(pts[i + 1]))
    return out


def plan_platform_route_with_obstacle(
        start_xy: tuple, goal_xy: tuple,
        max_wp_dist: float = MAX_WAYPOINT_DIST,
        max_half_size: float = VIRTUAL_OBSTACLE_HALF_SIZE,
        min_half_size: float = VIRTUAL_OBSTACLE_MIN_HALF_SIZE,
        offset_frac_range: tuple = VIRTUAL_OBSTACLE_OFFSET_FRAC):
    """Coloca PRIMERO el obstaculo (random, sobre el paso directo) y luego PLANEA
    la ruta TENIENDO EN CUENTA la caja: si el paso directo la cruza, la rodea por
    las esquinas (infladas por el cuerpo del robot) del lado mas corto y dentro
    de la plataforma. Es un shortest-path exacto de un solo obstaculo via grafo
    de visibilidad de pocos nodos (spawn, meta, 4 esquinas). El vortex hace la
    evasion FINA; estos waypoints ya doblan alrededor, asi el lookahead que ve la
    politica es coherente con el rodeo (no recto por encima de la caja).

    SOLO para la plataforma plana. La pista de pallets usa plan_route/_plan_segment
    (A* sobre el JSON), que esta funcion NO toca.

    Devuelve (waypoints, obstacle_or_None) -- None si la ruta es muy corta para
    meter un obstaculo con margen."""
    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    delta = goal - start
    total = float(np.hypot(*delta))
    if total < 4.0 * max_half_size + 2.0 * ROBOT_RADIUS:
        return _resample_polyline([start, goal], max_wp_dist), None
    direction = delta / total
    perp = np.array([-direction[1], direction[0]])

    # 1) Obstaculo PRIMERO: proyeccion random sobre la linea + offset lateral
    #    (< half) para que la caja inflada intersecte el paso directo -> rodeo.
    half = float(np.random.uniform(min_half_size, max_half_size))
    u = float(np.random.uniform(0.35, 0.65))
    base = start + direction * (u * total)
    side = 1.0 if np.random.rand() < 0.5 else -1.0
    lateral = side * float(np.random.uniform(*offset_frac_range)) * half
    center = base + perp * lateral
    obstacle = Obstacle2D("virtual_obstacle", float(center[0]), float(center[1]), half, half)

    # 2) Planear CON el obstaculo. Caja inflada por el cuerpo del robot + holgura.
    margin = ROBOT_RADIUS + 0.10
    hi_half = half + margin
    box_lo, box_hi = center - hi_half, center + hi_half
    if not _seg_hits_box(start, goal, box_lo, box_hi):
        return _resample_polyline([start, goal], max_wp_dist), obstacle   # no estorba

    # Nodos de visibilidad: esquinas empujadas un pelin afuera (para que los
    # segmentos hacia ellas no rocen el interior); descarta las que caen fuera
    # de la plataforma transitable.
    eps = 0.05
    plat = PLATFORM_HALF_EXTENT - ROBOT_RADIUS
    corners = [center + np.array([sx, sy]) * (hi_half + eps)
               for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    corners = [c for c in corners if abs(c[0]) <= plat and abs(c[1]) <= plat]
    nodes = [start, goal] + corners
    blo, bhi = center - (hi_half - eps), center + (hi_half - eps)   # caja de bloqueo

    # Dijkstra sobre el grafo (aristas cuyo segmento no cruza la caja de bloqueo).
    n = len(nodes)
    INF = float("inf")
    dist = [INF] * n; dist[0] = 0.0
    prev = [-1] * n; done = [False] * n
    for _ in range(n):
        u2, best = -1, INF
        for i in range(n):
            if not done[i] and dist[i] < best:
                best, u2 = dist[i], i
        if u2 == -1:
            break
        done[u2] = True
        for v in range(n):
            if done[v] or _seg_hits_box(nodes[u2], nodes[v], blo, bhi):
                continue
            w = dist[u2] + float(np.hypot(*(nodes[v] - nodes[u2])))
            if w < dist[v]:
                dist[v], prev[v] = w, u2

    if dist[1] == INF:                        # sin rodeo valido (raro) -> el vortex esquiva
        return _resample_polyline([start, goal], max_wp_dist), obstacle
    path, k = [], 1
    while k != -1:
        path.append(nodes[k]); k = prev[k]
    path.reverse()
    return _resample_polyline(path, max_wp_dist), obstacle


def build_safe_zone(json_path: str):
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

    blocker_polys = []
    for s in data.get("sticks", []):
        corners = box_corners_2d(np.array(s["center_xy"]), np.array(s["size"]), np.array(s["rot_mat"]))
        blocker_polys.append(ShapelyPolygon(corners))
    for o in data.get("obstacles", []):
        if o["name"] == "col_manija":
            continue
        corners = box_corners_2d(np.array(o["center_xy"]), np.array(o["size"]), np.array(o["rot_mat"]))
        blocker_polys.append(ShapelyPolygon(corners))
    if blocker_polys:
        blockers_inflated = unary_union(blocker_polys).buffer(ROBOT_RADIUS)
        safe_zone_union = safe_zone_union.difference(blockers_inflated)

    return safe_zone_union, data

def _astar_route(safe_zone_union, data: dict, start_xy: tuple, goal_xy: tuple) -> list[tuple]:
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
            
            corners = [path[0]]
            for i in range(1, len(path) - 1):
                p_prev, p_curr, p_next = np.array(corners[-1]), np.array(path[i]), np.array(path[i+1])
                v1 = (p_curr - p_prev) / (np.linalg.norm(p_curr - p_prev) + 1e-6)
                v2 = (p_next - p_curr) / (np.linalg.norm(p_next - p_curr) + 1e-6)
                if np.dot(v1, v2) < CORNER_DOT_THRESHOLD:
                    corners.append(path[i])
            corners.append(path[-1])

            final_waypoints = [corners[0]]
            for i in range(len(corners) - 1):
                p_start = np.array(corners[i])
                p_end = np.array(corners[i + 1])
                segment_vector = p_end - p_start
                segment_dist = np.linalg.norm(segment_vector)

                if segment_dist > MAX_WAYPOINT_DIST:
                    num_segments = int(np.ceil(segment_dist / MAX_WAYPOINT_DIST))
                    for j in range(1, num_segments):
                        alpha = j / num_segments
                        p_interp = p_start + alpha * segment_vector
                        final_waypoints.append(tuple(p_interp))

                final_waypoints.append(tuple(p_end))

            return final_waypoints
            
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

def plan_route(json_path: str, start_xy: tuple, goal_xy: tuple, safe_zone=None) -> list[tuple]:
    if safe_zone is None:
        safe_zone = build_safe_zone(json_path)
    safe_zone_union, data = safe_zone
    return _astar_route(safe_zone_union, data, start_xy, goal_xy)

class GlobalNavigator:
    def __init__(self, json_path, waypoints: list,
                 att_gain: float = ATT_GAIN, att_range: float = ATT_RANGE,
                 rep_gain: float = REP_GAIN, rep_range: float = REP_RANGE,
                 rh: float = ROBOT_HALF, swirl: float = SWIRL,
                 min_progress: float = MIN_PROGRESS,
                 pallet_edges: bool = True,
                 n_lookahead: int = N_LOOKAHEAD,
                 lookahead_step: float = LOOKAHEAD_STEP,
                 obstacles: list = None,
                 edges_zone=None):
        """json_path=None -> modo plataforma plana (sin pallets/sticks fisicos)"""
        self._wps = [np.array(w) for w in waypoints]

        if json_path is None:
            self._vo = list(obstacles) if obstacles is not None else []
            self._edges = edges_zone
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
            self._vo = [Obstacle2D.from_entry(o) for o in data["obstacles"] if o["name"] != "col_manija"]
            self._vo += [Obstacle2D.from_entry(s) for s in data["sticks"]]
            if obstacles is not None:
                self._vo += list(obstacles)

            self._edges = edges_zone
            if pallet_edges and self._edges is None:
                polys = [ShapelyPolygon(box_corners_2d(np.array(p["center_xy"]),
                                                       np.array(p["size"]),
                                                       np.array(p["rot_mat"])))
                         for p in data["pallets"]]
                merged = unary_union(polys)
                self._edges = merged.buffer(GAP_BRIDGE_DISTANCE).buffer(-GAP_BRIDGE_DISTANCE)

        self._att_gain, self._att_range = att_gain, att_range
        self._rep_gain, self._rep_range = rep_gain, rep_range
        self._rh, self._swirl = rh, swirl
        self._min_progress = min_progress
        self._n_look = int(n_lookahead)
        self._look_step = float(lookahead_step)
        self._wi = 0

    def reset(self, robot_xy: np.ndarray):
        self._wi = 0

    def replan(self, waypoints: list, obstacles: list = None):
        self._wps = [np.array(w) for w in waypoints]
        self._wi = 0
        if obstacles is not None:
            self._vo = list(obstacles)

    def _vortex_at(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        return vortex_apf(pos, target, self._vo, edges=self._edges,
                          att_gain=self._att_gain, att_range=self._att_range,
                          rep_gain=self._rep_gain, rep_range=self._rep_range,
                          rh=self._rh, swirl=self._swirl,
                          min_progress=self._min_progress)

    def _lookahead(self, robot_xy: np.ndarray, robot_yaw: float) -> np.ndarray:
        """Previsualiza la ruta que viene: n_look puntos muestreados a arco fijo
        (lookahead_step). Cada punto = 3 numeros en el FRAME DEL CUERPO:
            [dx, dy, theta_tan]
          - (dx, dy)   : posicion del punto relativa al robot (cartesiana, rotada
                         por -yaw). Cartesiana en vez de polar -> captura posicion
                         + curvatura directo, sin el canal de distancia casi
                         constante de antes.
          - theta_tan  : angulo (rad) de la direccion HACIA DONDE VA la ruta en ese
                         punto, relativo al heading del robot -> la politica
                         anticipa curvas, no solo posiciones. Un solo escalar (sin
                         ambiguedad); va ~0 en ruta recta de frente.
        Devuelve (feats 3*n_look, samples Nx2 en mundo para dibujar)."""
        if self._n_look <= 0:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 2), dtype=float)
        c, s_yaw = np.cos(robot_yaw), np.sin(robot_yaw)

        def to_body(vec):   # R(-yaw) @ vec_world
            x, y = float(vec[0]), float(vec[1])
            return c * x + s_yaw * y, -s_yaw * x + c * y

        origin = np.array([float(robot_xy[0]), float(robot_xy[1])])
        pts = [origin] + [self._wps[i] for i in range(self._wi, len(self._wps))]
        cum = np.concatenate([[0.0], np.cumsum(
            [float(np.hypot(*(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)])])
        total = float(cum[-1])
        feats, samples = [], []
        for k in range(1, self._n_look + 1):
            s = min(k * self._look_step, total)
            j = int(min(np.searchsorted(cum, s, side="right") - 1, len(pts) - 2))
            j = max(j, 0)
            if len(pts) < 2:
                sample = pts[0]
                seg = np.zeros(2)
            else:
                seglen = float(cum[j + 1] - cum[j])
                t = 0.0 if seglen < 1e-9 else (s - float(cum[j])) / seglen
                sample = pts[j] + t * (pts[j + 1] - pts[j])
                seg = pts[j + 1] - pts[j]
            seg_n = float(np.hypot(*seg))
            if seg_n > 1e-9:
                dir_world = seg / seg_n
            else:                                    # segmento degenerado: fallbacks
                to = sample - origin
                to_n = float(np.hypot(*to))
                dir_world = to / to_n if to_n > 1e-9 else np.array([c, s_yaw])
            dx, dy = to_body(sample - origin)        # posicion relativa (cartesiana)
            tx, ty = to_body(dir_world)              # tangente en frame del cuerpo
            theta_tan = float(np.arctan2(ty, tx))    # heading de la ruta, relativo
            feats.append(np.array([dx, dy, theta_tan], dtype=np.float32))
            samples.append(sample)
        # feats: lo que ve la politica (relativo); samples: mundo (para dibujar)
        return np.concatenate(feats).astype(np.float32), np.asarray(samples, dtype=float)

    def step(self, robot_xy: np.ndarray, robot_yaw: float) -> dict:
        rxy = np.asarray(robot_xy, dtype=float)
        d = np.linalg.norm(np.asarray(self._wps, dtype=float) - rxy, axis=1)
        nearest = int(np.argmin(d))
        self._wi = max(self._wi, min(nearest + 1, len(self._wps) - 1))
        if float(d[self._wi]) < REACH_DIST:          # llegada fina al target actual
            self._wi = min(self._wi + 1, len(self._wps) - 1)

        target = self._wps[min(self._wi, len(self._wps) - 1)]
        vortex_pt = self._vortex_at(robot_xy, target)
        obs = _rel_obs(vortex_pt - np.asarray(robot_xy, dtype=float), robot_yaw)
        # Waypoint-objetivo CRUDO relativo al robot (el subobjetivo "limpio",
        # aparte del punto del vortex que ya mezcla atraccion+repulsion).
        target_obs = _rel_obs(np.asarray(target, dtype=float) - np.asarray(robot_xy, dtype=float),
                              robot_yaw)
        lookahead, lookahead_xy = self._lookahead(robot_xy, robot_yaw)

        return {"obs": obs, "target": target, "target_obs": target_obs,
                "vortex": vortex_pt, "wp": self._wi,
                "lookahead": lookahead, "lookahead_xy": lookahead_xy,
                "goal": self._wps[-1]}     # meta final (ultimo waypoint), para modular velocidad

def quat_to_yaw(xquat: np.ndarray) -> float:
    w, x, y, z = xquat
    return np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))

def quat_upright(xquat) -> float:
    """Componente z del eje z del chasis (R[2,2]): 1 = vertical, 0 = tumbado."""
    w, x, y, z = xquat
    return 1.0 - 2.0 * (x * x + y * y)

def quat_to_grav_body(xquat) -> np.ndarray:
    """Vector de GRAVEDAD (unitario, apunta hacia abajo) expresado en el frame
    del cuerpo: g_body = R^T · [0,0,-1]. En plano da [0,0,-1]; al subir/bajar una
    pendiente gx captura el PITCH y gy el ROLL, asi la politica distingue
    'inclinado hacia adelante subiendo' de 'ladeado a punto de volcar' — cosa que
    el escalar upright (solo magnitud) no puede. gz == -upright."""
    w, x, y, z = [float(v) for v in xquat]
    return np.array([
        2.0 * (w * y - x * z),          # gx  (pitch)
        -2.0 * (y * z + w * x),         # gy  (roll)
        2.0 * (x * x + y * y) - 1.0,    # gz  (= -upright)
    ], dtype=np.float32)