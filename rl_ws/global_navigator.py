"""
global_navigator.py — Navegacion global compartida: planeacion A* sobre la zona
segura de pallets + seguimiento con vortex APF (atraccion al waypoint, repulsion
de obstaculos/bordes con componente tangencial anti-minimo-local) + lookahead.

Lo usan los DOS backends (mujoco_sim_base.py directo y base_ros_env.py sobre el
bridge ROS2) y los scripts de visualizacion/tests. Todos los parametros
ajustables viven en base_training/config.py.
"""
import json
import math
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
        MAX_WAYPOINT_DIST,
        REACH_DIST, MAX_GUIDE_DIST, N_LOOKAHEAD, LOOKAHEAD_STEP,
        ATT_GAIN, ATT_RANGE, REP_GAIN, REP_RANGE, SWIRL, MIN_PROGRESS,
        PLATFORM_HALF_EXTENT, VIRTUAL_OBSTACLE_HALF_SIZE,
        VIRTUAL_OBSTACLE_MIN_HALF_SIZE, VIRTUAL_OBSTACLE_OFFSET_FRAC,
        ROBOT_TOP_M, GOAL_MIN_DIST_M, TRACK_DEFS, START_XY, track_get,
        ROBOT_HALF_WIDTH, ROBOT_HALF_LENGTH, PLAN_N_HEADINGS, PLAN_TURN_COST,
        PLAN_REVERSE_COST, PLAN_ARC_RADII, TURN_SUBSAMPLE,
    )
except ModuleNotFoundError:
    from base_training.config import (
        ROBOT_RADIUS, GAP_BRIDGE_DISTANCE, GRID_RESOLUTION,
        MAX_WAYPOINT_DIST,
        REACH_DIST, MAX_GUIDE_DIST, N_LOOKAHEAD, LOOKAHEAD_STEP,
        ATT_GAIN, ATT_RANGE, REP_GAIN, REP_RANGE, SWIRL, MIN_PROGRESS,
        PLATFORM_HALF_EXTENT, VIRTUAL_OBSTACLE_HALF_SIZE,
        VIRTUAL_OBSTACLE_MIN_HALF_SIZE, VIRTUAL_OBSTACLE_OFFSET_FRAC,
        ROBOT_TOP_M, GOAL_MIN_DIST_M, TRACK_DEFS, START_XY, track_get,
        ROBOT_HALF_WIDTH, ROBOT_HALF_LENGTH, PLAN_N_HEADINGS, PLAN_TURN_COST,
        PLAN_REVERSE_COST, PLAN_ARC_RADII, TURN_SUBSAMPLE,
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
               rh: float = ROBOT_HALF_WIDTH, swirl: float = SWIRL,
               min_progress: float = MIN_PROGRESS,
               obs_bounds: np.ndarray = None) -> np.ndarray:
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
    # PREFILTRO vectorizado: con rep_range de pocos cm, en una pista con paredes
    # (el maze tiene 91) casi ninguna esta en rango, pero el bucle las recorria
    # TODAS en cada paso (medido: 11.5 -> 591.6 us). `obs_bounds` es el array
    # (N,4) cacheado por el llamante; se calcula el hueco de golpe y solo se
    # entra al cuerpo por las pocas que de verdad repelen.
    if obs_bounds is not None and len(obs_bounds):
        gx = np.maximum(np.maximum(obs_bounds[:, 0] - rb[2], rb[0] - obs_bounds[:, 2]), 0.0)
        gy = np.maximum(np.maximum(obs_bounds[:, 1] - rb[3], rb[1] - obs_bounds[:, 3]), 0.0)
        cerca = np.nonzero(np.hypot(gx, gy) < rep_range)[0]
        obstacles = [obstacles[i] for i in cerca]
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


# ── Planeacion comun a todas las pistas ─────────────────────────────────────
# UN solo planificador (TrackMap) para todas: A* consciente de la huella sobre
# (x, y, rumbo). Lo que cambia por pista es solo de donde salen las cajas, y de
# eso se encargan los EXTRACTORES (obstacles_from_mujoco, boxes_from_pallets_json)
# mas abajo -- la capa intermedia. Añadir una pista es escribir su extractor.
_TRACK_MAPS: dict = {}


def _wrap_pi(a: float) -> float:
    """Angulo a (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _bresenham(a: tuple, b: tuple) -> list:
    """Celdas de la rejilla que pisa el segmento a->b, extremos incluidos."""
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    out = []
    while True:
        out.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return out
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


class TrackMap:
    """Mapa del laberinto + planeacion A* consciente de la huella, una vez por XML.

    Se arma leyendo el modelo MuJoCo de la pista, no una lista escrita a mano:
      1. Cajas de colision que NO son del robot (todas las paredes del maze son
         cajas). Se DESCARTAN las que empiezan por encima de `robot_top`: son
         dinteles de puerta y el robot pasa por debajo (68 de 159 en este maze).
         Tratarlas como pared taparia pasillos que si son transitables.
      2. Rasterizado a rejilla de `res` m/celda, acotada al bbox de las paredes.
         Fuera del bbox se marca OCUPADO: asi el exterior del laberinto queda
         descartado y el waypoint nunca sale de la pista.

    POR QUE LA ZONA SEGURA NO ES UN CIRCULO. Un solo radio no puede describir a
    este chasis, porque avanzar y girar piden cosas distintas (medido sobre la
    huella real de 0.55 x 0.57 m):
        avanzar recto  -> holgura lateral >= 0.274 m (media-anchura)
        girar en sitio -> holgura         >= 0.397 m (radio circunscrito)
    Con ROBOT_RADIUS=0.35 (entre los dos) el A* devolvia rutas transitables pero
    con giros en celdas donde el chasis NO cabe rotando -- el robot entraba al
    pasillo y se clavaba en la primera esquina. Subir el radio a 0.40 no arregla
    nada: fragmenta el laberinto en 5 componentes y deja el spawn en un bolsillo
    de 0.9 m2 (de 27.8), con lo que ni hay rutas.

    La solucion es meter la ORIENTACION en el estado y usar la huella
    RECTANGULAR, no un disco:
      - `self.free[h]`  (h = 0..N_HEAD-1): celdas donde el rectangulo ROTADO al
        rumbo h cabe sin tocar pared. Una mascara por rumbo, obtenida dilatando
        las paredes con el rectangulo rotado.
      - `self.rotable`: celdas donde cabe el disco circunscrito -> donde el robot
        puede girar sobre si mismo.
    El A* avanza solo en la direccion a la que MIRA (que es lo que puede hacer un
    diferencial) y solo puede girar donde `rotable`. Con eso descubre por si solo
    lo que hace un conductor: cruzar el estrecho recto y colocar el giro en el
    ensanchamiento. Medido: desde cualquier punto transitable hay una celda
    `rotable` a <= 0.35 m, asi que las rutas existen.

    Con N_HEAD=16 los 8 rumbos impares no apuntan a un vecino de la rejilla, asi
    que el avance usa un LATTICE de primitivas ((2,1) celdas para 22.5 grados,
    etc.) y se comprueban las celdas intermedias. Error angular maximo 4 grados.
    """

    def __init__(self, obstacles, drivable=None, spawn_xy=(0.0, 0.0),
                 spawn_yaw: float = 0.0, res: float = GRID_RESOLUTION,
                 half_width: float = ROBOT_HALF_WIDTH,
                 half_length: float = ROBOT_HALF_LENGTH,
                 n_headings: int = PLAN_N_HEADINGS,
                 gap_bridge: float = 0.0, margin: float = 0.5, name: str = "",
                 compute_reach: bool = True):
        """obstacles / drivable: AABB (N,4) = (x0,y0,x1,y1) en el mundo.

          obstacles  lo que NO se puede pisar (paredes del maze, sticks...).
          drivable   si se da, SOLO se puede pisar dentro de estas cajas y todo
                     lo demas es vacio (pallets: se conduce ENCIMA de las
                     tarimas y fuera se cae). Si es None, todo el bbox es piso.
          gap_bridge cierra huecos menores que esto entre cajas de `drivable`
                     (pallets pegadas que dejan una junta de pocos cm).
        """
        from scipy import ndimage

        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
        drv = None if drivable is None else np.asarray(drivable, dtype=float).reshape(-1, 4)
        if obstacles.shape[0] == 0 and drv is None:
            raise RuntimeError(f"{name}: ni obstaculos ni zona transitable")

        # Se guardan las cajas para poder DIBUJAR la pista: es la unica
        # descripcion 2D fiel que hay (el mesh visual no sirve de fondo y el .bt
        # del heatmap colapsa la altura). La usa tests/test_base.py.
        self.walls = obstacles
        self.drivable_boxes = drv
        self.name = name
        self.res = float(res)

        allb = obstacles if drv is None else np.vstack([obstacles, drv]) if obstacles.shape[0] else drv
        self.origin = allb[:, :2].min(axis=0) - margin
        upper = allb[:, 2:].max(axis=0) + margin
        self.dims = np.ceil((upper - self.origin) / self.res).astype(int) + 1

        if drv is None:
            occ = np.zeros(tuple(self.dims), dtype=bool)
        else:
            # Todo vacio salvo lo que sea explicitamente pisable.
            occ = np.ones(tuple(self.dims), dtype=bool)
            for x0, y0, x1, y1 in drv:
                i0, j0 = np.maximum(np.ceil((([x0, y0] - self.origin)) / self.res
                                            ).astype(int), 0)
                i1, j1 = np.minimum(np.floor((([x1, y1] - self.origin)) / self.res
                                             ).astype(int), self.dims - 1)
                occ[i0:i1 + 1, j0:j1 + 1] = False
            if gap_bridge > 0.0:
                # Cierre morfologico: junta tarimas contiguas separadas por una
                # junta mas fina que gap_bridge (si no, el planificador ve una
                # grieta infranqueable donde el robot pasa sin enterarse).
                k = max(1, int(round(gap_bridge / self.res)))
                free = ndimage.binary_closing(~occ, structure=np.ones((k, k), bool))
                occ = ~free

        # Obstaculos: rasterizado por SOLAPAMIENTO REAL de la celda. La celda c
        # cubre [c*res - res/2, c*res + res/2] (su centro es c*res, la convencion
        # de _cell/_world) y se marca si ese intervalo corta la caja. Antes era
        # floor/ceil MAS un +1 en el slice, con lo que un tabique de 5 cm mal
        # alineado ocupaba hasta 3 celdas = 15 cm: el pasillo se estrechaba hasta
        # 10 cm en el modelo y el planificador declaraba imposibles pasos que el
        # robot hace con holgura. Medido contra la geometria exacta, aquel
        # rasterizado subestimaba la holgura hasta 5 cm; este, 0.0 cm.
        half = self.res / 2.0
        for x0, y0, x1, y1 in obstacles:
            i0, j0 = np.maximum(np.ceil((([x0, y0] - self.origin) - half) / self.res
                                        ).astype(int), 0)
            i1, j1 = np.minimum(np.floor((([x1, y1] - self.origin) + half) / self.res
                                         ).astype(int), self.dims - 1)
            occ[i0:i1 + 1, j0:j1 + 1] = True

        self.n_head = int(n_headings)
        self.half_width = float(half_width)
        self.half_length = float(half_length)
        self._r_rot = float(np.hypot(half_width, half_length))

        # Borde del bbox = pared virtual -> el area navegable no se escapa fuera.
        occ_pad = np.pad(occ, 1, constant_values=True)

        # ── Huellas a resolucion FINA (SUB sub-angulos por rumbo) ────────────
        # De aqui salen las dos cosas: las mascaras de avance por rumbo (que son
        # un submuestreo de estas) y el barrido de cada giro. Sub-muestrear hace
        # falta porque el barrido es CONTINUO: entre dos rumbos del lattice el
        # rectangulo pasa por angulos intermedios que sobresalen en las esquinas.
        nfine = self.n_head * TURN_SUBSAMPLE
        fine = [self._footprint_free(occ_pad, 2.0 * np.pi * j / nfine)
                for j in range(nfine)]

        # ── Mascara de AVANCE, una por rumbo: la huella RECTANGULAR rotada ───
        self.free = [fine[h * TURN_SUBSAMPLE] for h in range(self.n_head)]

        # ── Mascaras de GIRO, una POR CANTIDAD DE GIRO ───────────────────────
        # El barrido de rotar de `h` a `nh` es la union de la huella en todos los
        # angulos intermedios, luego la celda vale si y solo si TODOS caben ahi:
        # un AND de mascaras que ya tenemos, sin una sola dilatacion extra.
        self._rot_ok = [[None] * self.n_head for _ in range(self.n_head)]
        half = self.n_head // 2
        for h in range(self.n_head):
            for sgn in (+1, -1):                  # se gira por el lado mas corto
                acc = fine[h * TURN_SUBSAMPLE].copy()
                for k in range(1, half + 1):
                    for s in range(1, TURN_SUBSAMPLE + 1):
                        acc &= fine[(h * TURN_SUBSAMPLE + sgn * ((k - 1) * TURN_SUBSAMPLE + s)) % nfine]
                    nh = (h + sgn * k) % self.n_head
                    prev = self._rot_ok[h][nh]
                    # d == n_head/2: los dos sentidos empatan en coste, vale
                    # cualquiera de los dos -> OR. Para el resto solo entra el corto.
                    self._rot_ok[h][nh] = acc.copy() if prev is None else (prev | acc)

        # Disco circunscrito: ya NO decide los giros del lattice (lo hace _rot_ok),
        # pero se conserva porque overlay_masks lo usa para el obstaculo movil y
        # porque es la referencia que se reporta.
        dist = ndimage.distance_transform_edt(~occ_pad)[1:-1, 1:-1] * self.res
        self.rotable = dist >= self._r_rot

        # Primitivas del lattice: el paso entero que mejor aproxima cada rumbo.
        self._moves = self._build_moves()
        # arcos: girar mientras se avanza. Ver _build_arcs.
        self._arcs = self._build_arcs(occ_pad)

        # ── Alcanzabilidad REAL: BFS sobre (celda, rumbo) ───────────────────
        spawn_cell = self._cell(spawn_xy)
        h0 = self._yaw_to_head(spawn_yaw)
        if not self.free[h0][spawn_cell]:
            print(f"[maze_map] AVISO: el spawn {tuple(spawn_xy)} con yaw "
                  f"{spawn_yaw:.2f} NO cabe (la huella toca pared); se arranca "
                  f"del estado viable mas cercano.")
        # compute_reach=False para areas ABIERTAS (platform): el BFS sobre
        # (celda,rumbo) costaria minutos sobre 1.3M estados y no aporta nada
        # donde todo es alcanzable. Sin el, `navigable` = cabe en algun rumbo, y
        # plan() usa A* en vez del arbol cacheado.
        self._reach = self._reachable_from(spawn_cell, h0) if compute_reach else None
        self.navigable = (self._reach.any(axis=0) if self._reach is not None
                          else np.stack(self.free).any(axis=0))
        self._free_cells = np.argwhere(self.navigable)
        if self._free_cells.shape[0] == 0:
            raise RuntimeError(
                f"{self.name}: nada alcanzable desde el spawn {tuple(spawn_xy)}. "
                f"Huella {2*half_width:.2f}x{2*half_length:.2f} m; girar necesita "
                f"{self._r_rot:.3f} m de holgura.")
        self._plan_cache = {}
        self._tree_cache = {}
        # Las cajas crudas se guardan para el VORTEX: el planificador las rasteriza
        # a rejilla, pero la guia reactiva las necesita como geometria para repeler
        # (ver vortex_obstacles). Sin esto el maze corria con el vortex CIEGO a sus
        # paredes -- medido: el 16.4% de los puntos-guia caian donde el robot no cabe.
        self.obstacle_boxes = np.asarray(obstacles, dtype=float).reshape(-1, 4)
        print(f"[trackmap] {self.name}: {len(obstacles)} obstaculos, rejilla {tuple(self.dims)} "
              f"@{self.res} m, {self.n_head} rumbos | huella "
              f"{2*half_width:.2f}x{2*half_length:.2f} m (giro >= {self._r_rot:.3f} m) | "
              f"alcanzable {self._free_cells.shape[0] * self.res**2:.1f} m2, "
              f"rotable {self.rotable.sum() * self.res**2:.1f} m2")

    # ── Mascaras, primitivas y sucesores ────────────────────────────────────
    def _footprint_free(self, occ_pad: np.ndarray, theta: float) -> np.ndarray:
        """Celdas donde el rectangulo del robot ROTADO a `theta` no toca pared.

        Se dilata la ocupacion con el rectangulo rotado como elemento
        estructurante: una celda queda bloqueada si al centrar ahi la huella esta
        pisa pared. Es la version "cuadrado" de la erosion por radio."""
        from scipy import ndimage
        hl, hw = self.half_length, self.half_width
        rad = int(np.ceil(np.hypot(hl, hw) / self.res))
        di, dj = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1),
                             indexing="ij")
        c, s = np.cos(theta), np.sin(theta)
        wx, wy = di * self.res, dj * self.res
        xp = wx * c + wy * s          # offset en el frame del ROBOT
        yp = -wx * s + wy * c
        kernel = (np.abs(xp) <= hl) & (np.abs(yp) <= hw)
        blocked = ndimage.binary_dilation(occ_pad, structure=kernel)
        return ~blocked[1:-1, 1:-1]

    def _build_moves(self) -> list:
        """(di, dj, coste) por rumbo: paso entero que mejor aproxima la direccion.
        Los rumbos multiplos de 45 grados caen en un vecino; los intermedios usan
        pasos de 2 celdas ((2,1) = 26.6 grados para el rumbo de 22.5)."""
        cand = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
                (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]
        moves = []
        for h in range(self.n_head):
            ang = 2.0 * np.pi * h / self.n_head
            best = min(cand, key=lambda v: abs(_wrap_pi(math.atan2(v[1], v[0]) - ang)))
            moves.append((best[0], best[1], float(np.hypot(best[0], best[1]))))
        return moves

    def _build_arcs(self, occ_pad):
        """Primitivas de ARCO: girar 360/N grados MIENTRAS se avanza.

        Es lo que le faltaba al lattice. Con solo "avanzar recto" y "rotar en
        sitio", cada curva de un pasillo obligaba a PARARSE y rotar, y rotar
        exige el disco circunscrito entero -- holgura que en un pasillo no hay.
        Un diferencial toma la curva girando mientras avanza: la huella barre una
        BANDA a lo largo del arco, no un disco en un punto. MEDIDO: pallets pasa
        de 2.5 a 21.2 m2 alcanzables y el maze de 37.1 a 71.0.

        Devuelve {(rumbo, sentido, i): (di, dj, mascara, coste)}, donde la
        mascara dice, POR CELDA DE PARTIDA, si el barrido completo cabe. Se
        precalcula una vez por pista (2 radios x N rumbos x 2 sentidos)."""
        from scipy import ndimage
        arcs = {}
        if not PLAN_ARC_RADII:
            return arcs
        res, hl, hw = self.res, self.half_length, self.half_width
        dth = 2.0 * np.pi / self.n_head
        for ri, R in enumerate(PLAN_ARC_RADII):
            for h in range(self.n_head):
                a = 2.0 * np.pi * h / self.n_head
                for sgn in (+1, -1):
                    # Centro de giro: a R metros perpendicular al rumbo.
                    ctr = np.array([-np.sin(a), np.cos(a)]) * (R * sgn)
                    poses = []
                    for t in np.linspace(0.0, 1.0, 17):
                        th = t * dth * sgn
                        rot = np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th),  np.cos(th)]])
                        poses.append((ctr + rot @ (-ctr), a + th))
                    disp = poses[-1][0]
                    rad = int(np.ceil((np.hypot(hl, hw) + np.linalg.norm(disp)) / res)) + 1
                    di, dj = np.meshgrid(np.arange(-rad, rad + 1),
                                         np.arange(-rad, rad + 1), indexing="ij")
                    wx, wy = di * res, dj * res
                    kern = np.zeros(wx.shape, dtype=bool)
                    for pos, th in poses:
                        ox, oy = wx - pos[0], wy - pos[1]
                        xp = ox * np.cos(th) + oy * np.sin(th)
                        yp = -ox * np.sin(th) + oy * np.cos(th)
                        kern |= (np.abs(xp) <= hl) & (np.abs(yp) <= hw)
                    free = ~ndimage.binary_dilation(occ_pad, structure=kern)[1:-1, 1:-1]
                    # Coste = LONGITUD DEL ARCO (>= la cuerda), asi la heuristica
                    # euclidea del A* sigue siendo admisible.
                    arcs[(h, sgn, ri)] = (int(round(disp[0] / res)),
                                          int(round(disp[1] / res)),
                                          free, float(R * dth / res))
        return arcs

    def vortex_obstacles(self) -> list:
        """Paredes de la pista como Obstacle2D, para la repulsion del vortex.
        Misma geometria que rasteriza el planificador, asi que guia reactiva y
        ruta global ven el MISMO mundo en cualquier pista."""
        return [Obstacle2D(f"wall_{i}", (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                           (x1 - x0) / 2.0, (y1 - y0) / 2.0)
                for i, (x0, y0, x1, y1) in enumerate(self.obstacle_boxes)]

    def _yaw_to_head(self, yaw: float) -> int:
        return int(round((yaw % (2.0 * np.pi)) / (2.0 * np.pi) * self.n_head)) % self.n_head

    def _successors(self, cell: tuple, h: int, blocked=None):
        """(celda, rumbo, coste) en un paso. LA REGLA del planificador:
        avanzar SOLO hacia donde mira y solo si la huella a ese rumbo cabe en
        todo el trayecto; girar a CUALQUIER rumbo, solo donde `rotable`.

        El giro NO es incremental. Encadenar saltos de 360/N grados no aportaba
        nada -- todos exigen `rotable` en la MISMA celda, asi que la factibilidad
        es identica -- y a cambio inflaba la busqueda y salian rutas quebradas.
        La ruta es GUIA para una politica que aprende a ejecutarla, no una
        trayectoria: al planificador le toca decir por donde se puede pasar, no
        como mover las orugas. El coste sigue siendo proporcional al angulo para
        que A* prefiera rutas con menos giro.

        `blocked`: OVERLAY DINAMICO -- estados (celda,rumbo) que una caja movil
        bloquea SOLO en este episodio (el obstaculo virtual de
        las pistas platform, que se recoloca en cada reset). Se pasan aparte en
        vez de recalcular las mascaras porque rehacerlas cuesta ~1 s y esto son
        microsegundos: la caja afecta a una ventana de ~1.3 m."""
        di, dj, cost = self._moves[h]
        fh = self.free[h]
        W, H = int(self.dims[0]), int(self.dims[1])
        # Avanzar (sign=+1) y RETROCEDER (sign=-1). La huella es la misma en los
        # dos casos -- el rectangulo no cambia por ir marcha atras -- asi que se
        # comprueba `free[h]` igual; solo cambia el coste.
        for sign, mult in ((1, 1.0), (-1, PLAN_REVERSE_COST)):
            nb = (cell[0] + sign * di, cell[1] + sign * dj)
            if 0 <= nb[0] < W and 0 <= nb[1] < H:
                if all(fh[c] for c in _bresenham(cell, nb)) and (
                        blocked is None or not any((c, h) in blocked
                                                   for c in _bresenham(cell, nb))):
                    yield nb, h, cost * mult
        for nh in range(self.n_head):
            if nh == h:
                continue
            # Barrido de 45 grados pide
            # bastante menos hueco que dar la vuelta entera. Mismo criterio para
            # las paredes (mascara precalculada) y para la caja movil (overlay).
            if not self._rot_ok[h][nh][cell]:
                continue
            if self._turn_hits_overlay(blocked, cell, h, nh):
                continue
            d = (nh - h) % self.n_head
            yield cell, nh, PLAN_TURN_COST * min(d, self.n_head - d)
        # arco: cambiar de rumbo sin pararse. No exige `rotable` -- el barrido a
        # lo largo del arco es mucho menor que el disco de una rotacion en sitio.
        for (ah, sgn, _ri), (dx, dy, free, cost) in self._arcs.items():
            if ah != h:
                continue
            nb = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nb[0] < W and 0 <= nb[1] < H) or not free[cell]:
                continue
            nh = (h + sgn) % self.n_head
            if not self.free[nh][nb]:
                continue
            if blocked is not None and any((c, h) in blocked or (c, nh) in blocked
                                           for c in _bresenham(cell, nb)):
                continue
            yield nb, nh, cost

    # ── Overlay dinamico ────────────────────────────────────────────────────
    def overlay_masks(self, boxes):
        """Cajas moviles -> `blocked` = {(celda, rumbo)} donde la huella a ese
        rumbo tocaria la caja. Solo se recorre la ventana que la caja puede
        afectar (su AABB inflado por el radio circunscrito), no la rejilla entera.

        Con esto basta TAMBIEN para los giros: rotar de h a nh barre los rumbos
        intermedios, asi que el giro toca la caja si y solo si alguno de esos
        rumbos esta en `blocked` (ver _turn_hits_overlay). Antes se devolvia
        ademas un `no_rot` con las celdas donde no cabia el disco circunscrito
        -- el barrido de girar 360 grados -- lo que prohibia girar cerca de la
        caja aunque el giro fuera de 45. Era la ultima parte del planificador que
        seguia con el modelo del disco: ahora paredes estaticas y obstaculo movil
        usan el MISMO criterio por cantidad de giro, y sale gratis porque
        `blocked` ya se calculaba."""
        blocked = set()
        if boxes is None:
            return blocked
        pad = self._r_rot + self.res
        hl, hw = self.half_length, self.half_width
        grow = 0.5 * self.res
        for x0, y0, x1, y1 in np.asarray(boxes, dtype=float).reshape(-1, 4):
            x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
            b = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            bh = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0])
            i0, j0 = self._cell((x0 - pad, y0 - pad))
            i1, j1 = self._cell((x1 + pad, y1 + pad))
            ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
            px = self.origin[0] + ii * self.res
            py = self.origin[1] + jj * self.res
            dx, dy = px - b[0], py - b[1]
            # SAT entre el rectangulo a rumbo h y el AABB de la caja. Sirve para
            # avanzar (rumbo fijo) y, por union sobre el arco, tambien para girar.
            for h in range(self.n_head):
                th = 2.0 * np.pi * h / self.n_head
                c, s = np.cos(th), np.sin(th)
                sep = (np.abs(dx) > bh[0] + hl * abs(c) + hw * abs(s))
                sep |= (np.abs(dy) > bh[1] + hl * abs(s) + hw * abs(c))
                sep |= (np.abs(dx * c + dy * s) > hl + bh[0] * abs(c) + bh[1] * abs(s))
                sep |= (np.abs(-dx * s + dy * c) > hw + bh[0] * abs(s) + bh[1] * abs(c))
                hit = ~sep
                for a, bcell in zip(ii[hit].ravel(), jj[hit].ravel()):
                    blocked.add(((int(a), int(bcell)), h))
        return blocked

    def _turn_hits_overlay(self, blocked, cell: tuple, h: int, nh: int) -> bool:
        """True si girar de `h` a `nh` en `cell` barreria la caja movil. Mismo
        criterio que `_rot_ok` usa con las paredes -- la union de la huella sobre
        los rumbos intermedios -- pero leyendo el `blocked` del overlay en vez de
        una mascara precalculada, porque la caja cambia en cada episodio."""
        if not blocked:
            return False
        n = self.n_head
        d = (nh - h) % n
        step = 1 if d <= n - d else -1              # se gira por el lado mas corto
        for s in range(min(d, n - d) + 1):
            if (cell, (h + step * s) % n) in blocked:
                return True
        return False

    def _reachable_from(self, cell: tuple, h0: int) -> np.ndarray:
        """(n_head, W, H) bool: estados alcanzables desde (cell, h0)."""
        from collections import deque
        reach = np.zeros((self.n_head,) + tuple(self.dims), dtype=bool)
        # SOLO el estado inicial real. Antes se sembraba la celda del spawn en
        # TODOS los rumbos que cupieran, lo que regalaba una rotacion gratis ahi
        # aunque la celda no fuera `rotable`: `navigable` decia 31 m2 mientras el
        # planificador de verdad solo alcanzaba 33 celdas en linea recta, y
        # sample_goal sorteaba metas sin ruta. La alcanzabilidad TIENE que usar
        # exactamente los mismos movimientos que plan().
        if self.free[h0][cell]:
            starts = [(cell, h0)]
        else:               # spawn pegado a pared: arrancar del viable mas cercano
            viable = np.argwhere(np.stack(self.free).any(axis=0))
            if viable.shape[0] == 0:
                return reach
            k = int(np.abs(viable - np.asarray(cell)).sum(axis=1).argmin())
            c2 = (int(viable[k][0]), int(viable[k][1]))
            starts = [(c2, h) for h in range(self.n_head) if self.free[h][c2]]
        q = deque()
        for c, h in starts:
            if not reach[h][c]:
                reach[h][c] = True
                q.append((c, h))
        while q:
            c, h = q.popleft()
            for nc, nh, _ in self._successors(c, h):
                if not reach[nh][nc]:
                    reach[nh][nc] = True
                    q.append((nc, nh))
        return reach

    # ── Conversion celda <-> mundo ──────────────────────────────────────────
    def _cell(self, xy) -> tuple:
        c = np.round((np.asarray(xy, dtype=float) - self.origin) / self.res).astype(int)
        return tuple(np.clip(c, 0, self.dims - 1))

    def _world(self, cell) -> np.ndarray:
        return self.origin + np.asarray(cell, dtype=float) * self.res

    def is_navigable(self, xy) -> bool:
        return bool(self.navigable[self._cell(xy)])

    def _nearest_navigable(self, xy) -> tuple:
        """Celda navegable mas cercana a xy. Necesario porque el robot REAL
        puede estar dentro de la banda inflada (rozando una pared) y desde ahi
        A* no arrancaria."""
        cell = self._cell(xy)
        if self.navigable[cell]:
            return cell
        d = np.abs(self._free_cells - np.asarray(cell)).sum(axis=1)
        return tuple(self._free_cells[int(np.argmin(d))])

    # ── Muestreo del waypoint ───────────────────────────────────────────────
    def sample_goal(self, from_xy, min_dist: float = GOAL_MIN_DIST_M) -> np.ndarray:
        """Celda navegable aleatoria a >= min_dist del robot. Si ninguna cumple
        (laberinto chico), se relaja al punto navegable mas lejano que haya."""
        pts = self.origin + self._free_cells * self.res
        far = pts[np.linalg.norm(pts - np.asarray(from_xy, dtype=float), axis=1) >= min_dist]
        if far.shape[0] == 0:
            d = np.linalg.norm(pts - np.asarray(from_xy, dtype=float), axis=1)
            return pts[int(np.argmax(d))].copy()
        return far[np.random.randint(far.shape[0])].copy()

    # ── A* sobre (celda, rumbo) ─────────────────────────────────────────────
    def plan(self, start_xy, goal_xy, max_wp_dist: float = MAX_WAYPOINT_DIST,
             start_yaw: float = 0.0, overlay=None) -> list[tuple]:
        """Ruta start->goal como lista de waypoints (x,y).

        A* sobre (celda, RUMBO): avanzar solo hacia donde el robot mira, girar
        solo donde cabe rotando. El rumbo se usa unicamente para decidir que es
        factible -- lo que sale sigue siendo una polilinea plana, asi que el
        vortex y la observacion no cambian.

        DOS caminos segun la pista:
          - estatica y con spawn fijo (maze, pallets) -> arbol de Dijkstra
            completo desde el arranque, cacheado: cada ruta es un backtrack.
          - con `overlay` (obstaculo movil) o spawn aleatorio (platform) -> A*
            hacia esa meta concreta. El arbol no serviria: cambia cada episodio.
        """
        start = self._nearest_navigable(start_xy)
        goal = self._nearest_navigable(goal_xy)
        h0 = self._yaw_to_head(start_yaw)
        if self._reach is not None and not self._reach[h0][start]:
            cand = [h for h in range(self.n_head) if self._reach[h][start]]
            if cand:
                h0 = min(cand, key=lambda h: abs(_wrap_pi(
                    2.0 * np.pi * (h - h0) / self.n_head)))
        if start == goal:
            return _resample_polyline([np.asarray(start_xy, dtype=float),
                                       self._world(goal)], max_wp_dist)

        if overlay is not None or self._reach is None:
            blocked = self.overlay_masks(overlay)
            _, parent, end = self._search(start, h0, goal, blocked)
            if end is None:
                # Ni un paso viable: el robot esta encerrado. Se queda donde esta
                # (un waypoint en su sitio) en vez de recibir una recta que
                # atraviesa obstaculos.
                print(f"[trackmap] {self.name}: sin salida desde "
                      f"{tuple(np.round(start_xy,2))}")
                return [tuple(np.asarray(start_xy, dtype=float))]
            cells = self._shortcut(self._backtrack(parent, end), blocked)
            pts = [np.asarray(start_xy, dtype=float)] + [self._world(c) for c, _ in cells[1:]]
            return _resample_polyline(pts, max_wp_dist)

        cells = self._plan_cache.get((start, h0, goal))
        if cells is None:
            dist, parent, _ = self._search(start, h0)
            # De todos los rumbos posibles al llegar a la meta, el mas barato.
            end, best = None, float("inf")
            for h in range(self.n_head):
                d = dist.get((goal, h))
                if d is not None and d < best:
                    end, best = (goal, h), d
            if end is None:
                # No deberia pasar: la meta se sortea sobre `self.navigable`, que
                # YA es la alcanzabilidad real en (celda, rumbo).
                print(f"[maze_map] AVISO: sin ruta {tuple(start_xy)} -> "
                      f"{tuple(goal_xy)} (rumbo inicial {h0})")
                return _resample_polyline([np.asarray(start_xy, dtype=float),
                                           np.asarray(goal_xy, dtype=float)], max_wp_dist)
            cells = self._shortcut(self._backtrack(parent, end))
            self._plan_cache[(start, h0, goal)] = cells

        pts = [np.asarray(start_xy, dtype=float)] + [self._world(c) for c, _ in cells[1:]]
        return _resample_polyline(pts, max_wp_dist)

    def _search(self, start: tuple, h0: int, goal: tuple = None,
                blocked=None):
        """Busqueda UNICA sobre (celda, rumbo). Dijkstra y A* son el MISMO bucle:
        la unica diferencia es la heuristica, y A* con heuristica 0 ES Dijkstra.
        Devuelve (dist, parent, end).

          goal=None  -> explora TODO el grafo y CACHEA el arbol. En pistas con
                        spawn fijo (maze, pallets) un unico arbol sirve para
                        TODAS las metas: cada plan() es un backtrack en vez de
                        una busqueda (medido: 1876 ms el arbol, 11 ms por ruta
                        despues). end=None, el caller elige a que estado-meta ir.
          goal=(i,j) -> A* con heuristica euclidea en celdas (admisible: avanzar
                        cuesta la distancia y girar solo suma). Para cuando el
                        arbol NO se puede reutilizar porque cambia cada episodio:
                        hay `overlay` (obstaculo movil) o el spawn es aleatorio
                        (platform). `end` es el estado meta, o el nodo mas cercano
                        alcanzado (MEJOR ESFUERZO), o None si no se dio un paso.

        El mejor-esfuerzo existe porque devolver una recta cruda al fallar era
        peor que inutil: pasa por encima de lo que haya en medio y le enseña a la
        politica a conducir contra el obstaculo.

        El estado INICIAL se admite siempre, aunque el overlay lo marque en
        colision: es donde el robot ESTA. Puede pasar (medido) que el obstaculo
        virtual se coloque solapando la huella del spawn; prohibir esa pose
        dejaba el problema sin solucion y se caia al fallback de recta cruda.
        """
        import heapq
        # El arbol completo (goal=None, sin overlay) es lo unico cacheable: con
        # meta o con overlay el resultado vale solo para ESE episodio.
        cacheable = goal is None and blocked is None
        if cacheable:
            cached = self._tree_cache.get((start, h0))
            if cached is not None:
                return cached[0], cached[1], None

        dist = {(start, h0): 0.0}
        parent = {}
        heap = [(0.0, start, h0)]
        closed = set()
        end = None
        # Mejor nodo visto por cercania a la meta, para el mejor-esfuerzo.
        best_node = (start, h0)
        best_h = (float("inf") if goal is None
                  else math.hypot(goal[0] - start[0], goal[1] - start[1]))
        while heap:
            _, cur, ch = heapq.heappop(heap)
            if (cur, ch) in closed:
                continue
            if goal is not None and cur == goal:
                end = (cur, ch)
                break
            closed.add((cur, ch))
            gc = dist[(cur, ch)]
            for nb, nh, step in self._successors(cur, ch, blocked):
                if (nb, nh) in closed:
                    continue
                t = gc + step
                if t < dist.get((nb, nh), float("inf")):
                    dist[(nb, nh)] = t
                    parent[(nb, nh)] = (cur, ch)
                    hh = 0.0                      # heuristica 0 -> Dijkstra
                    if goal is not None:
                        hh = math.hypot(goal[0] - nb[0], goal[1] - nb[1])
                        if hh < best_h:
                            best_h, best_node = hh, (nb, nh)
                    heapq.heappush(heap, (t + hh, nb, nh))

        if goal is not None and end is None and best_node != (start, h0):
            end = best_node
        if cacheable:
            self._tree_cache[(start, h0)] = (dist, parent)
        return dist, parent, end

    @staticmethod
    def _backtrack(parent: dict, end: tuple) -> list:
        """Cadena de estados (celda, rumbo) desde el arranque hasta `end`."""
        seq = [end]
        while seq[-1] in parent:
            seq.append(parent[seq[-1]])
        seq.reverse()
        return seq

    def _shortcut(self, states: list, blocked=None) -> list:
        """String-pulling CONSCIENTE DEL RUMBO sobre la secuencia (celda, rumbo).

        El atajo ingenuo desharia todo el trabajo: cambiaria un rodeo del tipo
        "recto -> giro en zona ancha -> recto" por una diagonal que exige girar
        en el estrecho. Aqui un tramo i->j solo se acepta si la huella cabe al
        rumbo de ESE tramo en todo el recorrido y, si hay que cambiar de rumbo
        para tomarlo, la celda i permite rotar. Asi se conserva la invariante:
        todo cambio de direccion ocurre donde el chasis puede girar."""
        out = [states[0]]
        i = 0
        cur_h = states[0][1]        # rumbo con el que se LLEGA al punto i
        while i < len(states) - 1:
            j = len(states) - 1
            chosen = None
            while j > i + 1:
                h = self._segment_head(states[i][0], states[j][0], cur_h,
                                       blocked)
                # INVARIANTE: solo se acepta el atajo si deja al robot con el
                # MISMO rumbo que traia el camino original en ese nodo. Asi
                # cur_h siempre coincide con states[i][1], y el caso de reserva
                # (avanzar al siguiente nodo del A*) es legal por construccion.
                # Sin esto, el reserva colaba un cambio de rumbo "gratis" en
                # celdas donde el chasis no puede rotar -- medido: 132 de 522.
                if h is not None and h == states[j][1]:
                    chosen = h
                    break
                j -= 1
            if chosen is None:
                j = i + 1
                chosen = states[j][1]
            out.append(states[j])
            cur_h = chosen
            i = j
        return out

    def _segment_ok(self, a: tuple, b: tuple) -> bool:
        """Compat: ¿tramo recto viable llegando con el rumbo guardado en `a`?"""
        return self._segment_head(a[0], b[0], a[1]) is not None

    def _segment_head(self, ca: tuple, cb: tuple, h_in: int,
                      blocked=None):
        """Rumbo del tramo recto ca->cb si es viable llegando con rumbo `h_in`,
        o None. OJO con `h_in`: es el rumbo con el que se LLEGA a ca DESPUES de
        atajar, no el que traia el camino original -- usar el original hacia que
        el atajo colase giros en celdas donde el chasis no puede rotar."""
        d = (cb[0] - ca[0], cb[1] - ca[1])
        if d == (0, 0):
            return h_in
        h_seg = self._yaw_to_head(math.atan2(d[1], d[0]))
        if h_seg != h_in and (not self._rot_ok[h_in][h_seg][ca]
                              or self._turn_hits_overlay(blocked, ca, h_in, h_seg)):
            return None            # habria que girar aqui, y aqui no cabe
        fh = self.free[h_seg]
        seg = _bresenham(ca, cb)
        if not all(fh[c] for c in seg):
            return None
        # El OVERLAY tambien manda aqui. Sin esto el atajo deshacia el trabajo del
        # A*: en una plaza abierta free[h] es cierto en todas partes, asi que el
        # string-pulling reconectaba en recto por encima del obstaculo movil que
        # la busqueda acababa de esquivar (medido: 2 tramos cruzando la caja).
        if blocked is not None and any((c, h_seg) in blocked for c in seg):
            return None
        return h_seg


# ── Extractores: de la descripcion de CADA pista a cajas ────────────────────
# Es la capa intermedia. El planificador (TrackMap) no sabe si las cajas vienen
# de un XML de MuJoCo, de un JSON o de donde sea; solo recibe AABB. Añadir una
# pista nueva es escribir su extractor, no otro planificador.

def obstacles_from_mujoco(xml_path: str, robot_top: float = ROBOT_TOP_M):
    """Cajas de colision de un XML de MuJoCo que NO son del robot -> AABB (N,4).

    Descarta las que empiezan por encima de `robot_top`: son dinteles de puerta
    y el robot pasa por debajo (68 de 159 en el maze). Tratarlas como pared
    taparia pasillos que si son transitables."""
    # Import local a proposito: este modulo lo usan tambien scripts sin MuJoCo.
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)              # geom_xpos/xmat en frame mundo
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "footprint_link")
    boxes = []
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue                            # geom puramente visual
        if int(model.body_rootid[model.geom_bodyid[g]]) == robot_root:
            continue                            # el robot no es pared
        pos = np.asarray(data.geom_xpos[g], dtype=float)
        rot = np.asarray(data.geom_xmat[g], dtype=float).reshape(3, 3)
        # AABB de la caja (posiblemente rotada) = |R| @ half, conservador.
        ext = np.abs(rot) @ np.asarray(model.geom_size[g], dtype=float)
        if pos[2] - ext[2] >= robot_top:
            continue                            # dintel: se pasa por debajo
        boxes.append((pos[0] - ext[0], pos[1] - ext[1],
                      pos[0] + ext[0], pos[1] + ext[1]))
    if not boxes:
        raise RuntimeError(f"{xml_path}: no se encontro ninguna pared de colision")
    return np.asarray(boxes, dtype=float)


def boxes_from_pallets_json(json_path: str):
    """JSON de pallets -> (drivable, obstacles), ambos AABB (N,4).

    En esta pista la logica esta INVERTIDA respecto al maze: se conduce ENCIMA
    de las tarimas (`pallets`) y fuera de ellas se cae al vacio, asi que las
    tarimas son la zona transitable y no obstaculos. Los `sticks` (postes
    fatales) y los `obstacles` del JSON si son obstaculos."""
    with open(json_path, "r") as f:
        data = json.load(f)

    def _aabb(entries, skip=()):
        out = []
        for e in entries:
            if e.get("name") in skip:
                continue
            c = np.asarray(e["center_xy"], dtype=float)
            s = np.asarray(e["size"], dtype=float)
            hx, hy = (s[0], s[1]) if e.get("gtype") == "box" else (s[0], s[0])
            R = np.asarray(e.get("rot_mat", np.eye(3)), dtype=float).reshape(3, 3)
            ext = np.abs(R[:2, :2]) @ np.array([hx, hy])
            out.append((c[0] - ext[0], c[1] - ext[1], c[0] + ext[0], c[1] + ext[1]))
        return np.asarray(out, dtype=float).reshape(-1, 4)

    drivable = _aabb(data["pallets"])
    obstacles = np.vstack([_aabb(data.get("sticks", [])),
                           _aabb(data.get("obstacles", []), skip=("col_manija",))])
    return drivable, obstacles


def get_track_map(track: dict = None) -> TrackMap:
    """TrackMap de la pista, cacheado (construirlo cuesta ~1 s y los N envs de
    VecMujocoEnv comparten el mismo, que es de solo lectura).

    Aqui es donde se elige el EXTRACTOR segun el tipo de pista; el planificador
    que sale es el mismo para todas."""
    track = track if track is not None else TRACK_DEFS["maze"]
    kind = track.get("kind", "maze")
    key = (track.get("xml"), kind)
    if key in _TRACK_MAPS:
        return _TRACK_MAPS[key]

    if kind == "platform":
        # Plaza abierta: no hay obstaculos ESTATICOS, solo el borde. El unico
        # obstaculo (la caja virtual) se recoloca cada episodio y entra como
        # OVERLAY en plan(), no en las mascaras. Rejilla mas gruesa a proposito:
        # son 20x20 m con una sola caja de 60 cm, y a 0.05 serian 401x401 celdas
        # (4x el maze) para una precision que aqui no compra nada.
        he = PLATFORM_HALF_EXTENT
        tm = TrackMap(np.zeros((0, 4)), drivable=[(-he, -he, he, he)],
                      spawn_xy=(0.0, 0.0), res=track_get(track, "grid_res", GRID_RESOLUTION),
                      compute_reach=False, name=track.get("name", "platform"))
    elif kind == "pallets":
        drivable, obstacles = boxes_from_pallets_json(track["nav_json"])
        tm = TrackMap(obstacles, drivable=drivable,
                      spawn_xy=track_get(track, "spawn_xy", START_XY),
                      spawn_yaw=float(track_get(track, "spawn_yaw", 0.0)),
                      res=track_get(track, "grid_res", GRID_RESOLUTION),
                      gap_bridge=GAP_BRIDGE_DISTANCE, name=track.get("name", "pallets"))
    else:
        tm = TrackMap(obstacles_from_mujoco(track["xml"]),
                      spawn_xy=track_get(track, "spawn_xy", (0.0, 0.0)),
                      spawn_yaw=float(track_get(track, "spawn_yaw", 0.0)),
                      res=track_get(track, "grid_res", GRID_RESOLUTION),
                      name=track.get("name", "maze"))
    _TRACK_MAPS[key] = tm
    return tm


def make_navigator(track: dict, waypoints: list, extra_obstacles=None,
                   edges_zone=None, **kw) -> "GlobalNavigator":
    """GlobalNavigator de una pista, SIN que el llamante tenga que saber su `kind`.

    Existe para que los dos backends (directo-MuJoCo y ROS) construyan la guia
    igual y para que añadir una pista no obligue a tocar ninguno: toda la decision
    por tipo de pista vive AQUI. Antes cada backend tenia su propio if/elif y ya
    habian divergido -- pallets acababa con 42 obstaculos en ROS y 84 en
    entrenamiento (los mismos, contados dos veces).

    De donde salen los obstaculos del vortex, en UNA sola via:
      - la pista trae nav_json (pallets) -> el JSON ya aporta sticks/puerta y los
        bordes de tarima. NO se añade vortex_obstacles() ademas: el TrackMap de
        pallets se construye de ese MISMO json, asi que se duplicaria la repulsion.
      - sin nav_json (maze, platform) -> las cajas salen del XML de MuJoCo via
        TrackMap.obstacle_boxes, la misma geometria que rasteriza el planificador.
        En platform esa lista es vacia (no hay paredes), que es lo correcto.

    `extra_obstacles` es para lo DINAMICO (el obstaculo virtual del episodio);
    `edges_zone` para la zona que repele hacia adentro (plataforma). Si no se
    pasa y la pista es platform, se deriva sola.
    """
    nav_json = track.get("nav_json")
    obstacles = list(extra_obstacles or [])
    if not nav_json:
        obstacles += get_track_map(track).vortex_obstacles()
    if edges_zone is None and track.get("kind") == "platform":
        edges_zone = build_platform_zone()
    return GlobalNavigator(nav_json, waypoints=waypoints,
                           obstacles=obstacles or None, edges_zone=edges_zone, **kw)


def plan_track_route(start_xy: tuple, goal_xy: tuple = None,
                     max_wp_dist: float = MAX_WAYPOINT_DIST,
                     track: dict = None, start_yaw: float = 0.0,
                     overlay=None) -> tuple:
    """Ruta consciente de la huella dentro de la pista. Devuelve (waypoints, goal_xy).

    goal_xy=None (lo normal) -> se SORTEA una meta en zona libre y ALCANZABLE,
    a >= GOAL_MIN_DIST_M del robot. Devolver la meta es obligatorio: el que
    llama tiene que guardarla en env.goal_xy, porque de ahi salen tanto la
    condicion de exito como GOAL_BONUS.

    start_yaw importa: el planificador razona con la ORIENTACION del robot, asi
    que la ruta depende de hacia donde mira al empezar (girar cuesta, y en los
    pasillos estrechos ni siquiera se puede)."""
    mmap = get_track_map(track)
    if goal_xy is None:
        goal_xy = mmap.sample_goal(start_xy)
    goal_xy = np.asarray(goal_xy, dtype=float)
    return mmap.plan(start_xy, goal_xy, max_wp_dist, start_yaw=start_yaw,
                     overlay=overlay), goal_xy


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

    SOLO coloca la caja y devuelve una ruta de referencia; la ruta que se USA la
    da plan_track_route con esa caja como overlay dinamico (ver mujoco_sim_base y
    base_ros_env). Antes esta funcion era el planificador de platform y el
    docstring apuntaba a plan_route para pallets -- las dos cosas ya no existen.

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

class GlobalNavigator:
    def __init__(self, json_path, waypoints: list,
                 att_gain: float = ATT_GAIN, att_range: float = ATT_RANGE,
                 rep_gain: float = REP_GAIN, rep_range: float = REP_RANGE,
                 rh: float = ROBOT_HALF_WIDTH, swirl: float = SWIRL,
                 min_progress: float = MIN_PROGRESS,
                 pallet_edges: bool = True,
                 n_lookahead: int = N_LOOKAHEAD,
                 lookahead_step: float = LOOKAHEAD_STEP,
                 obstacles: list = None,
                 edges_zone=None):
        """json_path=None -> modo plataforma plana (sin pallets/sticks fisicos)"""
        self._wps = [np.array(w) for w in waypoints]

        # Obstaculos del vortex en DOS grupos: los ESTATICOS son la geometria de
        # la pista (paredes del maze, sticks/puerta de pallets) y viven todo el
        # episodio; los DINAMICOS son el obstaculo virtual, que replan() cambia
        # en cada reset. Antes habia una sola lista y replan(obstacles=[]) --
        # que se llama en varios sitios -- borraba tambien la geometria fija.
        if json_path is None:
            self._vo_static = list(obstacles) if obstacles is not None else []
            self._edges = edges_zone
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
            self._vo_static = [Obstacle2D.from_entry(o) for o in data["obstacles"]
                               if o["name"] != "col_manija"]
            self._vo_static += [Obstacle2D.from_entry(s) for s in data["sticks"]]
            if obstacles is not None:
                self._vo_static += list(obstacles)

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
        self._vo_dyn = []
        self._sync_obstacles()
        self._wi = 0

    def _sync_obstacles(self):
        """Refresca la lista efectiva y su array de AABB, que vortex_apf usa para
        descartar de golpe las que estan fuera de rep_range."""
        self._vo = self._vo_static + self._vo_dyn
        self._vo_bounds = (np.array([o.bounds() for o in self._vo], dtype=float)
                           if self._vo else None)

    def reset(self, robot_xy: np.ndarray):
        self._wi = 0

    def replan(self, waypoints: list, obstacles: list = None):
        self._wps = [np.array(w) for w in waypoints]
        self._wi = 0
        if obstacles is not None:
            self._vo_dyn = list(obstacles)      # NO pisa la geometria de la pista
            self._sync_obstacles()

    def _vortex_at(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        return vortex_apf(pos, target, self._vo, edges=self._edges,
                          att_gain=self._att_gain, att_range=self._att_range,
                          rep_gain=self._rep_gain, rep_range=self._rep_range,
                          rh=self._rh, swirl=self._swirl,
                          min_progress=self._min_progress,
                          obs_bounds=self._vo_bounds)

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