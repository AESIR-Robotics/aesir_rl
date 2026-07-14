"""
plot_path_vortex.py — Extiende plot_path.py mostrando el efecto del VORTEX APF.

plot_path.py solo dibuja los waypoints A* (la ruta global de referencia). Este
script ademas:

  1. Simula un robot puntual que sigue GlobalNavigator.step() — que en cada
     tick aplica vortex_apf() para esquivar obstaculos/sticks — desde el spawn
     hasta la meta, y dibuja la trayectoria REAL resultante (solida).
  2. Opcional (--field): dibuja el campo vectorial del vortex (quiver) tomando
     la meta como atractor global, para ver el "remolino" alrededor de los
     obstaculos.

Asi se compara la ruta A* ideal (punteada) contra el camino que de verdad
recorreria el navegador con la correccion vortex (solida).

Uso:
    cd rl_ws
    python3 tests/plot_path_vortex.py                 # ruta A* + trayectoria vortex
    python3 tests/plot_path_vortex.py --field         # + campo vectorial
    python3 tests/plot_path_vortex.py --save vortex.png
"""
import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_WS_ROOT = os.path.dirname(_THIS_DIR)
if _RL_WS_ROOT not in sys.path:
    sys.path.insert(0, _RL_WS_ROOT)

from global_navigator import (
    plan_route, box_corners_2d, GlobalNavigator, vortex_apf, REACH_DIST,
)
# TODOS los parametros compartidos (inicio/meta de la mision, escalas de
# comando, geometria de la zona segura) salen de base_training/config.py — la
# misma fuente que usan los dos backends de entrenamiento. Sin arrastrar
# ROS/mujoco/torch, asi este script sigue corriendo sin sourcear el workspace.
from base_training.config import (
    START_XY as _START_XY, GOAL_XY as _GOAL_XY, NAV_JSON as _WB_NAV, SPAWN_Z,
    V_MAX_MPS, W_MAX_RADPS, FINISH_DIST,
    ROBOT_RADIUS, GAP_BRIDGE_DISTANCE,
)

def _resolve_goal():
    if _GOAL_XY is not None:
        return tuple(_GOAL_XY)
    with open(_WB_NAV, "r") as _f:                 # None -> ultimo pallet del JSON
        return tuple(json.load(_f)["pallets"][-1]["center_xy"])

SPAWN_XYZ = (float(_START_XY[0]), float(_START_XY[1]), float(SPAWN_Z))   # compat: [:2] es el inicio
GOAL_XY   = _resolve_goal()

STYLE_OBSTACLE    = dict(fc="#2f3542", ec="#000000", alpha=0.90, zorder=2, lw=0.8)
STYLE_STICK       = dict(fc="#e17055", ec="#d63031", alpha=0.95, zorder=2, lw=1.2)
STYLE_REAL_PALLET = dict(fc="#ced6e0", ec="#a4b0be", alpha=0.50, zorder=1, lw=0.8)
STYLE_SAFE_PALLET = dict(fc="#00b894", ec="#00cec9", alpha=0.85, zorder=3, lw=0.8)

MAX_ITERS   = 20000   # tope de ticks para evitar bucle infinito si se atasca

# Controlador de respaldo del simulador puntual (mismo que test_path.py /
# test_nav_ros2.py): cinematica diferencial acotada en vez de "teletransportarse"
# hacia vortex_pt cada tick. Un punto sin inercia que salta directo al punto
# vortex amplifica cualquier fluctuacion de direccion en desplazamiento real --
# puede marcar "atascado" por una oscilacion de guia que un robot con velocidad
# angular acotada jamas notaria, porque no puede girar 180 grados en un tick.
# Esto hace que la trayectoria dibujada (y la deteccion de atasco) reflejen lo
# que el robot real -- o una politica PPO ya entrenada -- efectivamente haria.
# (V_MAX_MPS / W_MAX_RADPS / FINISH_DIST vienen de config; esto es solo del plot)
CONTROL_DT     = 0.05
SPIN_THRESHOLD = 0.50
K_SPEED        = 0.60
K_TURN         = 1.00

STUCK_WINDOW = 3000   # ticks sin avance neto que consideramos "atascado" (150s:
                       # tramos dificiles pueden tardar >2min en despejarse sin
                       # estar realmente atascados, ver max_stall_between_wps
                       # medido en la verificacion del fix de vortex_apf)
STUCK_MOVE   = 0.08   # avance neto minimo [m] en la ventana para no estar atascado


def simulate_vortex_path(nav: GlobalNavigator, start_xy, goal_xy):
    """Recorre el navegador (con vortex) con cinematica diferencial acotada,
    igual que el controlador de respaldo de test_path.py: gira hacia la guia
    (sin/cos del angulo) con velocidad angular limitada, y solo avanza cuando
    esta razonablemente encarado. Devuelve (trayectoria Nx2, status) donde
    status in {'goal','stuck','maxiter'}.

    Detecta atasco real (no artefacto del modelo de movimiento): si en
    STUCK_WINDOW ticks el robot no se desplazo mas de STUCK_MOVE en neto,
    corta -- asi la linea termina donde de verdad se traba."""
    pos  = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)
    yaw  = 0.0
    nav.reset(pos)
    path = [pos.copy()]

    for _ in range(MAX_ITERS):
        guidance = nav.step(pos, yaw)
        _dn, sin_t, cos_t = guidance["obs"]
        w_norm = float(np.clip(K_TURN * sin_t, -1.0, 1.0))
        v_norm = float(np.clip(K_SPEED * cos_t, 0.0, 1.0)) if cos_t >= SPIN_THRESHOLD else 0.0

        yaw = yaw + w_norm * W_MAX_RADPS * CONTROL_DT
        pos = pos + np.array([np.cos(yaw), np.sin(yaw)]) * (v_norm * V_MAX_MPS) * CONTROL_DT
        path.append(pos.copy())

        if float(np.hypot(*(pos - goal))) < FINISH_DIST:
            return np.array(path), "goal"
        if len(path) > STUCK_WINDOW:
            net = float(np.hypot(*(pos - path[-STUCK_WINDOW])))
            if net < STUCK_MOVE:
                return np.array(path), "stuck"

    return np.array(path), "maxiter"


def draw_map(ax, data):
    """Dibuja obstaculos, sticks, pallets y zona segura (identico a plot_path.py)."""
    for o in data["obstacles"]:
        corners = box_corners_2d(np.array(o["center_xy"]), np.array(o["size"]), np.array(o["rot_mat"]))
        ax.add_patch(MPolygon(corners, closed=True, **STYLE_OBSTACLE))

    for s in data["sticks"]:
        corners = box_corners_2d(np.array(s["center_xy"]), np.array(s["size"]), np.array(s["rot_mat"]))
        ax.add_patch(MPolygon(corners, closed=True, **STYLE_STICK))

    pallets_poly = []
    for p in data["pallets"]:
        corners = box_corners_2d(np.array(p["center_xy"]), np.array(p["size"]), np.array(p["rot_mat"]))
        ax.add_patch(MPolygon(corners, closed=True, **STYLE_REAL_PALLET))
        pallets_poly.append(ShapelyPolygon(corners))
        name = p["name"].split()[-1]
        ax.annotate(f"P{name}", xy=p["center_xy"], fontsize=7, ha="center", va="center",
                    color="#ffffff", alpha=0.3, zorder=5)

    merged = unary_union(pallets_poly)
    safe_zone = merged.buffer(GAP_BRIDGE_DISTANCE).buffer(-GAP_BRIDGE_DISTANCE).buffer(-ROBOT_RADIUS)
    geoms = [safe_zone] if safe_zone.geom_type == "Polygon" else list(safe_zone.geoms)
    labeled = False
    for poly in geoms:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        patch = MPolygon(np.column_stack((x, y)), closed=True, **STYLE_SAFE_PALLET)
        if not labeled:
            patch.set_label("Safe Drivable Zone")
            labeled = True
        ax.add_patch(patch)


# Escala de color compartida por todo el campo (magnitud del vector, recortada:
# cerca de un obstaculo la repulsion diverge ~1/do^2 y saturaria el color).
_FIELD_CMAP = "plasma"
_FIELD_VMIN, _FIELD_VMAX = 0.0, 1.5
_FIELD_NORM = Normalize(vmin=_FIELD_VMIN, vmax=_FIELD_VMAX)


def _field_vec(p, target, obstacles, edges, att_gain, att_range, rep_gain, rep_range,
              rh, swirl, min_progress, repulsion_only):
    """Vector de guia en el punto p. repulsion_only=True resta la atraccion
    para dejar SOLO la repulsion (obstaculos + bordes de pallets); False
    devuelve el campo completo (atraccion + repulsion) que ve el robot.

    IMPORTANTE: recibe TODOS los parametros del campo (no solo rep_gain/
    rep_range/rh) para que el quiver dibujado sea matematicamente el MISMO
    campo que evalua GlobalNavigator.step() -> vortex_apf() durante la
    simulacion. Antes, swirl/att_gain/att_range/min_progress no se pasaban
    y vortex_apf() caia en sus propios defaults de modulo -- coincidian
    por casualidad con los defaults de GlobalNavigator, pero se
    desincronizaban en silencio si 'nav' se construia con otros valores."""
    full = vortex_apf(p, target, obstacles, edges=edges,
                      att_gain=att_gain, att_range=att_range,
                      rep_gain=rep_gain, rep_range=rep_range,
                      rh=rh, swirl=swirl, min_progress=min_progress) - p
    if not repulsion_only:
        return full
    tgt = np.asarray(target, dtype=float)
    d = tgt - p
    dt = float(np.hypot(*d))
    # atraccion con los MISMOS att_gain/att_range que usa el campo completo
    # (antes estaba hardcodeada a att_gain=1, att_range=1 sin importar el
    # valor real pasado/usado por nav).
    att = (d / dt) * att_gain * min(dt / att_range, 1.0) if dt > 1e-9 else np.zeros(2)
    return full - att


def _quiver_field(ax, X, Y, target, obstacles, edges, att_gain, att_range, rep_gain,
                  rep_range, rh, swirl, min_progress, repulsion_only):
    """Dibuja un quiver coloreado por magnitud sobre la grilla X,Y."""
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            p = np.array([X[r, c], Y[r, c]])
            vec = _field_vec(p, target, obstacles, edges, att_gain, att_range,
                             rep_gain, rep_range, rh, swirl, min_progress, repulsion_only)
            nrm = float(np.hypot(*vec))
            M[r, c] = nrm
            if nrm > 1e-6:
                U[r, c], V[r, c] = vec / nrm
    return ax.quiver(X, Y, U, V, np.clip(M, _FIELD_VMIN, _FIELD_VMAX), cmap=_FIELD_CMAP,
                     norm=_FIELD_NORM, zorder=1, alpha=0.85,
                     scale=45, width=0.0018, headwidth=3)


def draw_vortex_field(ax, nav: GlobalNavigator, goal_xy, xlim, ylim,
                      att_gain=1.0, att_range=1.0, rep_gain=0.35, rep_range=0.35,
                      rh=0.30, swirl=1.0, min_progress=0.1, repulsion_only=False, n=32):
    """Campo vortex coloreado por magnitud tomando la meta FINAL como atractor
    unico global. Vista de conjunto rapida (--field-mode global). El campo REAL
    por tramo lo da draw_vortex_field_route."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    return _quiver_field(ax, X, Y, np.array(goal_xy, float),
                         nav._vo, nav._edges, att_gain, att_range, rep_gain,
                         rep_range, rh, swirl, min_progress, repulsion_only)


def draw_vortex_field_route(ax, nav: GlobalNavigator, waypoints, density=10.0,
                            local_margin=0.7, min_grid=5,
                            att_gain=1.0, att_range=1.0, rep_gain=0.35, rep_range=0.35,
                            rh=0.30, swirl=1.0, min_progress=0.1, repulsion_only=False):
    """Campo vortex REAL a lo largo de TODA la ruta, coloreado por magnitud:
    para cada tramo wps[i] -> wps[i+1] el atractor es wps[i+1] -- exactamente
    el target que usa GlobalNavigator.step() mientras self._wi == i+1 -- en vez
    de la meta final fija. Mosaico de quivers locales, uno por tramo, para que
    el remolino de repulsion (obstaculos + bordes de pallets) se vea tal como
    lo experimenta el robot en ESE tramo.

    density = puntos de grilla por metro (mas alto = mas denso/lento)."""
    wps_arr = np.array(waypoints, dtype=float)
    q = None
    for i in range(len(wps_arr) - 1):
        p0, p1 = wps_arr[i], wps_arr[i + 1]
        seg_xlim = (min(p0[0], p1[0]) - local_margin, max(p0[0], p1[0]) + local_margin)
        seg_ylim = (min(p0[1], p1[1]) - local_margin, max(p0[1], p1[1]) + local_margin)
        nx = max(min_grid, int(round((seg_xlim[1] - seg_xlim[0]) * density)))
        ny = max(min_grid, int(round((seg_ylim[1] - seg_ylim[0]) * density)))
        xs = np.linspace(*seg_xlim, nx)
        ys = np.linspace(*seg_ylim, ny)
        X, Y = np.meshgrid(xs, ys)
        q = _quiver_field(ax, X, Y, p1, nav._vo, nav._edges, att_gain, att_range,
                          rep_gain, rep_range, rh, swirl, min_progress, repulsion_only)
    return q


def main():
    parser = argparse.ArgumentParser(description="Vortex APF path visualizer")
    parser.add_argument("--json", default=_WB_NAV,
                        help="mapa de obstaculos (default: el de config, no depende del CWD)")
    parser.add_argument("--save", default=None)
    parser.add_argument("--field", action="store_true", help="dibujar campo vectorial vortex")
    parser.add_argument("--field-mode", choices=["route", "global"], default="route",
                        help="route = atractor real por tramo (fiel a GlobalNavigator); "
                             "global = atractor unico en la meta final (aproximado, mas rapido)")
    parser.add_argument("--density", type=float, default=10.0,
                        help="puntos de grilla por metro en modo --field-mode=route")
    parser.add_argument("--repulsion", action="store_true",
                        help="dibujar SOLO la repulsion (resta la atraccion) — muestra"
                             " el remolino puro de obstaculos + bordes de pallets")
    parser.add_argument("--rep-gain", type=float, default=None,
                        help="grado de repulsion (default: el mismo que usa 'nav', "
                             "que es el que realmente siguio el robot)")
    parser.add_argument("--rep-range", type=float, default=None,
                        help="distancia de repulsion [m] (default: el de 'nav')")
    parser.add_argument("--rh", type=float, default=None,
                        help="media anchura del robot [m] (default: el de 'nav')")
    parser.add_argument("--att-gain", type=float, default=None,
                        help="grado de atraccion (default: el de 'nav')")
    parser.add_argument("--att-range", type=float, default=None,
                        help="distancia de atraccion [m] (default: el de 'nav')")
    parser.add_argument("--swirl", type=float, default=None,
                        help="peso de la componente tangencial (default: el de 'nav')")
    parser.add_argument("--min-progress", type=float, default=None,
                        help="fraccion minima de avance garantizado (default: el de 'nav')")
    parser.add_argument("--start", type=float, nargs=2, default=list(SPAWN_XYZ[:2]))
    parser.add_argument("--goal", type=float, nargs=2, default=list(GOAL_XY))
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found. Generate the JSON map first.")
        return

    with open(args.json, "r") as f:
        data = json.load(f)

    start_xy = tuple(args.start)
    goal_xy  = tuple(args.goal)

    print("Planeando ruta A* ...")
    waypoints = plan_route(args.json, start_xy, goal_xy)
    wps_arr = np.array(waypoints)

    nav = GlobalNavigator(args.json, waypoints=waypoints)
    print("Simulando seguimiento con correccion vortex ...")
    vortex_path, status = simulate_vortex_path(nav, start_xy, goal_xy)
    _status_txt = {"goal": "llego a meta",
                   "stuck": "ATASCADO (minimo local del APF)",
                   "maxiter": "tope de iteraciones"}[status]
    print(f"  trayectoria vortex: {len(vortex_path)} puntos -> {_status_txt}")
    if status != "goal":
        print(f"  se traba en ({vortex_path[-1][0]:+.2f}, {vortex_path[-1][1]:+.2f})")

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1e222b")   # fondo oscuro tambien fuera del axes
    ax.set_aspect("equal")               # (si no, la colorbar sale ilegible)
    ax.set_facecolor("#1e222b")
    ax.grid(True, linestyle=":", alpha=0.2, color="#ffffff", zorder=0)
    ax.set_xlabel("X [m]", fontsize=11, color="#ffffff")
    ax.set_ylabel("Y [m]", fontsize=11, color="#ffffff")
    ax.set_title("Vortex APF Path — A* reference vs. real vortex-followed trajectory",
                 fontsize=13, fontweight="bold", color="#ffffff", pad=14)
    ax.tick_params(colors="#ffffff")

    draw_map(ax, data)

    all_pts = wps_arr.tolist() + vortex_path.tolist() + [list(start_xy), list(goal_xy)]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xlim = (min(xs) - 1.5, max(xs) + 1.5)
    ylim = (min(ys) - 1.5, max(ys) + 1.5)

    if args.field:
        # Los parametros del campo se toman de 'nav' por default -- el MISMO
        # objeto que genero vortex_path -- y solo se pisan si el usuario los
        # especifico explicitamente por CLI. Asi el quiver siempre es fiel a
        # la trayectoria dibujada, salvo que uno pida explorar "que pasaria
        # si..." con otros valores.
        field_params = dict(
            att_gain=args.att_gain if args.att_gain is not None else nav._att_gain,
            att_range=args.att_range if args.att_range is not None else nav._att_range,
            rep_gain=args.rep_gain if args.rep_gain is not None else nav._rep_gain,
            rep_range=args.rep_range if args.rep_range is not None else nav._rep_range,
            rh=args.rh if args.rh is not None else nav._rh,
            swirl=args.swirl if args.swirl is not None else nav._swirl,
            min_progress=args.min_progress if args.min_progress is not None else nav._min_progress,
        )
        kind = "repulsion pura" if args.repulsion else "campo completo (atrac+repul)"
        print(f"Dibujando campo vortex (modo={args.field_mode}, {kind}) con los "
              f"parametros REALES de nav: {field_params}")
        if args.field_mode == "route":
            draw_vortex_field_route(ax, nav, waypoints, density=args.density,
                                    repulsion_only=args.repulsion, **field_params)
        else:
            draw_vortex_field(ax, nav, goal_xy, xlim, ylim,
                              repulsion_only=args.repulsion, **field_params)
        # Colorbar de la magnitud del campo (una sola, escala compartida).
        sm = ScalarMappable(norm=_FIELD_NORM, cmap=_FIELD_CMAP)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("|vector de guia|  (claro = empuje fuerte cerca de obstaculo)",
                       color="#ffffff", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#ffffff")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    # Ruta A* de referencia (punteada)
    ax.plot(wps_arr[:, 0], wps_arr[:, 1], color="#ffffff", linewidth=1.4,
            linestyle="--", alpha=0.45, zorder=4, label="A* reference path")
    for idx, (x, y) in enumerate(waypoints):
        ax.plot(x, y, marker="o", color="#ff9f43", markersize=9, zorder=6, linestyle="None")
        ax.annotate(str(idx), xy=(x, y), fontsize=6, ha="center", va="center",
                    color="#000000", fontweight="bold", zorder=7)

    # Trayectoria REAL con vortex (solida)
    ax.plot(vortex_path[:, 0], vortex_path[:, 1], color="#00e5ff", linewidth=2.4,
            alpha=0.95, zorder=5, label="Vortex-followed trajectory")

    ax.plot(*start_xy, marker="o", color="#e84393", markersize=14, zorder=8,
            linestyle="None", label="Spawn")
    ax.plot(*goal_xy, marker="*", color="#ff4757", markersize=20, zorder=8,
            linestyle="None", label="Goal")

    if status != "goal":
        ax.plot(vortex_path[-1, 0], vortex_path[-1, 1], marker="X", color="#ffeaa7",
                markersize=16, mew=2, mec="#d63031", zorder=9,
                linestyle="None", label="Vortex stuck (local min)")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              facecolor="#1e222b", edgecolor="#a4b0be", labelcolor="#ffffff")

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figura guardada en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()