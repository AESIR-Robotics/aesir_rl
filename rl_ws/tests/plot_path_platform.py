"""
plot_path_platform.py — Igual que plot_path_vortex.py pero para el mapa NUEVO
(plataform.xml, plataforma plana SIN pallets fisicos). obstacles.json ya no
describe el mundo real (los pallets/sticks que tenia dejaron de existir en el
XML), asi que aqui no se usa: en su lugar se reproduce EXACTAMENTE lo que hace
BaseMujocoEnv.reset() --

  1. Sortea spawn/meta random en [-SPAWN_XY_RANGE, SPAWN_XY_RANGE] /
     [-GOAL_XY_RANGE, GOAL_XY_RANGE] (config.py).
  2-3. Arma la ruta y el obstaculo virtual UNICO EN UN SOLO PASO
     (plan_platform_route_with_obstacle): elige primero donde va el
     obstaculo y arma la ruta directa (recta remuestreada, sin A* porque la
     plataforma no tiene obstaculos fisicos) dejando un hueco a su alrededor
     -- el tamano del obstaculo sale de ese hueco, nunca encierra un waypoint.
     No existe en el XML, solo en la capa de planificacion/vortex.
  4. Simula el seguimiento vortex (GlobalNavigator.step) desde el spawn hasta
     la meta y dibuja la trayectoria REAL resultante.

Sirve tanto para INSPECCIONAR visualmente un episodio (modo plot, default)
como para un TEST rapido sin display: --trials N corre N episodios random y
valida invariantes basicas (ruta bien formada, obstaculo realmente sobre la
ruta, spawn/goal dentro de la plataforma, el vortex efectivamente llega).

Uso:
    cd rl_ws
    python3 tests/plot_path_platform.py                # 1 episodio random, plot
    python3 tests/plot_path_platform.py --seed 3        # reproducible
    python3 tests/plot_path_platform.py --field         # + campo vectorial vortex
    python3 tests/plot_path_platform.py --save out.png
    python3 tests/plot_path_platform.py --trials 50      # test sin grafica
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon, Rectangle
from matplotlib.cm import ScalarMappable

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_WS_ROOT = os.path.dirname(_THIS_DIR)
if _RL_WS_ROOT not in sys.path:
    sys.path.insert(0, _RL_WS_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from global_navigator import (
    build_platform_zone, plan_platform_route, plan_platform_route_with_obstacle,
    GlobalNavigator,
)
from base_training.config import (
    SPAWN_XY_RANGE, GOAL_XY_RANGE, PLATFORM_HALF_EXTENT,
    USE_VIRTUAL_OBSTACLE,
)
# Reusa la simulacion vortex + el dibujado del campo vectorial global tal cual
# -- son genericos (solo dependen de nav._vo / nav._edges), no de pallets.
# draw_vortex_field_route NO se reusa: dibuja una caja de quiver por tramo con
# local_margin=0.7m fijo, que se SOLAPA entre tramos consecutivos cuando los
# waypoints estan a ~0.5m (rutas rectas de la plataforma) -- el solape mezcla
# vectores de DOS targets distintos en la misma zona y parece que la repulsion
# "va contra el camino" cuando en realidad es el target del tramo ANTERIOR
# bleeding en la caja del tramo siguiente. draw_vortex_field_route_exact (mas
# abajo) evita esto asignando a cada punto el target del tramo REALMENTE mas
# cercano (proyeccion sobre la polilinea), sin cajas fijas que se solapen.
from plot_path_vortex import (
    simulate_vortex_path, draw_vortex_field, _field_vec,
    _FIELD_NORM, _FIELD_CMAP, _FIELD_VMIN, _FIELD_VMAX,
)

STYLE_PLATFORM    = dict(fill=False, ec="#ffa502", lw=2.0, zorder=1)
STYLE_SAFE_ZONE   = dict(fc="#00b894", ec="#00cec9", alpha=0.12, zorder=1, lw=0.8)
STYLE_VIRTUAL_OBS = dict(fc="#e17055", ec="#d63031", alpha=0.95, zorder=2, lw=1.4)


def sample_episode(seed=None):
    """Sortea spawn/meta random (igual que BaseMujocoEnv.reset()) y arma la
    ruta directa + obstaculo virtual. Devuelve (start_xy, goal_xy, waypoints,
    virtual_obs_or_None, platform_zone)."""
    if seed is not None:
        np.random.seed(seed)
    start_xy = tuple(np.random.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE, 2))
    goal_xy  = tuple(np.random.uniform(-GOAL_XY_RANGE, GOAL_XY_RANGE, 2))
    if USE_VIRTUAL_OBSTACLE:
        waypoints, vobs = plan_platform_route_with_obstacle(start_xy, goal_xy)
    else:
        waypoints, vobs = plan_platform_route(start_xy, goal_xy), None
    zone = build_platform_zone()
    return start_xy, goal_xy, waypoints, vobs, zone


def draw_platform(ax, zone, half_extent=PLATFORM_HALF_EXTENT):
    """Dibuja el limite fisico de la plataforma (cuadrado 2*half_extent) y la
    zona segura erosionada por ROBOT_RADIUS (el area donde el robot puede
    pisar sin acercarse tanto al borde como para caerse)."""
    he = half_extent
    ax.add_patch(Rectangle((-he, -he), 2 * he, 2 * he, **STYLE_PLATFORM))
    x, y = zone.exterior.xy
    patch = MPolygon(np.column_stack((x, y)), closed=True, **STYLE_SAFE_ZONE)
    patch.set_label("Safe zone (erosionada)")
    ax.add_patch(patch)


def draw_virtual_obstacle(ax, obs):
    if obs is None:
        return
    corners = np.array([
        [obs.x - obs.hx, obs.y - obs.hy],
        [obs.x + obs.hx, obs.y - obs.hy],
        [obs.x + obs.hx, obs.y + obs.hy],
        [obs.x - obs.hx, obs.y + obs.hy],
    ])
    patch = MPolygon(corners, closed=True, **STYLE_VIRTUAL_OBS)
    patch.set_label("Virtual obstacle")
    ax.add_patch(patch)


def _nearest_segment_idx(p, wps_arr):
    """Indice i del tramo wps_arr[i]->wps_arr[i+1] cuya distancia PERPENDICULAR
    (recortada a los extremos) a p es minima -- el tramo que de verdad
    'controla' ese punto, a diferencia de una caja de margen fijo por tramo."""
    best_i, best_d = 0, np.inf
    for i in range(len(wps_arr) - 1):
        a, b = wps_arr[i], wps_arr[i + 1]
        ab = b - a
        l2 = float(np.dot(ab, ab))
        t = 0.0 if l2 < 1e-12 else float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
        closest = a + t * ab
        d = float(np.linalg.norm(p - closest))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def draw_vortex_field_route_exact(ax, nav, waypoints, density=10.0, margin=0.8,
                                  att_gain=1.0, att_range=1.0, rep_gain=0.35, rep_range=0.35,
                                  rh=0.30, swirl=1.0, min_progress=0.1, repulsion_only=False):
    """Campo vortex REAL a lo largo de la ruta, SIN el artefacto de solape de
    draw_vortex_field_route (cajas de margen fijo por tramo que se pisan
    cuando los waypoints estan muy juntos). Una sola grilla sobre toda la
    ruta; cada punto usa el target del tramo mas cercano por proyeccion
    (_nearest_segment_idx), asi que nunca se mezclan dos targets en la misma
    zona."""
    wps_arr = np.array(waypoints, dtype=float)
    x0, x1 = wps_arr[:, 0].min() - margin, wps_arr[:, 0].max() + margin
    y0, y1 = wps_arr[:, 1].min() - margin, wps_arr[:, 1].max() + margin
    nx = max(10, int(round((x1 - x0) * density)))
    ny = max(10, int(round((y1 - y0) * density)))
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X); V = np.zeros_like(Y); M = np.zeros_like(X)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            p = np.array([X[r, c], Y[r, c]])
            i = _nearest_segment_idx(p, wps_arr)
            target = wps_arr[i + 1]
            vec = _field_vec(p, target, nav._vo, nav._edges, att_gain, att_range,
                             rep_gain, rep_range, rh, swirl, min_progress, repulsion_only)
            nrm = float(np.hypot(*vec))
            M[r, c] = nrm
            if nrm > 1e-6:
                U[r, c], V[r, c] = vec / nrm
    return ax.quiver(X, Y, U, V, np.clip(M, _FIELD_VMIN, _FIELD_VMAX), cmap=_FIELD_CMAP,
                     norm=_FIELD_NORM, zorder=1, alpha=0.85, scale=45, width=0.0018, headwidth=3)


def _point_segment_dist(p, a, b):
    """Distancia perpendicular de p a la RECTA que contiene el segmento a-b
    (no recorta a los extremos)."""
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        return float(np.linalg.norm(p - a))
    ap = p - a
    cross_z = ab[0] * ap[1] - ab[1] * ap[0]   # componente z del cross 2D (ab x ap)
    return float(abs(cross_z) / n)


def run_trials(n, verbose=True):
    """Modo TEST sin grafica: corre n episodios random y valida invariantes
    basicas de la generacion de ruta + obstaculo virtual. Devuelve
    (n_ok, n_fail) y hace print de un resumen."""
    n_ok = 0
    status_counts = {"goal": 0, "stuck": 0, "maxiter": 0}
    fails = []
    he = PLATFORM_HALF_EXTENT

    for i in range(n):
        start_xy, goal_xy, waypoints, vobs, zone = sample_episode()
        ok, reasons = True, []

        if not np.allclose(waypoints[0], start_xy, atol=1e-6):
            ok = False; reasons.append("waypoints[0] != start_xy")
        if not np.allclose(waypoints[-1], goal_xy, atol=1e-6):
            ok = False; reasons.append("waypoints[-1] != goal_xy")

        if not (-he <= start_xy[0] <= he and -he <= start_xy[1] <= he):
            ok = False; reasons.append("start fuera de la plataforma")
        if not (-he <= goal_xy[0] <= he and -he <= goal_xy[1] <= he):
            ok = False; reasons.append("goal fuera de la plataforma")

        if vobs is not None:
            p = np.array([vobs.x, vobs.y])
            wps_arr = np.array(waypoints)
            # OJO: toda la ruta es colineal (recta start->goal resampleada),
            # asi que la distancia a la RECTA de cualquier tramo da lo mismo
            # -- para identificar el tramo HOST real (el que se agrando para
            # dejarle hueco al obstaculo) hace falta la distancia al SEGMENTO
            # (recortada a sus extremos), no a la recta infinita.
            best = _point_segment_dist(p, wps_arr[0], wps_arr[1])
            host = _nearest_segment_idx(p, wps_arr)
            # Se desplaza lateral (perpendicular) del tramo elegido, proporcional
            # a su propio half_size -- ver offset_frac_range de
            # plan_platform_route_with_obstacle en global_navigator.py.
            if not (0.10 * vobs.hx <= best <= 0.70 * vobs.hx):
                ok = False; reasons.append(f"offset del obstaculo fuera de lo esperado (dist={best:.4f}, half_size={vobs.hx:.3f})")

            # Invariante de SEGURIDAD real: ningun waypoint debe caer DENTRO
            # de la caja del obstaculo -- eso es lo que causaba la orbita
            # infinita del vortex (atractor encerrado por la repulsion).
            dx = np.abs(wps_arr[:, 0] - vobs.x)
            dy = np.abs(wps_arr[:, 1] - vobs.y)
            if not np.all((dx > vobs.hx) | (dy > vobs.hy)):
                ok = False; reasons.append("obstaculo virtual ENCIERRA un waypoint real (riesgo de orbita)")

            # Invariante de TAMANIO: el obstaculo nunca deberia ocupar mas de
            # la mitad del tramo que lo bordea (host segment) menos el
            # clearance minimo esperado a cada lado -- confirma que el tamano
            # sale de la distancia real entre esos dos waypoints, no al reves.
            host_len = float(np.linalg.norm(wps_arr[host + 1] - wps_arr[host]))
            if 2 * vobs.hx > host_len + 1e-6:
                ok = False; reasons.append(
                    f"obstaculo mas grande que su propio tramo (2*hx={2*vobs.hx:.3f} > host_len={host_len:.3f})")

        nav = GlobalNavigator(None, waypoints=waypoints,
                              obstacles=[vobs] if vobs is not None else [],
                              edges_zone=zone)
        _path, status = simulate_vortex_path(nav, start_xy, goal_xy)
        status_counts[status] += 1
        if status != "goal":
            reasons.append(f"vortex no llego a la meta ({status})")

        if ok:
            n_ok += 1
        else:
            fails.append((i, start_xy, goal_xy, reasons))

    if verbose:
        print(f"\n{n_ok}/{n} episodios con ruta+obstaculo bien formados")
        print(f"Vortex: goal={status_counts['goal']}  stuck={status_counts['stuck']}  "
              f"maxiter={status_counts['maxiter']}  (de {n})")
        for i, s, g, reasons in fails[:10]:
            print(f"  [FAIL trial {i}] start=({s[0]:.2f},{s[1]:.2f}) "
                  f"goal=({g[0]:.2f},{g[1]:.2f}): {'; '.join(reasons)}")
        if len(fails) > 10:
            print(f"  ... y {len(fails) - 10} fallos mas")
    return n_ok, n - n_ok


def main():
    parser = argparse.ArgumentParser(
        description="Test/visualizador de ruta + obstaculo virtual sobre la plataforma plana")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--field", action="store_true", help="dibujar campo vectorial vortex")
    parser.add_argument("--field-mode", choices=["route", "global"], default="route")
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--repulsion", action="store_true",
                        help="dibujar SOLO la repulsion (resta la atraccion)")
    parser.add_argument("--start", type=float, nargs=2, default=None)
    parser.add_argument("--goal", type=float, nargs=2, default=None)
    parser.add_argument("--trials", type=int, default=0,
                        help="si > 0, corre N episodios random SIN grafica (modo test) y sale")
    args = parser.parse_args()

    if args.trials > 0:
        _n_ok, n_fail = run_trials(args.trials)
        sys.exit(1 if n_fail > 0 else 0)

    if args.seed is not None:
        np.random.seed(args.seed)

    start_xy = tuple(args.start) if args.start else tuple(
        np.random.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE, 2))
    goal_xy = tuple(args.goal) if args.goal else tuple(
        np.random.uniform(-GOAL_XY_RANGE, GOAL_XY_RANGE, 2))

    print(f"Spawn random: ({start_xy[0]:.2f}, {start_xy[1]:.2f})   "
          f"Meta random: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    if USE_VIRTUAL_OBSTACLE:
        waypoints, vobs = plan_platform_route_with_obstacle(start_xy, goal_xy)
    else:
        waypoints, vobs = plan_platform_route(start_xy, goal_xy), None
    zone = build_platform_zone()
    if vobs is not None:
        print(f"Obstaculo virtual en ({vobs.x:.2f}, {vobs.y:.2f})  half_size={vobs.hx:.2f}")

    nav = GlobalNavigator(None, waypoints=waypoints,
                          obstacles=[vobs] if vobs is not None else [],
                          edges_zone=zone)
    print("Simulando seguimiento con correccion vortex ...")
    vortex_path, status = simulate_vortex_path(nav, start_xy, goal_xy)
    _status_txt = {"goal": "llego a meta", "stuck": "ATASCADO (minimo local del APF)",
                   "maxiter": "tope de iteraciones"}[status]
    print(f"  trayectoria vortex: {len(vortex_path)} puntos -> {_status_txt}")

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("#1e222b")
    ax.set_aspect("equal")
    ax.set_facecolor("#1e222b")
    ax.grid(True, linestyle=":", alpha=0.2, color="#ffffff", zorder=0)
    ax.set_xlabel("X [m]", fontsize=11, color="#ffffff")
    ax.set_ylabel("Y [m]", fontsize=11, color="#ffffff")
    ax.set_title("Plataforma plana — ruta directa + obstaculo virtual + trayectoria vortex",
                 fontsize=13, fontweight="bold", color="#ffffff", pad=14)
    ax.tick_params(colors="#ffffff")

    draw_platform(ax, zone)
    draw_virtual_obstacle(ax, vobs)

    wps_arr = np.array(waypoints)
    all_pts = wps_arr.tolist() + vortex_path.tolist() + [list(start_xy), list(goal_xy)]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xlim = (min(xs) - 1.0, max(xs) + 1.0)
    ylim = (min(ys) - 1.0, max(ys) + 1.0)

    if args.field:
        field_params = dict(att_gain=nav._att_gain, att_range=nav._att_range,
                            rep_gain=nav._rep_gain, rep_range=nav._rep_range,
                            rh=nav._rh, swirl=nav._swirl, min_progress=nav._min_progress)
        kind = "repulsion pura" if args.repulsion else "campo completo (atrac+repul)"
        print(f"Dibujando campo vortex (modo={args.field_mode}, {kind})")
        if args.field_mode == "route":
            draw_vortex_field_route_exact(ax, nav, waypoints, density=args.density,
                                          repulsion_only=args.repulsion, **field_params)
        else:
            draw_vortex_field(ax, nav, goal_xy, xlim, ylim,
                              repulsion_only=args.repulsion, **field_params)
        sm = ScalarMappable(norm=_FIELD_NORM, cmap=_FIELD_CMAP)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("|vector de guia|  (claro = empuje fuerte cerca del obstaculo)",
                       color="#ffffff", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#ffffff")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    ax.plot(wps_arr[:, 0], wps_arr[:, 1], color="#ffffff", linewidth=1.4,
            linestyle="--", alpha=0.45, zorder=4, label="Ruta directa (referencia)")
    for idx, (x, y) in enumerate(waypoints):
        ax.plot(x, y, marker="o", color="#ff9f43", markersize=8, zorder=6, linestyle="None")
        ax.annotate(str(idx), xy=(x, y), fontsize=6, ha="center", va="center",
                    color="#000000", fontweight="bold", zorder=7)

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
