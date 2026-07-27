"""
test_base.py — Corre/visualiza una politica base entrenada (.pt) sobre el MISMO
pipeline ROS2 que el entrenamiento: BaseRosEnv <-> mujoco_sim_rosbridge.py. El
viewer de MuJoCo lo abre el bridge, asi que ves el robot ejecutando la politica.

REQUIERE, en OTRA terminal, el bridge corriendo:
    cd rl_ws && MUJOCO_GL=glfw python3 base_training/mujoco_sim_rosbridge.py

Uso:
    cd rl_ws
    python3 tests/test_base.py                                   # base_best.pt, deterministico
    python3 tests/test_base.py --checkpoint checkpoints_base/base_iter00100.pt
    python3 tests/test_base.py --episodes 5 --stochastic
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

# Raiz del proyecto (aesir_rl) al path para resolver `rl_ws.*` (tests/ -> rl_ws -> aesir_rl)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RLWS = os.path.dirname(_HERE)                       # rl_ws/
_ROOT = os.path.dirname(_RLWS)                        # aesir_rl/
for _p in (_ROOT, _HERE):                            # _HERE: para importar plot_path_vortex
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_ws.base_training.base_ros_env import BaseRosEnv, NAV_JSON      # noqa: E402
import rl_ws.base_training.config as C                                 # noqa: E402
from rl_ws.base_training.ppo_cnn_extractor import CNNActorCritic       # noqa: E402

DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints_base", "fast_best.pt")


def load_policy(path: str, obs_dim: int, act_dim: int, device) -> CNNActorCritic:
    ckpt = torch.load(path, map_location=device)
    saved = ckpt["policy"]

    # Chequeo de compatibilidad: el .pt debe tener el mismo state_dim que el env
    # actual (si cambio N_LOOKAHEAD o HEATMAP_PIXELS, el .pt no sirve aqui).
    # act_dim ya lo valida el constructor de la politica (accion hibrida = 7
    # exacto, ver CNNActorCritic), asi que no hace falta sondearlo del .pt.
    ckpt_state_dim = int(saved["state_encoder.0.weight"].shape[1])
    state_dim = obs_dim - C.HEATMAP_PIXELS ** 2
    if ckpt_state_dim != state_dim:
        raise SystemExit(
            f"El checkpoint no coincide con el env: checkpoint(state={ckpt_state_dim}) "
            f"vs env(state={state_dim}). "
            f"Usa un .pt entrenado con esta misma configuracion.")

    policy = CNNActorCritic(obs_dim, act_dim, map_pixels=C.HEATMAP_PIXELS).to(device)
    policy.load_state_dict(saved)
    policy.eval()
    avg = ckpt.get("avg_ep_r", float("nan"))
    print(f"Modelo cargado: {path}  (iter={ckpt.get('iter', '?')}, avg_ep_r={avg:.2f})")
    return policy


@torch.no_grad()
def pick_action(policy: CNNActorCritic, obs: np.ndarray, device, stochastic: bool) -> np.ndarray:
    if stochastic:
        action, _raw, _logp, _val = policy.act(obs, device)   # muestrea de la distribucion
        return action
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    p, _val = policy(obs_t)                              # moda/media por bloque = accion deterministica
    vw = p["mu_vw"].clamp(-1.0, 1.0)                                        # (1,2)
    a, b = p["alpha"], p["beta"]
    flip = (a - 1.0) / (a + b - 2.0).clamp(min=1e-6)      # moda de la Beta (alpha,beta>=1)
    gate = (p["gate_logit"] > 0.0).float().unsqueeze(-1)  # Bernoulli mas probable
    return torch.cat([vw, flip, gate], dim=-1).squeeze(0).cpu().numpy()

_TERM_LABELS = [
    ("progress",          "Progreso (acercarse al wp)"),
    ("direction",         "Direccion (encarar guia)"),
    ("velocity",          "Velocidad (igualar v objetivo)"),
    ("wp_bonus",          "Bonus por waypoint"),
    ("time",              "Castigo de tiempo"),
    ("backward",          "Castigo por retroceder"),
    ("stuck",             "Castigo por atascarse"),
    ("energy",            "Castigo de energia"),
    ("flipper_jerk",      "Castigo jerk de flippers"),
    ("flipper_terrain",   "Bonus flipper cerca de trepable"),
    ("flipper_collision", "Castigo auto-colision flippers"),
    ("accel",             "Castigo aceleraciones fuertes"),
    ("fall",              "Castigo por caida/piso (terminal)"),
    ("obstacle",          "Castigo por choque obstaculo (terminal)"),
]


def print_reward_breakdown(title: str, sums: dict, steps: int):
    """Imprime cuanto aporto cada componente al reward: total, promedio/paso y
    % del reward positivo/negativo. La suma de todos == reward total."""
    total = sum(sums.values())
    pos = sum(v for v in sums.values() if v > 0) or 1e-9
    neg = sum(v for v in sums.values() if v < 0) or -1e-9
    print(f"\n  ── {title} ({steps} pasos) ──")
    print(f"    {'componente':<34}{'total':>11}{'/paso':>11}{'% +/-':>9}")
    for key, label in _TERM_LABELS:
        v = sums.get(key, 0.0)
        if abs(v) < 1e-9:
            continue
        share = 100.0 * v / (pos if v > 0 else -neg)   # % del lado (recompensa/castigo)
        per = v / steps if steps else 0.0
        print(f"    {label:<34}{v:>11.2f}{per:>11.4f}{share:>8.0f}%")
    print(f"    {'-'*34}{'-'*11}{'-'*11}{'-'*9}")
    print(f"    {'REWARD TOTAL':<34}{total:>11.2f}{total/max(steps,1):>11.4f}")
    print(f"    {'(suma positivos / negativos)':<34}{pos:>11.2f} / {neg:.2f}")


class LiveTrajectoryPlot:
    """Ventana matplotlib que se actualiza EN VIVO mientras corre el episodio.
    Dibuja el fondo del MUNDO ACTIVO una sola vez y, por episodio, la escena
    (ruta + meta + obstaculo virtual) que puede cambiar cada reset:

      - Plataforma (env.use_platform=True, mapa NUEVO plataform.xml): borde de
        la plataforma + zona segura erosionada (env.platform_zone) + ruta directa
        + obstaculo virtual. Igual look que tests/plot_path_platform.py.
      - Pallets (env.use_platform=False, mapa VIEJO): pista via
        plot_path_vortex.draw_map(obstacles.json) + ruta A*.

    A cada paso traza el camino REAL del robot y el punto-guia del vortex.
    Guarda un PNG al terminar cada episodio. `every` = cada cuantos pasos
    refresca el lienzo."""

    def __init__(self, env, map_data=None, every: int = 2):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPolygon, Rectangle

        self.plt = plt
        self.Rectangle = Rectangle
        self.use_platform = bool(getattr(env, "use_platform", False))
        self.every = max(1, int(every))
        self._n = 0
        self.live = not matplotlib.get_backend().lower().endswith("agg")
        if not self.live:
            print("[test_base] matplotlib sin backend interactivo (headless): "
                  "no habra ventana en vivo, solo se guardaran los PNG.")

        plt.ion()
        self.fig, ax = plt.subplots(figsize=(13, 9))
        self.ax = ax
        self.fig.patch.set_facecolor("#1e222b")
        ax.set_facecolor("#1e222b"); ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2, color="#ffffff", zorder=0)
        ax.tick_params(colors="#ffffff")
        ax.set_xlabel("X [m]", color="#ffffff"); ax.set_ylabel("Y [m]", color="#ffffff")

        # ── Fondo estatico del mundo activo ─────────────────────────────────
        if self.use_platform:
            he = float(C.PLATFORM_HALF_EXTENT)
            ax.add_patch(Rectangle((-he, -he), 2 * he, 2 * he, fill=False,
                                   ec="#ffa502", lw=2.0, zorder=1))
            zone = getattr(env, "platform_zone", None)
            if zone is not None:                          # zona segura erosionada
                zx, zy = zone.exterior.xy
                ax.add_patch(MPolygon(np.column_stack((zx, zy)), closed=True,
                                      fc="#00b894", ec="#00cec9", alpha=0.12,
                                      lw=0.8, zorder=1, label="Safe zone"))
            ax.set_xlim(-he - 1, he + 1); ax.set_ylim(-he - 1, he + 1)
        else:
            from rl_ws.utils.plot_path_vortex import draw_map          # pista de pallets (fondo)
            draw_map(ax, map_data)

        # ── Escena por episodio (se rellena en set_scene, cambia cada reset) ─
        (self.route_line,) = ax.plot([], [], "--", color="#ffffff", lw=1.3, alpha=0.45,
                                     zorder=4, label="Ruta")
        (self.wp_dots,) = ax.plot([], [], "o", color="#ff9f43", ms=5, ls="None", zorder=6)
        (self.goal_dot,) = ax.plot([], [], "*", color="#ff4757", ms=20, ls="None",
                                   zorder=8, label="Meta")
        self._obs_patch = None                            # caja del obstaculo virtual

        # ── Objetos dinamicos (se actualizan por paso) ──────────────────────
        (self.path_line,) = ax.plot([], [], "-", color="#00e5ff", lw=2.2, alpha=0.95,
                                    zorder=5, label="Robot (real)")
        (self.robot_dot,) = ax.plot([], [], "o", color="#00e5ff", ms=10, mec="#ffffff", zorder=9)
        (self.guide_dot,) = ax.plot([], [], "o", color="#ffeaa7", ms=8, mec="#d63031",
                                    zorder=9, label="Guia inmediata (vortex)")
        # lookahead: los N puntos futuros que VE la politica (estela verde adelante)
        (self.look_line,) = ax.plot([], [], "-", color="#55efc4", lw=1.2, alpha=0.7, zorder=7)
        (self.look_dots,) = ax.plot([], [], "o", color="#55efc4", ms=5, mec="#00b894",
                                    zorder=8, label="Lookahead (futuro)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, facecolor="#1e222b",
                  edgecolor="#a4b0be", labelcolor="#ffffff")
        self._rx, self._ry = [], []
        self.fig.canvas.draw(); plt.pause(0.001)

    def set_scene(self, waypoints, goal_xy, virtual_obstacle=None):
        """(Re)dibuja la ruta, la meta y el obstaculo virtual del episodio
        actual — cambian en cada reset() cuando el goal es random."""
        wp = np.asarray(waypoints, dtype=float)
        self.route_line.set_data(wp[:, 0], wp[:, 1])
        self.wp_dots.set_data(wp[:, 0], wp[:, 1])
        self.goal_dot.set_data([goal_xy[0]], [goal_xy[1]])
        if self._obs_patch is not None:
            self._obs_patch.remove(); self._obs_patch = None
        if virtual_obstacle is not None:
            o = virtual_obstacle
            self._obs_patch = self.ax.add_patch(self.Rectangle(
                (o.x - o.hx, o.y - o.hy), 2 * o.hx, 2 * o.hy,
                fc="#e17055", ec="#d63031", alpha=0.95, lw=1.4, zorder=3))
        if not self.use_platform:                         # pallets: encuadrar a la ruta
            pad = 1.0
            self.ax.set_xlim(wp[:, 0].min() - pad, wp[:, 0].max() + pad)
            self.ax.set_ylim(wp[:, 1].min() - pad, wp[:, 1].max() + pad)

    def start_episode(self, ep: int):
        self._rx, self._ry = [], []
        self.path_line.set_data([], [])
        self.robot_dot.set_data([], []); self.guide_dot.set_data([], [])
        self.look_line.set_data([], []); self.look_dots.set_data([], [])
        self.ax.set_title(f"Ep {ep}: siguiendo la trayectoria (EN VIVO)",
                          color="#ffffff", fontweight="bold")
        self.fig.canvas.draw_idle(); self.plt.pause(0.001)

    def update(self, robot_xy, guide_xy, lookahead_xy=None):
        self._rx.append(float(robot_xy[0])); self._ry.append(float(robot_xy[1]))
        self._n += 1
        self.path_line.set_data(self._rx, self._ry)
        self.robot_dot.set_data([robot_xy[0]], [robot_xy[1]])
        self.guide_dot.set_data([guide_xy[0]], [guide_xy[1]])
        if lookahead_xy is not None and len(lookahead_xy):
            la = np.asarray(lookahead_xy, dtype=float)
            self.look_dots.set_data(la[:, 0], la[:, 1])
            # linea robot -> puntos futuros, para ver la "estela" que anticipa
            self.look_line.set_data(np.r_[robot_xy[0], la[:, 0]], np.r_[robot_xy[1], la[:, 1]])
        if self._n % self.every == 0:
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def end_episode(self, ep: int, reached: bool):
        tag = "META ALCANZADA" if reached else "fin"
        self.ax.set_title(f"Ep {ep}: {tag}  ({len(self._rx)} pasos)",
                          color="#ffffff", fontweight="bold")
        self.fig.canvas.draw_idle(); self.plt.pause(0.001)

    def close(self):
        self.plt.ioff(); self.plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description="Test de una politica base entrenada (.pt)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--stochastic", action="store_true",
                    help="muestrea de la distribucion (default: deterministico = media)")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="grafica la trayectoria de cada episodio EN VIVO (--no-plot para desactivar)")
    args = ap.parse_args()

    # Resuelve el checkpoint sin importar el CWD: prueba tal cual, luego relativo
    # a la raiz del proyecto y a checkpoints_base/ (asi vale pasar la ruta corta).
    ckpt = args.checkpoint
    if not os.path.isfile(ckpt):
        for cand in (os.path.join(_ROOT, ckpt),
                     os.path.join(_ROOT, "checkpoints_base", os.path.basename(ckpt))):
            if os.path.isfile(cand):
                ckpt = cand
                break
    if not os.path.isfile(ckpt):
        raise SystemExit(
            f"No existe el checkpoint: {args.checkpoint}\n"
            f"  (buscado tambien en {_ROOT}/ y {_ROOT}/checkpoints_base/)")
    args.checkpoint = ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    env = BaseRosEnv()   # se conecta al bridge (espera su feedback); el bridge abre el viewer
    policy = load_policy(args.checkpoint, env.obs_dim, env.act_len, device)

    live = None
    if args.plot:
        # Fondo del mapa viejo (pallets) solo si el env NO usa la plataforma nueva.
        map_data = None
        if not env.use_platform:
            with open(NAV_JSON, "r") as f:
                map_data = json.load(f)
        live = LiveTrajectoryPlot(env, map_data=map_data)

    global_terms = defaultdict(float)   # acumulado de todos los episodios
    global_steps = 0
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            ep_reward, steps, done, info = 0.0, 0, False, {}
            ep_terms = defaultdict(float)
            if live:
                # La ruta/meta/obstaculo cambian cada reset (goal random en modo
                # plataforma) -> redibujar la escena antes de trazar el episodio.
                live.set_scene(env.waypoints, env.goal_xy, env.virtual_obstacle)
                live.start_episode(ep + 1)
            while not done:
                action = pick_action(policy, obs, device, args.stochastic)

                #if len(action) > 2:
                #    action[2:] = 0.0
                obs, rew, done, info = env.step(action)
                ep_reward += rew
                steps += 1
                for k, v in info.get("reward_terms", {}).items():
                    ep_terms[k] += v
                    global_terms[k] += v
                if live and "xy" in info:
                    live.update(info["xy"], info["guide"], info.get("lookahead_xy"))
            global_steps += steps
            tag = "🏁 META" if info.get("reached") else "fin"
            print(f"[Ep {ep + 1}/{args.episodes}] {tag}  reward={ep_reward:8.2f}  "
                  f"steps={steps}  wp={info.get('wp')}  dist_goal={info.get('dist_goal', 0.0):.2f}")
            print_reward_breakdown(f"Ep {ep + 1}", ep_terms, steps)
            if live:
                live.end_episode(ep + 1, info.get("reached", False))
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        if global_steps:
            print_reward_breakdown("TOTAL (todos los episodios)", global_terms, global_steps)
        if live:
            live.close()
        env.close()


if __name__ == "__main__":
    main()
