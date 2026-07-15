"""
view_octomap.py
================
Visualiza un archivo .bt de OctoMap en 3D usando matplotlib, sin ROS
ni octovis. Útil para verificar rápido que el mapa generado corresponde
a tu pista real.

Uso:
    python3 view_octomap.py occupied_map.bt --resolution 0.05
    python3 view_octomap.py inflated_map.bt --resolution 0.05 --max-points 20000
"""

import os
import sys

# --- Fix para conflicto ROS2 + pip con mpl_toolkits (paquete namespace) ---
# Esto DEBE ir antes de cualquier otro import (incluso numpy/octomap), por si
# alguno de ellos importa matplotlib/mpl_toolkits indirectamente y "fija"
# la resolución equivocada antes de que podamos corregir sys.path.
#
# 1. Quitamos por completo la ruta del sistema (apt/ROS2) que trae una copia
#    de mpl_toolkits incompatible con el matplotlib nuevo de pip.
_BAD_PATHS = ["/usr/lib/python3/dist-packages"]
sys.path = [p for p in sys.path if p not in _BAD_PATHS]

# 2. Por si algo ya alcanzó a importar mpl_toolkits/matplotlib con la ruta
#    vieja antes de este punto, lo purgamos de la caché de módulos.
for _mod in list(sys.modules):
    if _mod == "mpl_toolkits" or _mod.startswith("mpl_toolkits.") \
       or _mod == "matplotlib" or _mod.startswith("matplotlib."):
        del sys.modules[_mod]

import argparse
import numpy as np
import pyoctomap
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_BT_PATH = os.path.join(_HERE, "occupied_map.bt")
_FALLBACK_BT_PATH = os.path.join(_PROJECT_ROOT, "models", "occupied_map.bt")


def load_occupied_points(bt_path, resolution):
    tree = pyoctomap.OcTree(resolution)
    ok = tree.readBinary(bt_path)
    if not ok:
        raise RuntimeError(f"No se pudo leer {bt_path}")

    points, _empty = tree.extractPointCloud()
    points = np.asarray(points)
    return points, tree


def plot_points(points, title, out_name, max_points=30000):
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        title += f"  (mostrando {max_points} de más puntos)"

    try:
        # Import explícito: en sistemas con ROS2 suele haber dos matplotlib
        # (apt + pip) peleándose, y mplot3d no se auto-registra.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap="viridis", s=2, marker="s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), max(np.ptp(points[:, 2]), 0.5)])

    except (ImportError, ValueError) as e:
        # Fallback: vista 2D top-down (X-Y), color = altura Z.
        # Para navegación en piso esta vista suele ser más útil de todos modos.
        print(f"⚠️  Proyección 3D no disponible ({e}); usando vista 2D top-down.")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2],
                        cmap="viridis", s=4, marker="s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title + "  [vista 2D top-down, color=altura Z]")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Z (m)")

    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    print(f"Guardado: {out_name}")
    plt.show()


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("bt_path", nargs="?", default=None,
                        help="Ruta al archivo .bt. Por defecto usa el generado en rl_ws.")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=30000)
    args = parser.parse_args()

    bt_path = args.bt_path
    if bt_path is None:
        if os.path.exists(_DEFAULT_BT_PATH):
            bt_path = _DEFAULT_BT_PATH
        elif os.path.exists(_FALLBACK_BT_PATH):
            bt_path = _FALLBACK_BT_PATH
        else:
            raise FileNotFoundError(
                f"No se encontró un .bt para visualizar. Pasa la ruta explícitamente o genera uno en { _DEFAULT_BT_PATH }"
            )

    points, tree = load_occupied_points(bt_path, args.resolution)
    print(f"Voxeles ocupados: {len(points)}")
    base = os.path.splitext(os.path.basename(bt_path))[0]
    plot_points(points, args.bt_path, out_name=f"{base}_preview.png", max_points=args.max_points)