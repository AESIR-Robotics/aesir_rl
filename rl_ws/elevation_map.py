"""
elevation_map.py
=================
Convierte un OctoMap (.bt) en un mapa de elevación 2.5D que el PPO puede
usar como observación tipo "imagen". Diseñado para que la conversión
pesada (leer el octree) se haga UNA sola vez al iniciar el entorno, y que
cada step de RL solo haga un recorte/rotación rápida sobre un array de
numpy ya calculado (barato, apto para miles de steps/segundo).

Flujo:
    1. octree_to_global_elevation()  -> corre UNA vez, al construir el env
    2. classify_elevation()          -> corre UNA vez, junto con lo anterior
    3. get_egocentric_patch()        -> corre CADA STEP, con la pose actual
                                         del robot (rápido, solo numpy)

Valores semánticos en el mapa clasificado (ver classify_elevation()):
    -1.0  -> VOID: hueco entre pallets, el robot NO puede pisar ahí (se cae)
     0.0  -> WALKABLE: superficie de un pallet, aquí SÍ camina el robot
     1.0  -> OBSTACLE: un fatal_stick (más alto que un pallet), evitar
"""

import os
import numpy as np
import pyoctomap
from scipy.ndimage import affine_transform

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_BT_PATH = os.path.join(_HERE, "occupied_map.bt")
_FALLBACK_BT_PATH = os.path.join(_PROJECT_ROOT, "models", "occupied_map.bt")


UNKNOWN = -1.0
FLOOR = 0.0
OBSTACLE = 1.0


# ----------------------------------------------------------------------
# 1. Octree -> grilla de elevación GLOBAL (correr una sola vez)
# ----------------------------------------------------------------------
def octree_to_global_elevation(bt_path, resolution, margin=0.5):
    """
    Lee el .bt y colapsa el eje Z: para cada columna (x,y) guarda la altura
    MÁXIMA de cualquier voxel ocupado en esa columna. Si no hay nada
    ocupado en la columna, queda como NaN (= desconocido, se clasifica
    después).

    Devuelve:
        elevation : np.ndarray (H, W) float, NaN donde no hay datos
        origin_xy : (2,) esquina mínima (x,y) del mapa en coords mundo
        resolution: tamaño de celda en metros
        z_min     : altura MÍNIMA real de cualquier voxel ocupado en TODO el octomap
        z_max     : altura MÁXIMA real (típicamente el borde superior de un fatal_stick)
    """
    tree = pyoctomap.OcTree(resolution)
    ok = tree.readBinary(bt_path)
    if not ok:
        raise RuntimeError(f"No se pudo leer {bt_path}")

    # extractPointCloud() de pyoctomap regresa (ocupados, libres) ya como
    # arrays de numpy -- evita el loop de Python sobre begin_leafs() que
    # usábamos con octomap-python (mucho más rápido en mapas grandes).
    points, _empty = tree.extractPointCloud()
    points = np.asarray(points)

    if len(points) == 0:
        raise ValueError("El octomap no tiene voxeles ocupados.")

    xy_min = points[:, :2].min(axis=0) - margin
    xy_max = points[:, :2].max(axis=0) + margin
    dims = np.ceil((xy_max - xy_min) / resolution).astype(int)

    elevation = np.full(dims, np.nan, dtype=np.float32)

    ix = np.clip(((points[:, 0] - xy_min[0]) / resolution).astype(int), 0, dims[0] - 1)
    iy = np.clip(((points[:, 1] - xy_min[1]) / resolution).astype(int), 0, dims[1] - 1)
    z = points[:, 2]

    z_min = float(z.min())
    z_max = float(z.max())

    # Para cada columna, nos quedamos con el Z máximo (np.maximum.at soporta duplicados)
    flat_idx = ix * dims[1] + iy
    flat_elev = elevation.ravel()
    # inicializa en -inf temporalmente para poder usar np.maximum.at, luego regresamos a NaN donde no tocó nada
    tmp = np.full(dims[0] * dims[1], -np.inf, dtype=np.float32)
    np.maximum.at(tmp, flat_idx, z)
    touched = tmp > -np.inf
    flat_elev[touched] = tmp[touched]
    elevation = flat_elev.reshape(dims)

    return elevation, xy_min, resolution, z_min, z_max


# ----------------------------------------------------------------------
# 2. Elevación -> mapa semántico (piso / obstáculo / desconocido)
# ----------------------------------------------------------------------
def classify_elevation(elevation, pallet_max_height=0.25):
    """
    3 clases, con la semántica REAL de tu pista (no genérica):

        VOID (-1)      -> la columna NO tiene ningún objeto (ni pallet ni
                           stick). Esto es un HUECO entre pallets -- el
                           robot NO puede pisar ahí, se caería.
        WALKABLE (0)   -> la columna tiene un objeto BAJO (un pallet) --
                           esta es la superficie real donde el robot camina.
        OBSTACLE (1)   -> la columna tiene un objeto ALTO (un fatal_stick,
                           por encima de pallet_max_height) -- evitar,
                           el robot no debe chocar contra esto.

    pallet_max_height: separador real entre "es un pallet" y "es un stick".
    Por default 0.25m -- calculado a partir de la distribución real de tu
    pista: los pallets caen entre 0.075-0.100m, los sticks saltan directo
    a ~0.425-0.450m, con un hueco grande en medio. Si tu pista tiene otras
    dimensiones, ajusta este valor viendo el histograma de alturas real
    (occtree_to_global_elevation ya te da z_min/z_max para calibrarlo).
    """
    sem = np.full(elevation.shape, UNKNOWN, dtype=np.float32)  # UNKNOWN == VOID

    has_data = ~np.isnan(elevation)
    sem[has_data] = FLOOR  # FLOOR == WALKABLE (tiene un objeto, por ahora asumimos pallet)

    is_obstacle = has_data & (elevation > pallet_max_height)
    sem[is_obstacle] = OBSTACLE  # pero si es más alto que un pallet, es un stick

    return sem


# ----------------------------------------------------------------------
# 3. Recorte egocéntrico rotado (correr en CADA step del entorno RL)
# ----------------------------------------------------------------------
def get_egocentric_patch(global_map, origin_xy, resolution,
                          robot_xy, robot_yaw,
                          patch_size_m=3.0, patch_pixels=64,
                          fill_value=UNKNOWN):
    """
    Extrae una ventana local del mapa GLOBAL ya precomputado, centrada en
    el robot y rotada para que "arriba" en la imagen = "adelante" del robot.

    Esto NO toca el octree — solo hace una transformación afín sobre el
    array de numpy ya calculado, así que es apto para correr en cada step.

    robot_xy  : (x, y) posición actual del robot en coords mundo
    robot_yaw : orientación actual del robot (radianes)
    patch_size_m : tamaño físico del parche (ej. 3.0 = ventana de 3x3 m)
    patch_pixels  : resolución de salida en píxeles (ej. 64 -> imagen 64x64)

    Devuelve: np.ndarray (patch_pixels, patch_pixels), listo para meter
    como observación tipo imagen al PPO (agrégale un canal: obs[None, :, :]).
    """
    out_res = patch_size_m / patch_pixels  # metros por pixel de salida

    # Centro del robot en índices de la grilla GLOBAL (float, para sub-pixel)
    cx = (robot_xy[0] - origin_xy[0]) / resolution
    cy = (robot_xy[1] - origin_xy[1]) / resolution

    cos_a, sin_a = np.cos(robot_yaw), np.sin(robot_yaw)

    # Matriz que mapea índice de SALIDA (fila=adelante, col=lateral) -> índice de ENTRADA (mapa global)
    # escala: de pixeles de salida a "celdas de mapa global equivalentes"
    scale = out_res / resolution
    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]]) * scale

    half = patch_pixels / 2.0
    offset = np.array([cx, cy]) - rot @ np.array([half, half])

    patch = affine_transform(
        global_map,
        matrix=rot,            # mapea índice de SALIDA -> índice del mapa GLOBAL
        offset=offset,
        output_shape=(patch_pixels, patch_pixels),
        order=0,              # nearest-neighbor: no queremos "inventar" alturas intermedias
        cval=fill_value,
        mode="constant",
    )
    return patch


# ----------------------------------------------------------------------
# 4. Escala de grises + máscara circular (lo que pidió el usuario)
# ----------------------------------------------------------------------
def semantic_to_grayscale(sem_patch, unknown_value=0.5):
    """
    Convierte el parche semántico {-1 desconocido, 0 piso, 1 obstáculo}
    a escala de grises continua en [0,1]:
        1.0 (blanco) -> piso / vacío
        0.0 (negro)  -> obstáculo / pared
        unknown_value (gris medio, default 0.5) -> desconocido
    """
    gray = np.full(sem_patch.shape, unknown_value, dtype=np.float32)
    gray[sem_patch == FLOOR] = 1.0
    gray[sem_patch == OBSTACLE] = 0.0
    return gray


def apply_circular_mask(patch, radius_m, patch_size_m, out_value=0.5):
    """
    Pone en `out_value` (gris neutro por default) todo lo que quede fuera
    del círculo de radio `radius_m` centrado en el parche. El robot se
    asume siempre en el centro exacto del parche (esa es la definición
    de "egocéntrico").
    """
    h, w = patch.shape
    res_x = patch_size_m / w
    res_y = patch_size_m / h

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dist_m = np.sqrt(((xx - cx) * res_x) ** 2 + ((yy - cy) * res_y) ** 2)

    masked = patch.copy()
    masked[dist_m > radius_m] = out_value
    return masked


def get_circular_heatmap(global_semantic_map, origin_xy, resolution,
                          robot_xy, robot_yaw,
                          radius_m=1.0, patch_pixels=64,
                          unknown_value=0.5, shape="square"):
    """
    Todo-en-uno: esta es la función que llamas EN CADA STEP del entorno RL.

    shape: "square" (default, ventana cuadrada de 2*radius_m x 2*radius_m,
           SIN recortar esquinas) o "circle" (recorta a un círculo de
           radio radius_m, poniendo gris neutro fuera de él).

    Devuelve un array (patch_pixels, patch_pixels) en escala de grises [0,1]:
        1.0 = blanco = piso/vacío
        0.0 = negro  = obstáculo/pared
        0.5 = gris   = desconocido (o fuera del círculo, si shape="circle")

    El robot siempre está exactamente en el centro del array, orientado
    hacia "arriba" (fila menor = adelante del robot).

    NOTA: esta versión NO distingue pallets de sticks (todo lo que no es
    piso es "obstáculo" por igual). Para diferenciarlos por altura real,
    usa get_circular_height_heatmap() más abajo.
    """
    # patch_size_m = lado completo del cuadrado = 2 * radius_m
    patch_size_m = radius_m * 2.0
    sem_patch = get_egocentric_patch(
        global_semantic_map, origin_xy, resolution,
        robot_xy, robot_yaw,
        patch_size_m=patch_size_m, patch_pixels=patch_pixels,
        fill_value=UNKNOWN,
    )
    gray = semantic_to_grayscale(sem_patch, unknown_value=unknown_value)
    if shape == "circle":
        gray = apply_circular_mask(gray, radius_m, patch_size_m, out_value=unknown_value)
    # shape == "square" -> no se recorta nada, se queda el cuadrado completo
    return gray


# ----------------------------------------------------------------------
# 5. Escala de grises por ALTURA REAL (diferencia pallets de sticks)
# ----------------------------------------------------------------------
def height_to_grayscale(height_patch, z_min, z_max, robot_z, eps=1e-6):
    """
    Convierte alturas Z reales a escala de grises [0,1], usando robot_z
    como PUNTO MEDIO del umbral (0.5), no el punto medio de z_min/z_max.

    Esto es justo el diseño que pediste:
        - z_min global -> 1.0 (blanco)  = lo más bajo del mapa
        - robot_z       -> 0.5 (gris medio) = referencia = el robot
        - z_max global -> 0.0 (negro)   = lo más alto (el fatal_stick más alto)

    Como un pallet normalmente queda un poco MÁS BAJO que la altura actual
    del robot (el robot está parado ENCIMA de él), su gris sale un poco
    MÁS CLARO que 0.5. Un fatal_stick, al ser más alto que el robot, sale
    MÁS OSCURO que 0.5. Así diferencias pallets de sticks sin necesitar
    sus nombres -- la altura real ya los distingue.

    Se normaliza en dos tramos (por debajo y por encima de robot_z) para
    que robot_z caiga EXACTO en 0.5 sin importar si z_min/z_max son
    simétricos respecto al robot o no.
    """
    height_patch = np.asarray(height_patch, dtype=np.float32)
    norm = np.empty_like(height_patch)

    below = height_patch <= robot_z
    denom_low = max(robot_z - z_min, eps)
    denom_high = max(z_max - robot_z, eps)

    norm[below] = 0.5 * (height_patch[below] - z_min) / denom_low
    norm[~below] = 0.5 + 0.5 * (height_patch[~below] - robot_z) / denom_high
    norm = np.clip(norm, 0.0, 1.0)

    gray = 1.0 - norm  # invertido: bajo=blanco, alto=negro
    return gray.astype(np.float32)


def get_circular_height_heatmap(global_elevation, origin_xy, resolution,
                                 robot_xy, robot_z, robot_yaw,
                                 z_min, z_max,
                                 radius_m=1.0, patch_pixels=64,
                                 unknown_value=0.5, shape="square"):
    """
    Versión que SÍ diferencia pallets de sticks por altura real. Esta es
    la función que quieres llamar EN CADA STEP del entorno RL (reemplaza
    a get_circular_heatmap si necesitas la distinción de altura).

    shape: "square" (default, ventana completa 2*radius_m x 2*radius_m)
           o "circle" (recorta a un círculo de radio radius_m).

    global_elevation: el array crudo con NaN que devuelve
                       octree_to_global_elevation() (NO el clasificado).
    robot_z: altura ACTUAL del robot (cambia si sube/baja de un pallet,
             así que se lo pasas de nuevo en cada step).
    z_min, z_max: los devuelve octree_to_global_elevation(), son fijos
                  para todo el mapa (no cambian con la posición del robot).
    """
    # Columnas sin dato (NaN) = no hay obstáculo ahí = tratamos como el
    # punto más bajo del mapa (blanco, igual que "piso").
    filled = np.where(np.isnan(global_elevation), z_min, global_elevation)

    patch_size_m = radius_m * 2.0
    height_patch = get_egocentric_patch(
        filled, origin_xy, resolution,
        robot_xy, robot_yaw,
        patch_size_m=patch_size_m, patch_pixels=patch_pixels,
        fill_value=z_min,  # fuera del área mapeada = igual que "sin obstáculo"
    )

    gray = height_to_grayscale(height_patch, z_min, z_max, robot_z)
    if shape == "circle":
        gray = apply_circular_mask(gray, radius_m, patch_size_m, out_value=unknown_value)
    return gray


# ----------------------------------------------------------------------
# MODO STANDALONE — correr este script directo para ver el mapa de
# elevación sin necesitar el entorno de PPO todavía.
#
# Uso:
#   python3 elevation_map.py occupied_map.bt --resolution 0.05
#   python3 elevation_map.py inflated_map.bt --obstacle-height 0.05 \
#           --robot-x 3.0 --robot-y -2.5 --robot-yaw 90
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Genera y visualiza el mapa de elevación/semántico desde un .bt de OctoMap."
    )
    parser.add_argument("bt_path", nargs="?", default=None,
                        help="Ruta al archivo .bt (occupied_map.bt o inflated_map.bt). Por defecto usa el generado en rl_ws.")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--pallet-max-height", type=float, default=0.25,
                        help="Alturas mayores a esto se clasifican como OBSTACLE (fatal_stick). "
                             "Por debajo = WALKABLE (pallet). Calibra viendo el histograma real "
                             "de tu pista (ver z_min/z_max que imprime el script).")
    parser.add_argument("--robot-x", type=float, default=None,
                        help="Si se da, además dibuja un parche egocéntrico en esta posición X")
    parser.add_argument("--robot-y", type=float, default=None)
    parser.add_argument("--robot-yaw", type=float, default=0.0,
                        help="Orientación del robot en GRADOS (default 0)")
    parser.add_argument("--patch-size", type=float, default=3.0,
                        help="Tamaño físico del parche egocéntrico cuadrado en metros (modo clases)")
    parser.add_argument("--patch-pixels", type=int, default=64)
    parser.add_argument("--heatmap", action="store_true",
                        help="Heatmap circular en escala de grises (radio 1m), SIN distinguir "
                             "pallets de sticks (todo obstáculo = negro por igual)")
    parser.add_argument("--height-heatmap", action="store_true",
                        help="Igual que --heatmap pero usando ALTURA REAL: pallets salen más "
                             "claros que el robot, sticks (más altos) salen más oscuros")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Radio del heatmap en metros (mitad del lado si shape=square)")
    parser.add_argument("--shape", choices=["square", "circle"], default="square",
                        help="Forma del recorte egocéntrico (default: square)")
    parser.add_argument("--robot-z", type=float, default=0.15,
                        help="Altura actual del robot en metros (solo con --height-heatmap; "
                             "es el punto que se mapea a gris medio=0.5)")
    args = parser.parse_args()

    bt_path = args.bt_path
    if bt_path is None:
        if os.path.exists(_DEFAULT_BT_PATH):
            bt_path = _DEFAULT_BT_PATH
        elif os.path.exists(_FALLBACK_BT_PATH):
            bt_path = _FALLBACK_BT_PATH
        else:
            raise FileNotFoundError(
                f"No se encontró un .bt para leer. Pasa la ruta explícitamente o genera uno en { _DEFAULT_BT_PATH }"
            )

    print(f"📥 Leyendo octomap: {bt_path}")
    elevation, origin, res, z_min, z_max = octree_to_global_elevation(
        bt_path, resolution=args.resolution, margin=args.margin
    )
    print(f"   -> grilla global: {elevation.shape}, origen: {origin}")
    print(f"   -> rango de alturas real del octomap: z_min={z_min:.3f}  z_max={z_max:.3f}")

    sem = classify_elevation(elevation, pallet_max_height=args.pallet_max_height)
    n_unknown = int((sem == UNKNOWN).sum())
    n_floor = int((sem == FLOOR).sum())
    n_obstacle = int((sem == OBSTACLE).sum())
    print(f"   -> hueco/void={n_unknown}  pallet/walkable={n_floor}  stick/obstacle={n_obstacle}")

    # Matplotlib se importa aquí adentro (no al inicio del módulo) para que
    # elevation_map.py se pueda importar desde tu entorno de RL sin forzar
    # una dependencia de matplotlib en el loop de entrenamiento.
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show_patch = args.robot_x is not None and args.robot_y is not None
    fig, axes = plt.subplots(1, 2 if show_patch else 1, figsize=(14 if show_patch else 7, 6))
    if not show_patch:
        axes = [axes]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    sem_cmap = ListedColormap(["white", "forestgreen", "firebrick"])  # VOID, WALKABLE, OBSTACLE
    sem_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], sem_cmap.N)

    axes[0].imshow(sem.T, origin="lower", cmap=sem_cmap, norm=sem_norm,
                   extent=[origin[0], origin[0] + sem.shape[0] * res,
                           origin[1], origin[1] + sem.shape[1] * res])
    axes[0].set_title("Mapa semántico GLOBAL\n"
                      "(blanco=hueco/no pisar, verde=pallet transitable, rojo=stick/obstáculo)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    if show_patch:
        yaw_rad = np.deg2rad(args.robot_yaw)

        if args.height_heatmap:
            patch = get_circular_height_heatmap(
                elevation, origin, res,
                robot_xy=(args.robot_x, args.robot_y), robot_z=args.robot_z, robot_yaw=yaw_rad,
                z_min=z_min, z_max=z_max,
                radius_m=args.radius, patch_pixels=args.patch_pixels, shape=args.shape,
            )
            cmap_patch, vmin_patch, vmax_patch = "gray", 0.0, 1.0
            patch_title = (f"Heatmap por ALTURA REAL [{args.shape}] (radio={args.radius}m, robot_z={args.robot_z}m,\n"
                           f"pallets=más claro que 0.5, sticks=más oscuro que 0.5)")
        elif args.heatmap:
            patch = get_circular_heatmap(
                sem, origin, res,
                robot_xy=(args.robot_x, args.robot_y), robot_yaw=yaw_rad,
                radius_m=args.radius, patch_pixels=args.patch_pixels, shape=args.shape,
            )
            cmap_patch, vmin_patch, vmax_patch = "gray", 0.0, 1.0
            patch_title = (f"Heatmap [{args.shape}] (radio={args.radius}m, escala de grises,\n"
                           f"robot al centro, blanco=vacío, negro=pared)")
        else:
            patch = get_egocentric_patch(
                sem, origin, res,
                robot_xy=(args.robot_x, args.robot_y), robot_yaw=yaw_rad,
                patch_size_m=args.patch_size, patch_pixels=args.patch_pixels,
            )
            cmap_patch, vmin_patch, vmax_patch = "RdYlGn_r", -1, 1
            patch_title = (f"Parche EGOCÉNTRICO ({args.patch_size}m, robot al centro,\n"
                           f"arriba = adelante del robot)")

        axes[0].plot(args.robot_x, args.robot_y, "b^", markersize=12, label="robot")
        axes[0].legend()

        axes[1].imshow(patch.T, origin="lower", cmap=cmap_patch, vmin=vmin_patch, vmax=vmax_patch)
        axes[1].set_title(patch_title)
        half = args.patch_pixels / 2
        axes[1].axhline(half, color="blue", lw=0.5)
        axes[1].axvline(half, color="blue", lw=0.5)

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(args.bt_path))[0]
    out_name = f"{base}_elevation.png"
    plt.savefig(out_name, dpi=150)
    print(f"✅ Guardado: {out_name}")
    plt.show()