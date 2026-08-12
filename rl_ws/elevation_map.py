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
def height_to_grayscale(height_patch, robot_z, z_range_m, eps=1e-6):
    """
    Convierte alturas Z reales a escala de grises [0,1] en una VENTANA FIJA
    RELATIVA AL ROBOT de +-z_range_m:

        robot_z - z_range_m  -> 1.0 (blanco) = z_range_m por DEBAJO del robot
        robot_z              -> 0.5 (gris medio) = el robot
        robot_z + z_range_m  -> 0.0 (negro)  = z_range_m por ENCIMA del robot

    Lo que quede fuera de la ventana se recorta (satura), que es el
    comportamiento deseado: un poste de 2.4 m es "obstaculo intransitable",
    no hace falta resolver CUANTO mide.

    POR QUE RELATIVA AL ROBOT Y NO A LOS EXTREMOS DE LA PISTA: antes se
    normalizaba con el z_min/z_max GLOBALES del octomap de cada pista, lo que
    tenia dos defectos medidos:

      1) OUTLIERS. En pallets, 37 celdas de 10103 (0.37%) -- los fatal_stick a
         2.45 m -- fijaban z_max y aplastaban TODO el terreno transitable
         (0.08..0.15 m, 8 cm de relieve) a 4 niveles de gris de 255. En
         steps1m el relieve equivalente recibia 212. O sea: el MISMO
         obstaculo fisico se veia 53x mas contrastado segun la pista, lo que
         rompe la transferencia entre pistas de una sola politica.

      2) COLAPSO AL SUBIR. El divisor de arriba era (z_max - robot_z), asi que
         se encogia a medida que el robot trepaba. En steps1m (z_max=0.375,
         escalones hasta 0.38) al llegar arriba valia ~0 y TODO lo que
         estuviera por encima del robot saturaba a negro puro: el robot se
         quedaba ciego hacia arriba justo mientras trepaba.

    Con una ventana fija y simetrica el mapa altura->gris es el MISMO en toda
    pista y en todo instante, y no hay divisor que colapse.
    """
    h = np.asarray(height_patch, dtype=np.float32)
    norm = 0.5 + 0.5 * (h - robot_z) / max(float(z_range_m), eps)
    gray = 1.0 - np.clip(norm, 0.0, 1.0)   # invertido: bajo=blanco, alto=negro
    return gray.astype(np.float32)


def get_circular_height_heatmap(global_elevation, origin_xy, resolution,
                                 robot_xy, robot_z, robot_yaw,
                                 z_min, z_range_m,
                                 radius_m=1.0, patch_pixels=64,
                                 unknown_value=0.5, shape="square"):
    """
    Heatmap egocentrico de altura real. Esta es la funcion que se llama EN
    CADA STEP del entorno RL.

    shape: "square" (default, ventana completa 2*radius_m x 2*radius_m)
           o "circle" (recorta a un círculo de radio radius_m).

    global_elevation: el array crudo con NaN que devuelve
                       octree_to_global_elevation() (NO el clasificado).
    robot_z: altura ACTUAL del robot (cambia si sube/baja de un escalon,
             así que se lo pasas de nuevo en cada step).
    z_min: SOLO se usa como valor de relleno (NaN dentro del mapa y area
           fuera del mapa = "sin obstaculo" = lo mas bajo). NO interviene ya
           en la normalizacion a grises.
    z_range_m: media-ventana de la normalizacion, RELATIVA a robot_z
               (config.HEATMAP_Z_RANGE_M). Ver height_to_grayscale para el
               porque de que sea relativa y no los extremos de la pista.
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

    gray = height_to_grayscale(height_patch, robot_z, z_range_m)
    if shape == "circle":
        gray = apply_circular_mask(gray, radius_m, patch_size_m, out_value=unknown_value)
    return gray


# ----------------------------------------------------------------------
# 5. Desnivel local por flipper (geometria PURA -- el juicio de "es
#    trepable" o no vive en la capa de tarea/reward, no aqui)
# ----------------------------------------------------------------------
def sample_elevation_at(global_elevation, origin_xy, resolution, z_min, points_xy):
    """Altura REAL (metros) en cada punto mundo de `points_xy` (...,2), por
    vecino-mas-cercano sobre la grilla global. NaN (sin dato) -> z_min,
    MISMA convencion que get_circular_height_heatmap (sin dato = piso).
    Acepta cualquier shape (...,2) -- devuelve la misma sin el ultimo eje.
    Si `global_elevation` ya viene sin NaN (ver MapContext.elevation_filled)
    el np.where de abajo es un no-op barato."""
    pts = np.asarray(points_xy, dtype=np.float64)
    ix = np.clip(np.round((pts[..., 0] - origin_xy[0]) / resolution).astype(int),
                0, global_elevation.shape[0] - 1)
    iy = np.clip(np.round((pts[..., 1] - origin_xy[1]) / resolution).astype(int),
                0, global_elevation.shape[1] - 1)
    h = global_elevation[ix, iy]
    return np.where(np.isnan(h), z_min, h)


def flipper_terrain_edge(global_elevation, origin_xy, resolution, z_min,
                         robot_xy, robot_yaw, mounts_xy, axis_sign,
                         look_ahead_m, min_step_m, max_climb_m, max_descent_m,
                         n_samples=8):
    """Busca el PRIMER borde real en la direccion de extension de cada flipper,
    SUBIDA o BAJADA -- SOLO geometria + umbrales de tarea (ver
    config.FLIPPER_TERRAIN_MIN_STEP_M / MAX_CLIMB_M / MAX_DESCENT_M):

    Escanea `n_samples` puntos a distancias crecientes (hasta look_ahead_m)
    desde el punto de montaje, hacia el lado al que ese flipper se extiende
    (adelante para delanteros, atras para traseros). Para cada flipper,
    devuelve el PRIMER punto donde |desnivel| sobre el mount cruza min_step_m
    (filtra ruido del piso -- micro-irregularidades que no son un borde real):
        found      (4,) bool  -- se encontro un borde en el rango escaneado
        d_edge     (4,) float -- distancia (m) del mount al borde (nan si no)
        h_edge     (4,) float -- desnivel CON SIGNO (m) sobre el mount:
                                 >0 escalon a subir, <0 caida a bajar (nan si no)
        actionable (4,) bool  -- found AND el borde esta dentro de lo que el
                                 flipper puede atacar: subida <= max_climb_m
                                 (arriba de eso es pared) o bajada <=
                                 max_descent_m (mas abajo la punta no llega)

    OJO: al detectar |dh| (no solo dh>0), en terreno mixto el "primer borde"
    puede ser ahora una BAJADA donde antes se reportaba la siguiente subida.

    mounts_xy: (4,2) posicion de cada mount en frame LOCAL del robot.
    axis_sign: (4,) signo del lado de extension (+1 delanteros, -1 traseros;
               multiplicar por travel_dir para invertirlo en reversa, ver
               base_env.flipper_travel_dir)."""
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    R = np.array([[c, -s], [s, c]])   # local (x=adelante,y=izq) -> mundo

    mounts_world = robot_xy + np.asarray(mounts_xy, dtype=np.float64) @ R.T
    look_local = np.stack([np.sign(axis_sign), np.zeros(4)], axis=1)
    look_world = look_local @ R.T

    # Los n_samples+1 puntos (mount + escaneo) de los 4 flippers en UN solo
    # gather: (K,4,2) -> una llamada a sample_elevation_at en vez de n_samples+1.
    # `d` conserva la expresion original (look_ahead_m*k/n_samples) porque el
    # orden de las operaciones decide de que lado del np.round cae el punto.
    d_steps = np.array([look_ahead_m * k / n_samples for k in range(1, n_samples + 1)])
    pts = np.empty((n_samples + 1, 4, 2), dtype=np.float64)
    pts[0] = mounts_world
    pts[1:] = mounts_world + look_world * d_steps[:, None, None]

    h = sample_elevation_at(global_elevation, origin_xy, resolution, z_min, pts)
    dh = (h[1:] - h[0]).astype(np.float32)          # (n_samples,4)

    hit = np.abs(dh) >= min_step_m                  # (n_samples,4) subida O bajada
    found = hit.any(axis=0)                         # (4,)
    first = hit.argmax(axis=0)                      # primer k con desnivel (0 si ninguno)
    d_edge = np.where(found, d_steps[first], np.nan).astype(np.float32)
    h_edge = np.where(found, dh[first, np.arange(4)], np.nan).astype(np.float32)

    # Atacable = subida no-pared, o bajada que la punta todavia alcanza.
    # (con h_edge=nan ambas comparaciones dan False, y found ya es False)
    with np.errstate(invalid="ignore"):
        actionable = found & np.where(h_edge > 0,
                                      h_edge <= max_climb_m,
                                      -h_edge <= max_descent_m)
    return found, d_edge, h_edge, actionable


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