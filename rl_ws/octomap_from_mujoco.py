"""
octomap_from_mujoco.py
=======================
Genera un OctoMap 3D a partir de un entorno MuJoCo (XML), usando SOLO los
geoms de colisión (contype != 0), sin depender de ROS/ROS2.

Librerías usadas (todas standalone, instalables con pip):
    - mujoco          -> parsear el XML y obtener las poses reales de los geoms
    - pyoctomap        -> construir y guardar el octree (.bt), batch ops nativas
    - numpy / scipy   -> rasterización de geoms y dilatación morfológica

Instalación:
    pip install mujoco pyoctomap numpy scipy --break-system-packages

Flujo:
    1. Carga el XML con mujoco y hace forward kinematics (mj_forward) para
       obtener la posición y rotación MUNDIAL de cada geom (geom_xpos/geom_xmat).
    2. Rasteriza cada geom de colisión (box, por ahora; ver nota abajo para mesh)
       en una grilla booleana 3D con la resolución deseada.
    3. Convierte la grilla ocupada en un point cloud y lo inserta en un
       octomap.OcTree -> guarda occupied_map.bt
    4. Dilata (Minkowski sum) la grilla booleana con scipy.ndimage.binary_dilation
       usando un radio en metros (= radio del robot + margen de seguridad)
       -> guarda inflated_map.bt, listo para usarse como costmap de navegación.

Nota sobre geoms tipo "mesh":
    Rasterizar una malla arbitraria con precisión requiere point-in-mesh
    testing (costoso). Como tu pista usa colliders tipo "box" para todo lo
    que debe evitar el robot (pallets, sticks), este script solo usa esos.
    Si necesitas incluir geoms mesh como colisión real, la forma más simple
    es exportar su bounding box (ver función `mesh_aabb_fallback`) o generar
    un point cloud de sus vértices con trimesh y unirlo al `occupied_points`
    antes del paso 3.
"""

import argparse
import os
import numpy as np
import mujoco
import pyoctomap
from scipy.ndimage import binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_XML = os.path.join(_PROJECT_ROOT, "models", "plataform.xml")
_DEFAULT_RAW_OUT = os.path.join(_HERE, "occupied_map.bt")
_DEFAULT_INFLATED_OUT = os.path.join(_HERE, "inflated_map.bt")


# ----------------------------------------------------------------------
# 1. Extraer geoms de colisión en coordenadas MUNDIALES desde el XML
# ----------------------------------------------------------------------
# Geoms que NO deben entrar al mapa estatico (el obstaculo virtual se
# reposiciona por episodio; incluirlo congelaria una caja fantasma en el bt).
_EXCLUDE_GEOM_NAMES = {"virtual_obstacle", "maze_floor"}


def load_collision_boxes(xml_path, include_names_prefix=None, exclude_planes=True):
    """
    Devuelve (boxes, meshes, model, data):
      boxes : lista de dicts {"name","center","half_size","rot"} por geom BOX
              colisionable (contype != 0).
      meshes: lista de dicts {"name","gid","center","rot","aabb"} por geom MESH
              colisionable (aabb = (2,3) centro+half en frame local del geom) —
              se rasterizan aparte por RAYCAST (rasterize_meshes_into).
      model/data: el modelo cargado (lo reusa el raycast de meshes).

    include_names_prefix: si se da (ej. "fatal_"), solo toma geoms cuyo
    nombre empiece con ese prefijo.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # calcula geom_xpos / geom_xmat reales

    boxes, meshes = [], []
    for gid in range(model.ngeom):
        geom_type = model.geom_type[gid]
        contype = model.geom_contype[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"

        if exclude_planes and geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if contype == 0:
            continue          # geom puramente visual, no es obstáculo real
        if name in _EXCLUDE_GEOM_NAMES:
            continue          # dinamico por episodio, no va al mapa estatico
        if include_names_prefix and not name.startswith(include_names_prefix):
            continue

        center = data.geom_xpos[gid].copy()
        rot = data.geom_xmat[gid].reshape(3, 3).copy()

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            boxes.append({"name": name, "center": center,
                          "half_size": model.geom_size[gid].copy(), "rot": rot})
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            aabb = model.geom_aabb[gid].reshape(2, 3).copy()   # (centro, half) local
            meshes.append({"name": name, "gid": gid, "center": center,
                           "rot": rot, "aabb": aabb})
        # otros tipos (esfera/cilindro/capsula) no aparecen en las pistas

    return boxes, meshes, model, data


def _mesh_world_corners(m):
    """8 esquinas MUNDIALES del AABB local de un geom mesh."""
    c_local, h_local = m["aabb"][0], m["aabb"][1]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = c_local + signs * h_local
    return (m["rot"] @ local.T).T + m["center"]


def rasterize_meshes_into(grid, world_min, resolution, model, data, meshes):
    """
    Rasteriza los geoms MESH en la MISMA grilla que los boxes, por RAYCAST:
    para cada columna (x,y) dentro del AABB del mesh se lanza un rayo vertical
    hacia abajo (mj_ray) y, si el primer impacto es ese mesh, se marca la
    columna desde la base del AABB del mesh hasta la altura del impacto.
    Captura la SUPERFICIE SUPERIOR real (lo que importa para el heatmap de
    elevacion) de cualquier malla — rampas STL incluidas — sin point-in-mesh.

    IMPORTANTE: usar un XML SIN robot (los track_*.xml), o los rayos cerca del
    origen chocarian con el chasis.
    """
    if not meshes:
        return 0
    dims = np.array(grid.shape)
    mesh_gids = {m["gid"] for m in meshes}
    geomid = np.zeros(1, dtype=np.int32)
    marked = 0
    for m in meshes:
        corners = _mesh_world_corners(m)
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        z_top = hi[2] + 0.5                              # origen del rayo, sobre el mesh
        ix0 = max(0, int((lo[0] - world_min[0]) / resolution))
        ix1 = min(dims[0] - 1, int((hi[0] - world_min[0]) / resolution))
        iy0 = max(0, int((lo[1] - world_min[1]) / resolution))
        iy1 = min(dims[1] - 1, int((hi[1] - world_min[1]) / resolution))
        vec = np.array([0.0, 0.0, -1.0])
        for ix in range(ix0, ix1 + 1):
            x = world_min[0] + (ix + 0.5) * resolution
            for iy in range(iy0, iy1 + 1):
                y = world_min[1] + (iy + 0.5) * resolution
                pnt = np.array([x, y, z_top])
                dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
                if dist < 0 or int(geomid[0]) not in mesh_gids:
                    continue                              # sin impacto o pego en un box
                hit_z = z_top - dist
                iz0 = max(0, int((lo[2] - world_min[2]) / resolution))
                iz1 = min(dims[2] - 1, int((hit_z - world_min[2]) / resolution))
                if iz1 >= iz0:
                    grid[ix, iy, iz0:iz1 + 1] = True
                    marked += iz1 - iz0 + 1
    return marked


# ----------------------------------------------------------------------
# 2. Rasterizar boxes -> grilla booleana 3D
# ----------------------------------------------------------------------
def rasterize_boxes(boxes, resolution, margin=0.5, extra_corners=None):
    """
    Construye una grilla booleana 3D que cubre el AABB total de todos los
    boxes (+ margen), marcando True en las celdas ocupadas por algún box
    (soporta rotación arbitraria, no solo ejes alineados).

    extra_corners: puntos (N,3) adicionales a cubrir por la grilla (p.ej. las
    esquinas de los AABB de geoms mesh, que se rasterizan despues con
    rasterize_meshes_into sobre ESTA misma grilla).

    Devuelve: (grid, origin, resolution)
        grid   -> np.ndarray bool, shape (nx, ny, nz)
        origin -> (3,) esquina mínima de la grilla en coordenadas mundo
    """
    if not boxes and extra_corners is None:
        raise ValueError("No se encontraron geoms colisionables en el XML.")

    # AABB global considerando la rotación de cada box (caso general)
    mins, maxs = [], []
    for b in boxes:
        corners = _box_world_corners(b["center"], b["half_size"], b["rot"])
        mins.append(corners.min(axis=0))
        maxs.append(corners.max(axis=0))
    if extra_corners is not None and len(extra_corners):
        pts = np.asarray(extra_corners, dtype=float)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))
    world_min = np.min(mins, axis=0) - margin
    world_max = np.max(maxs, axis=0) + margin

    dims = np.maximum(1, np.ceil((world_max - world_min) / resolution).astype(int))
    grid = np.zeros(dims, dtype=bool)

    # Centros de todas las celdas de la grilla, vectorizado
    xs = world_min[0] + (np.arange(dims[0]) + 0.5) * resolution
    ys = world_min[1] + (np.arange(dims[1]) + 0.5) * resolution
    zs = world_min[2] + (np.arange(dims[2]) + 0.5) * resolution

    for b in boxes:
        # Limita el test a la sub-región de la grilla que cubre este box
        # (evita testear TODA la grilla por cada box -> mucho más rápido)
        corners = _box_world_corners(b["center"], b["half_size"], b["rot"])
        sub_min = corners.min(axis=0) - resolution
        sub_max = corners.max(axis=0) + resolution

        ix = np.where((xs >= sub_min[0]) & (xs <= sub_max[0]))[0]
        iy = np.where((ys >= sub_min[1]) & (ys <= sub_max[1]))[0]
        iz = np.where((zs >= sub_min[2]) & (zs <= sub_max[2]))[0]
        if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
            continue

        gx, gy, gz = np.meshgrid(xs[ix], ys[iy], zs[iz], indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (M,3)

        # Transforma al frame local del box: local = R^T (p - center)
        local = (pts - b["center"]) @ b["rot"]  # rot es ortonormal -> R^T = rot con @ por la izq de (p-c)
        inside = np.all(np.abs(local) <= b["half_size"] + 1e-9, axis=1)

        if np.any(inside):
            sel = pts[inside]
            # Convertir puntos mundiales seleccionados a índices de grilla
            idx = np.round((sel - world_min) / resolution - 0.5).astype(int)
            idx = np.clip(idx, 0, np.array(dims) - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return grid, world_min


def _box_world_corners(center, half_size, rot):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local_corners = signs * half_size
    world_corners = (rot @ local_corners.T).T + center
    return world_corners


# ----------------------------------------------------------------------
# 3. Grilla booleana -> OctoMap (.bt)
# ----------------------------------------------------------------------
def grid_to_octree(grid, origin, resolution):
    tree = pyoctomap.OcTree(resolution)
    occ_idx = np.argwhere(grid)
    if len(occ_idx) == 0:
        print("⚠️  Grilla vacía, no se insertó ningún voxel.")
        return tree

    points = (origin + (occ_idx + 0.5) * resolution).astype(np.float64)
    # updateNodes = versión batch de pyoctomap, sin loop de Python por punto
    # (mucho más rápido que octomap-python cuando hay decenas de miles de voxeles)
    tree.updateNodes(points, True)

    tree.updateInnerOccupancy()
    return tree


# ----------------------------------------------------------------------
# 4. Dilatación (inflado) para navegación
# ----------------------------------------------------------------------
def inflate_grid(grid, resolution, inflate_radius_m):
    """
    Dilata la grilla ocupada con una esfera de radio `inflate_radius_m`
    (radio del robot + margen de seguridad, en metros). Esto es el
    equivalente 3D al "costmap inflation layer" de nav2, hecho a mano.
    """
    radius_vox = max(1, int(round(inflate_radius_m / resolution)))

    # Elemento estructurante esférico (no un cubo, para no sobre-inflar en diagonal)
    r = radius_vox
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    sphere = (xx**2 + yy**2 + zz**2) <= r**2

    inflated = binary_dilation(grid, structure=sphere)
    return inflated


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
# XML de pista SIN robot para cada pista del registro (los rayos del raster de
# meshes no deben chocar con el chasis). Las salidas van a rl_ws/tracks/<pista>/.
_TRACK_XMLS = {
    "flat":    os.path.join(_PROJECT_ROOT, "models", "plataform.xml"),
    "steps":   os.path.join(_PROJECT_ROOT, "models", "track_steps.xml"),
    "ramps":   os.path.join(_PROJECT_ROOT, "models", "track_ramps.xml"),
    "pallets": os.path.join(_PROJECT_ROOT, "models", "track_pallets.xml"),
    "maze":    os.path.join(_PROJECT_ROOT, "models", "maze.xml"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Genera un OctoMap (.bt) desde un XML de MuJoCo, sin ROS, "
                    "y su versión dilatada para navegación."
    )
    parser.add_argument("xml_path", nargs="?", default=_DEFAULT_XML,
                        help="Ruta al XML de MuJoCo. Por defecto usa el modelo de este proyecto.")
    parser.add_argument("--track", choices=sorted(_TRACK_XMLS), default=None,
                        help="Modo pista: usa el XML de la pista (sin robot) y guarda los .bt "
                             "en rl_ws/tracks/<pista>/ (lo que espera config.TRACK_DEFS).")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Tamaño de voxel en metros (default 0.05 = 5cm)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Margen extra alrededor del AABB total (metros)")
    parser.add_argument("--inflate-radius", type=float, default=0.25,
                        help="Radio de dilatación en metros (radio del robot + colchón de seguridad)")
    parser.add_argument("--prefix", type=str, default=None,
                        help='Solo usar geoms cuyo nombre empiece con este prefijo (ej. "fatal_")')
    parser.add_argument("--out-raw", type=str, default=_DEFAULT_RAW_OUT)
    parser.add_argument("--out-inflated", type=str, default=_DEFAULT_INFLATED_OUT)
    args = parser.parse_args()

    if args.track:
        args.xml_path = _TRACK_XMLS[args.track]
        track_dir = os.path.join(_HERE, "tracks", args.track)
        os.makedirs(track_dir, exist_ok=True)
        args.out_raw = os.path.join(track_dir, "occupied_map.bt")
        args.out_inflated = os.path.join(track_dir, "inflated_map.bt")

    print(f"📥 Cargando geoms de colisión desde: {args.xml_path}")
    boxes, meshes, model, data = load_collision_boxes(args.xml_path, include_names_prefix=args.prefix)
    print(f"   -> {len(boxes)} boxes + {len(meshes)} meshes colisionables")

    print(f"🧱 Rasterizando a grilla booleana (resolución={args.resolution} m)...")
    mesh_corners = np.concatenate([_mesh_world_corners(m) for m in meshes]) if meshes else None
    grid, origin = rasterize_boxes(boxes, args.resolution, margin=args.margin,
                                   extra_corners=mesh_corners)
    if meshes:
        n_mesh_vox = rasterize_meshes_into(grid, origin, args.resolution, model, data, meshes)
        print(f"   -> meshes por raycast: +{n_mesh_vox} voxeles")
    print(f"   -> grilla shape={grid.shape}, ocupados={grid.sum()} voxeles")

    print("🌳 Construyendo OctoMap base...")
    tree = grid_to_octree(grid, origin, args.resolution)
    tree.writeBinary(args.out_raw)
    print(f"   -> guardado en: {args.out_raw}")

    print(f"🛡️  Dilatando mapa (radio={args.inflate_radius} m) para navegación...")
    inflated_grid = inflate_grid(grid, args.resolution, args.inflate_radius)
    print(f"   -> ocupados tras dilatar: {inflated_grid.sum()} voxeles "
          f"(+{inflated_grid.sum() - grid.sum()})")

    inflated_tree = grid_to_octree(inflated_grid, origin, args.resolution)
    inflated_tree.writeBinary(args.out_inflated)
    print(f"   -> guardado en: {args.out_inflated}")

    print("\n✅ Listo. Puedes visualizar los .bt con octovis, o cargarlos de vuelta con:")
    print("   tree = pyoctomap.OcTree(resolution); tree.readBinary('archivo.bt')")


if __name__ == "__main__":
    main()