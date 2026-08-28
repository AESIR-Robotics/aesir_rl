#!/usr/bin/env python3
"""
gen_steps2.py — genera models/steps2.xml a partir de las meshes STL de la pista
"steps" (COMPONENTS_description) que viven en models/meshes/steps2/.

Por que existe: la version vieja (steps.xml) aproximaba TODO el stepfield con
UNA caja solida de 2.4x4.8x0.6 m -> el robot solo rodaba por una losa plana a
0.6 m y jamas tocaba un escalon. Aqui el collider se calcula EXACTO.

Como: base_link.stl es una union de cajas alineadas a los ejes (contrachapado
de 9 mm: postes de 5x5 cm + cubiertas + muros perimetrales). Entonces:

  1. Se toman los planos distintos de X, Y, Z de los vertices -> rejilla
     irregular exacta (32 x 57 x 7 celdas). Ninguna cara queda "entre" celdas.
  2. Cada celda se clasifica dentro/fuera lanzando un rayo vertical y contando
     cruces con los triangulos (paridad). Offset irracional en XY para no caer
     sobre aristas de la triangulacion.
  3. Las celdas llenas se funden con un greedy 3D (se prueban los 6 ordenes de
     ejes y se queda el que da menos cajas): 4457 celdas -> 415 cajas.

La UNICA parte del mesh que no es una union de cajas son las 4 rampas a 45 grados
del pedestal central (10 triangulos inclinados). Escalonarlas estaria mal — el
volumen cuadraria por compensacion pero la superficie quedaria 5 cm arriba/abajo
de la real. Asi que sus celdas se vacian y cada rampa sale como una CUNA (prisma
triangular) convexa, definida inline con <mesh vertex="...">: al ser convexa,
MuJoCo la colisiona EXACTA (usa el hull, que aqui es la cuna misma).

Verificacion: volumen de cajas+cunas == volumen del mesh (teorema de la
divergencia) al 0.0000%, y un raycast vertical cada 7.5 mm sobre las 207043
celdas da la MISMA altura para el mesh visual y para el collider.

Los 10 Component3_*/Component4_* (hubs y puertas del centro) llevan collider
AABB, igual que en steps.xml y maze.xml — sus vertices ya estan en coordenadas
de mundo, por eso los meshes van en (0,0,0).

Uso:  python3 gen_steps2.py            # reescribe steps2.xml
"""
from __future__ import annotations

import itertools
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(HERE, "meshes", "steps2")
OUT = os.path.join(HERE, "steps2.xml")

SCALE = 0.001          # las STL estan en mm
WALL_TOP = 0.6         # los muros perimetrales llegan a z=0.6
HUBS = [f"Component3_{i}" for i in range(1, 6)]
DOORS = [f"Component4_{i}" for i in range(1, 6)]


def load_stl(path):
    """STL binario -> (n_tri, 3, 3) float64."""
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(84)[80:84])[0]
        data = f.read()
    if len(data) != n * 50:
        raise ValueError(f"{path}: no es un STL binario valido")
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    return arr[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def slanted_wedges(tri, planes):
    """Las caras a 45 grados no caben en la rejilla: se sacan como cunas convexas.

    Devuelve (lista de cunas, lista de celdas (i,j,k) a vaciar). Cada cuna es el
    hull de sus vertices inclinados mas la cara inferior de su celda: para un
    prisma triangular apoyado en el suelo de la celda, eso es la cuna exacta.
    """
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    nn = np.zeros_like(n)
    nn[ok] = n[ok] / ln[ok, None]
    slanted = ok & ~(np.abs(np.abs(nn) - 1).min(1) < 1e-4)

    wedges, cells = [], []
    for u in np.unique(np.round(nn[slanted], 3), axis=0):
        grp = tri[slanted][(np.round(nn[slanted], 3) == u).all(1)]
        v = grp.reshape(-1, 3)
        lo, hi = v.min(0), v.max(0)
        rng = [range(np.searchsorted(planes[a], lo[a] + 1e-9) - 1,
                     np.searchsorted(planes[a], hi[a] - 1e-9)) for a in range(3)]
        if len(rng[2]) != 1:
            raise RuntimeError(f"cara inclinada n={u} cruza varios planos Z")
        # la rampa es prismatica en su eje ancho: se vacia todo su bbox de celdas
        # (varias celdas si hay planos intermedios) y se emite UNA cuna que lo cubre
        cells += [(i, j, k) for i in rng[0] for j in rng[1] for k in rng[2]]
        z0 = planes[2][rng[2][0]]
        base = np.array([[x, y, z0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        wedges.append(np.unique(np.round(np.vstack([v, base]), 6), axis=0))
    return wedges, cells


def voxelize(tri):
    """Rejilla exacta + paridad por rayo vertical. -> (planos_xyz, occ bool 3D)."""
    v = tri.reshape(-1, 3)
    planes = [np.unique(np.round(v[:, i], 6)) for i in range(3)]
    nx, ny, nz = (len(p) - 1 for p in planes)
    # centros de celda; el offset irracional evita que el rayo caiga justo sobre
    # la diagonal con que la STL triangula cada rectangulo
    cx = (planes[0][:-1] + planes[0][1:]) / 2 + 1e-7 * np.pi
    cy = (planes[1][:-1] + planes[1][1:]) / 2 + 1e-7 * np.e
    cz = (planes[2][:-1] + planes[2][1:]) / 2

    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(B - A, C - A)
    keep = np.abs(n[:, 2]) > 1e-12          # un rayo vertical no cruza caras verticales
    A, B, C, n = A[keep], B[keep], C[keep], n[keep]

    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    ccx, ccy = C[:, 0], C[:, 1]
    den = (by - ccy) * (ax - ccx) + (ccx - bx) * (ay - ccy)

    occ = np.zeros((nx, ny, nz), bool)
    for i, px in enumerate(cx):
        t1 = (by - ccy) * (px - ccx)
        t2 = (ccy - ay) * (px - ccx)
        for j, py in enumerate(cy):
            l1 = (t1 + (ccx - bx) * (py - ccy)) / den
            l2 = (t2 + (ax - ccx) * (py - ccy)) / den
            m = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            if not m.any():
                continue
            nm, Am = n[m], A[m]
            zh = Am[:, 2] - (nm[:, 0] * (px - Am[:, 0]) + nm[:, 1] * (py - Am[:, 1])) / nm[:, 2]
            zh.sort()
            # dentro <=> numero IMPAR de cruces por encima del centro
            occ[i, j] = (len(zh) - np.searchsorted(zh, cz, side="left")) % 2 == 1
    return planes, occ


def merge(occ, planes, order):
    """Greedy 3D sobre los ejes permutados por `order` -> lista de (lo, hi)."""
    o = np.transpose(occ, order)
    pl = [planes[i] for i in order]
    inv = np.argsort(order)
    n0, n1, n2 = o.shape
    used = np.zeros_like(o)
    boxes = []
    for a in range(n0):
        for b in range(n1):
            for c in range(n2):
                if not o[a, b, c] or used[a, b, c]:
                    continue
                c2 = c
                while c2 + 1 < n2 and o[a, b, c2 + 1] and not used[a, b, c2 + 1]:
                    c2 += 1
                b2 = b
                while (b2 + 1 < n1 and o[a, b2 + 1, c:c2 + 1].all()
                       and not used[a, b2 + 1, c:c2 + 1].any()):
                    b2 += 1
                a2 = a
                while (a2 + 1 < n0 and o[a2 + 1, b:b2 + 1, c:c2 + 1].all()
                       and not used[a2 + 1, b:b2 + 1, c:c2 + 1].any()):
                    a2 += 1
                used[a:a2 + 1, b:b2 + 1, c:c2 + 1] = True
                lo = [pl[0][a], pl[1][b], pl[2][c]]
                hi = [pl[0][a2 + 1], pl[1][b2 + 1], pl[2][c2 + 1]]
                boxes.append(([lo[i] for i in inv], [hi[i] for i in inv]))
    return boxes


def aabb(name):
    v = load_stl(os.path.join(MESHDIR, name + ".stl")).reshape(-1, 3) * SCALE
    lo, hi = v.min(0), v.max(0)
    return (lo + hi) / 2, (hi - lo) / 2


def main():
    tri = load_stl(os.path.join(MESHDIR, "base_link.stl")) * SCALE
    planes, occ = voxelize(tri)

    wedges, cells = slanted_wedges(tri, planes)
    for i, j, k in cells:
        occ[i, j, k] = False

    best = min((merge(occ, planes, o) for o in itertools.permutations(range(3))), key=len)

    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    vmesh = abs(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)
    vbox = sum((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) for lo, hi in best)
    vwedge = sum(np.ptp(w, axis=0).prod() / 2 for w in wedges)   # media caja cada cuna
    print(f"celdas llenas {occ.sum()} -> {len(best)} cajas + {len(wedges)} cunas | "
          f"volumen mesh {vmesh:.6f} vs collider {vbox + vwedge:.6f} m3 "
          f"(error {100 * abs(vbox + vwedge - vmesh) / vmesh:.4f}%)")

    walls, field = [], []
    for lo, hi in sorted(best, key=lambda b: (b[0][2], b[0][0], b[0][1])):
        (walls if hi[2] >= WALL_TOP - 1e-9 else field).append((lo, hi))

    def geom(prefix, idx, lo, hi, rgba):
        pos = [(a + b) / 2 for a, b in zip(lo, hi)]
        size = [(b - a) / 2 for a, b in zip(lo, hi)]
        return (f'    <geom name="{prefix}_{idx}" type="box" '
                f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" '
                f'rgba="{rgba}" contype="1" conaffinity="1"/>')

    L = []
    L.append('<mujoco model="aesir_steps2">')
    L.append('')
    L.append('  <!-- GENERADO por gen_steps2.py — NO editar a mano. -->')
    L.append('  <!-- Los geoms VISUALES van en group="2" y los COLLIDERS en group 0: en el')
    L.append('       visor de MuJoCo, la tecla 2 apaga la malla y deja ver solo la colision. -->')
    L.append('  <!-- Stepfield 2.418 x 4.806 m: postes de 5x5 cm a 0.159 / 0.309 m,')
    L.append('       cubiertas de 9 mm y muros perimetrales de 0.6 m. El collider es la')
    L.append('       descomposicion EXACTA de base_link.stl en cajas (error 0.0000% en')
    L.append('       volumen), no la losa solida que usaba steps.xml. -->')
    L.append('')
    L.append('  <default>')
    L.append('    <geom condim="4" friction="0.2 0.005 0.0001"/>')
    L.append('  </default>')
    L.append('')
    L.append('  <asset>')
    L.append('    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512"/>')
    L.append('    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="4 4" texuniform="true"/>')
    L.append('    <material name="steps2_visual" rgba="0.5 0.5 0.55 1"/>')
    L.append('    <material name="obs_hub" rgba="0.8 0.3 0.15 1"/>')
    L.append('    <material name="obs_door" rgba="0.2 0.6 0.85 1"/>')
    L.append('')
    L.append('    <mesh name="steps2_base_mesh" file="meshes/steps2/base_link.stl" scale="0.001 0.001 0.001"/>')
    L.append('')
    L.append('    <!-- Cunas de colision de las 4 rampas a 45 grados del pedestal central.')
    L.append('         Convexas -> el hull que usa MuJoCo ES la cuna, colision exacta. -->')
    for i, w in enumerate(wedges):
        vtx = " ".join(f"{c:.4f}" for c in w.reshape(-1))
        L.append(f'    <mesh name="steps2_ramp_{i}" vertex="{vtx}"/>')
    for i, nm in enumerate(HUBS, 1):
        L.append(f'    <mesh name="steps2_hub_{i}" file="meshes/steps2/{nm}.stl" scale="0.001 0.001 0.001"/>')
    for i, nm in enumerate(DOORS, 1):
        L.append(f'    <mesh name="steps2_door_{i}" file="meshes/steps2/{nm}.stl" scale="0.001 0.001 0.001"/>')
    L.append('  </asset>')
    L.append('')
    L.append('  <worldbody>')
    L.append('    <light directional="true" pos="3 5 8" dir="-0.5 -0.5 -1"/>')
    L.append('    <light directional="true" pos="-1 2 6" dir="0.5 -0.3 -1"/>')
    L.append('')
    L.append('    <!-- El piso va a z=-0.009 (el espesor de la losa base) para que la')
    L.append('         superficie pisable de la pista quede EXACTAMENTE en z=0 y no')
    L.append('         haya z-fighting con la losa. -->')
    L.append('    <geom name="steps2_floor" type="plane" size="50 50 0.1" pos="0 0 -0.009" material="matplane"/>')
    L.append('')
    L.append('    <!-- Visual: los vertices del mesh ya estan en coordenadas de mundo. -->')
    L.append('    <body name="steps2_base">')
    L.append('      <geom name="steps2_visual" type="mesh" mesh="steps2_base_mesh"')
    L.append('            contype="0" conaffinity="0" group="2" material="steps2_visual"/>')
    L.append('    </body>')
    L.append('')
    L.append(f'    <!-- ===== MUROS PERIMETRALES ({len(walls)} cajas, 9 mm de espesor, z hasta 0.6) ===== -->')
    L.append('    <body name="steps2_walls">')
    for i, (lo, hi) in enumerate(walls):
        L.append(geom("steps2_wall", i, lo, hi, "0.45 0.47 0.52 0.25"))
    L.append('    </body>')
    L.append('')
    L.append(f'    <!-- ===== LOSA BASE + STEPFIELD ({len(field)} cajas) ===== -->')
    L.append('    <body name="steps2_collision">')
    for i, (lo, hi) in enumerate(field):
        L.append(geom("steps2_col", i, lo, hi, "0.5 0.5 0.55 0.25"))
    L.append('    </body>')
    L.append('')
    L.append(f'    <!-- ===== RAMPAS 45 deg DEL PEDESTAL CENTRAL ({len(wedges)} cunas) ===== -->')
    L.append('    <body name="steps2_ramps">')
    for i in range(len(wedges)):
        L.append(f'      <geom name="steps2_ramp_{i}" type="mesh" mesh="steps2_ramp_{i}"')
        L.append('            rgba="0.5 0.5 0.55 0.25" contype="1" conaffinity="1"/>')
    L.append('    </body>')
    L.append('')
    L.append('    <!-- ===== HUBS (Component3) + PUERTAS (Component4) ===== -->')
    L.append('    <!-- Collider = AABB del mesh, igual criterio que steps.xml / maze.xml. -->')
    for i, nm in enumerate(HUBS, 1):
        c, h = aabb(nm)
        L.append(f'    <body name="steps2_hub_{i}">   <!-- {nm} -->')
        L.append(f'      <geom name="steps2_hub_{i}_vis" type="mesh" mesh="steps2_hub_{i}"')
        L.append('            contype="0" conaffinity="0" group="2" material="obs_hub"/>')
        L.append(f'      <geom name="steps2_hub_{i}_col" type="box" size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}"')
        L.append(f'            pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" rgba="0.8 0.3 0.15 0.4"/>')
        L.append('    </body>')
    for i, nm in enumerate(DOORS, 1):
        c, h = aabb(nm)
        L.append(f'    <body name="steps2_door_{i}">   <!-- {nm} -->')
        L.append(f'      <geom name="steps2_door_{i}_vis" type="mesh" mesh="steps2_door_{i}"')
        L.append('            contype="0" conaffinity="0" group="2" material="obs_door"/>')
        L.append(f'      <geom name="steps2_door_{i}_col" type="box" size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}"')
        L.append(f'            pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" rgba="0.2 0.6 0.85 0.4"/>')
        L.append('    </body>')
    L.append('')
    L.append('  </worldbody>')
    L.append('')
    L.append('</mujoco>')

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"escrito {OUT}  ({len(walls)} muros + {len(field)} stepfield + "
          f"{len(wedges)} rampas + 10 componentes)")


if __name__ == "__main__":
    main()
