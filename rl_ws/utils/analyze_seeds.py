#!/usr/bin/env python3
"""
analyze_seeds.py -- Tabla y curvas de un barrido de semillas, desde los .log.

Lee la salida de train_fast.py (la que guarda `tee`) y produce lo que hace falta
para un paper: media +- desviacion ENTRE SEMILLAS, no de una sola corrida.

Por que la media de las ULTIMAS K iteraciones y no el valor final: la politica
NO converge a un punto, ORBITA uno. Medido sobre 20 checkpoints (iter 50-1000):
recorre un camino de ~20.5 para acabar a 3.0 del origen (85% de cancelacion), y
el success oscila ~10 puntos entre iteraciones consecutivas. Reportar el ultimo
valor seria reportar DONDE CAYO la orbita, no lo que aprendio el metodo.
Promediar una ventana final es robusto a eso y es practica estandar.

Uso:
    python3 rl_ws/utils/analyze_seeds.py runs/seed1.log runs/seed2.log runs/seed3.log
    python3 rl_ws/utils/analyze_seeds.py runs/*.log --last 50 --csv out.csv --fig curva.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# [Iter  249] success=39.0% (100ep)  avg_ret= 7509.05  reached=2292 ...
RE_ITER = re.compile(r"\[Iter\s+(\d+)\]\s+success=\s*([\d.]+)%.*?avg_ret=\s*(-?[\d.]+)")
#            por pista: flat=82%  ramps=73%  steps1m=37%  pallets=3%
RE_TRACK = re.compile(r"(\w+)=(\d+)%")


def parse(path: Path):
    """-> (iters, success_global, avg_ret, {pista: [success por iter]})

    Indexa por NUMERO DE ITERACION y se queda con la ULTIMA aparicion de cada
    una. Es lo correcto cuando el log tiene un arranque abortado o un `tee -a`
    tras reanudar: si no, las repetidas se cuentan dos veces y el recorte a
    longitud comun acaba descartando iteraciones reales del final.
    (Pasa de verdad: seed1.log traia 0,1,2 duplicadas -> 503 lineas, 500 iters.)
    """
    reg = {}          # iter -> [success, avg_ret, {pista: success}]
    cur = None
    for line in path.read_text(errors="ignore").splitlines():
        m = RE_ITER.search(line)
        if m:
            cur = int(m.group(1))
            reg[cur] = [float(m.group(2)) / 100.0, float(m.group(3)), {}]
            continue
        if cur is not None and "por pista:" in line:
            reg[cur][2] = {t: float(v) / 100.0
                           for t, v in RE_TRACK.findall(line.split("por pista:")[1])}
            cur = None
    if not reg:
        return np.array([]), np.array([]), np.array([]), {}
    its = np.array(sorted(reg))
    sr = np.array([reg[i][0] for i in its])
    ret = np.array([reg[i][1] for i in its])
    tracks = sorted({t for i in its for t in reg[i][2]})
    per = {t: np.array([reg[i][2].get(t, np.nan) for i in its]) for t in tracks}
    return its, sr, ret, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--last", type=int, default=50,
                    help="iteraciones finales sobre las que promediar (default 50)")
    ap.add_argument("--csv", type=Path, default=None, help="volcar las curvas")
    ap.add_argument("--fig", type=Path, default=None, help="figura de curvas de aprendizaje")
    a = ap.parse_args()

    raw = []
    for p in a.logs:
        its, sr, ret, per = parse(p)
        if len(its) == 0:
            print(f"  AVISO: {p} no tiene lineas [Iter ...], se omite")
            continue
        raw.append((p.stem, its, sr, ret, per))
        print(f"  {p.stem}: {len(its)} iteraciones ({its[0]}-{its[-1]})")
    if not raw:
        raise SystemExit("Ningun log valido.")

    # Alinear por NUMERO DE ITERACION, no por posicion. Si un log empieza en otra
    # iteracion (p.ej. una reanudacion cuyo `tee` sin -a se comio el principio),
    # recortar por posicion compararia iteraciones DISTINTAS entre semillas y
    # daria numeros sin sentido sin avisar.
    common = sorted(set(raw[0][1]).intersection(*(set(r[1]) for r in raw[1:])))
    if not common:
        raise SystemExit("Los logs no comparten ninguna iteracion.")
    if any(len(r[1]) != len(common) for r in raw):
        print(f"\n  AVISO: los logs no cubren el mismo rango. Se usa la "
              f"INTERSECCION: iteraciones {common[0]}-{common[-1]} ({len(common)}).")
    runs = []
    for name, its, sr, ret, per in raw:
        idx = np.searchsorted(its, common)
        runs.append((name, np.array(common), sr[idx], ret[idx],
                     {t: v[idx] for t, v in per.items()}))

    n = len(common)
    K = min(a.last, n)
    tracks = sorted({t for r in runs for t in r[4]})

    print(f"\n{'='*66}")
    print(f"RESULTADOS  (media de las ultimas {K} iteraciones, +- entre semillas)")
    print(f"{'='*66}")
    print(f"{'':<12} " + " ".join(f"{r[0]:>12}" for r in runs) + f" | {'media':>14}")
    print("-" * 66)

    def fila(nombre, vals):
        m, s = np.mean(vals), np.std(vals)
        print(f"{nombre:<12} " + " ".join(f"{v:>11.1%}" for v in vals)
              + f" | {m:>7.1%} +-{s:.1%}")
        return m, s

    fila("global", [np.mean(r[2][:n][-K:]) for r in runs])
    for t in tracks:
        vals = [np.mean(r[4][t][:n][-K:]) for r in runs if t in r[4]]
        if len(vals) == len(runs):
            fila(t, vals)
    rr = [np.mean(r[3][:n][-K:]) for r in runs]
    print(f"{'avg_ret':<12} " + " ".join(f"{v:>11.0f}" for v in rr)
          + f" | {np.mean(rr):>7.0f} +-{np.std(rr):.0f}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "run", "success"] + tracks)
            for name, its, sr, _ret, per in runs:
                for i in range(n):
                    w.writerow([its[i], name, f"{sr[i]:.4f}"]
                               + [f"{per[t][i]:.4f}" if t in per else "" for t in tracks])
        print(f"\n  curvas -> {a.csv}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        S = np.stack([r[2][:n] for r in runs])           # (semillas, iters)
        x = runs[0][1][:n]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x, S.mean(0), lw=2, color="C0")
        ax[0].fill_between(x, S.mean(0) - S.std(0), S.mean(0) + S.std(0),
                           alpha=.25, color="C0")
        ax[0].set(xlabel="iteracion", ylabel="success rate",
                  title=f"Global (media +- desv., {len(runs)} semillas)")
        ax[0].grid(alpha=.3)
        for i, t in enumerate(tracks):
            P = np.stack([r[4][t][:n] for r in runs if t in r[4]])
            ax[1].plot(x, P.mean(0), lw=1.6, color=f"C{i}", label=t)
            ax[1].fill_between(x, P.mean(0) - P.std(0), P.mean(0) + P.std(0),
                               alpha=.18, color=f"C{i}")
        ax[1].set(xlabel="iteracion", title="Por pista")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.fig, dpi=160)
        print(f"  figura -> {a.fig}")


if __name__ == "__main__":
    main()
