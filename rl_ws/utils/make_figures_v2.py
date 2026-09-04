#!/usr/bin/env python3
"""
make_figures_v2.py -- Figuras del paper SII sobre la tanda NUEVA (post-lattice).

Se separa de make_figures.py a proposito: aquel lee la tanda de agosto
(`seed?.log`, `e8_seed?.log`, ...) y esta lee la tanda post-lattice
(`e1p_seed?.log`), que es otro sistema. Mezclarlas daria figuras que promedian
condiciones no comparables.

NO hay figuras que contrasten con y sin flippers: que un robot oruga necesite
flippers para subir escalones no es una pregunta de este trabajo, y una figura
organizada alrededor de esa comparacion le daria el protagonismo a lo unico
obvio que hay medido. La unica comparacion que queda es la del gate (figE),
que si es un hallazgo.

Misma paleta y mismo estilo que make_figures.py para que las figuras del paper
se lean como una sola familia.

Uso:
    python3 rl_ws/utils/make_figures_v2.py
    python3 rl_ws/utils/make_figures_v2.py --runs runs/paper_archive --out figs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze_seeds import parse

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIV_NEG, NEUTRAL = "#e34948", "#f0efec"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"

TRACKS = ["flat", "ramps", "steps1m", "pallets"]
WIN = (250, 300)   # ventana de reporte: la meseta del sistema con lattice
# Umbral de "corrida muerta". No es un juicio: la separacion es bimodal y de 33
# puntos (convergidas 41.9-53.1%, muertas 6.4-9.2%), sin nada en medio.
DEAD_T = 0.15

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK3, "axes.linewidth": .8,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.facecolor": "white",
})


def style(ax, pct=True):
    ax.grid(True, color=INK3, alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def smooth(y, w=15):
    return np.convolve(y, np.ones(w) / w, mode="valid")


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.png")


def load(runs: Path, pat: str):
    R = [(p.stem, *parse(p)) for p in sorted(runs.glob(pat))]
    R = [r for r in R if len(r[1])]
    if not R:
        raise SystemExit(f"sin logs para {pat} en {runs}")
    return R


def win_mean(its, y, lo=WIN[0], hi=WIN[1]):
    m = (its >= lo) & (its < hi)
    return float(np.nanmean(y[m]))


# ── fig A: curvas por semilla del sistema completo ───────────────────────────
def figA(E1, out):
    """Curvas por semilla del sistema completo. Las que nunca despegaron SE VEN:
    la bimodalidad del entrenamiento es el resultado, no un defecto que esconder
    promediando. No se contrasta contra la condicion sin flippers: que un robot
    oruga necesite flippers no es una pregunta de este trabajo."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    n_dead = first_conv = first_dead = 0
    for name, its, sr, *_ in E1:
        x, y = its[14:], smooth(sr)
        dead = win_mean(its, sr) < DEAD_T
        n_dead += dead
        lab = None
        if dead and not first_dead:
            lab, first_dead = "Never left the initial policy", 1
        elif not dead and not first_conv:
            lab, first_conv = "Converged", 1
        ax.plot(x, y, color=DIV_NEG if dead else CAT[0], lw=1.2 if dead else 1.8,
                ls=(0, (3, 2)) if dead else "-", alpha=.85 if dead else 1, label=lab)
    ax.axvspan(WIN[0], WIN[1] - 1, color=INK3, alpha=.07, lw=0)
    ax.text(WIN[0] + 3, .015, "reporting window", fontsize=7, color=INK2, style="italic")
    ax.text(.02, .95, f"{n_dead} of {len(E1)} runs never left the initial policy",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Success rate")
    ax.set_title("Per-seed learning curves", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(.02, .62)); style(ax)
    save(fig, out, "v2_fig1_learning_curves")


def figB_arenas(out, evaldir=Path("runs/eval"), pat="all_e1p_s*.json"):
    """Rendimiento del sistema POR ARENA, entrenadas y retenidas en la misma
    figura y con la MISMA medicion (evaluador offline, 50 episodios, criterio de
    exito identico al del entrenamiento).

    No compara con/sin flippers: que un robot oruga necesite flippers para subir
    escalones no es una pregunta de este paper, y organizar la figura principal
    alrededor de esa comparacion le da el protagonismo a lo unico obvio que hay
    medido. La condicion sin flippers aparece solo donde es un argumento -- la
    transferencia -- y alli basta con los numeros."""
    import json
    files = sorted(evaldir.glob(pat))
    if not files:
        print(f"  (figura de arenas omitida: no hay {pat} en {evaldir})")
        return
    data = [json.load(open(f))["tracks"] for f in files]
    TRAIN = ["flat", "ramps", "steps1m", "pallets"]
    HELD = ["steps", "steps2"]
    order = [t for t in TRAIN + HELD if any(t in d for d in data)]
    M = np.array([[d[t]["success"] if t in d else np.nan for t in order] for d in data])

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    xs = np.arange(len(order))
    cols = [CAT[0] if t in TRAIN else CAT[2] for t in order]
    ax.bar(xs, np.nanmean(M, 0), .58, color=cols, edgecolor="white", lw=.6, zorder=2)
    for j in range(len(order)):
        ax.scatter(np.full(M.shape[0], xs[j]), M[:, j], s=13, color=INK,
                   alpha=.75, zorder=4, linewidths=0)
    n_tr = sum(t in TRAIN for t in order)
    if 0 < n_tr < len(order):
        ax.axvline(n_tr - .5, color=INK3, lw=.9, ls=(0, (4, 3)))
        ax.text(n_tr - .42, .96, "held out", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=INK2, style="italic", va="top")
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_ylabel("Success rate")
    ax.set_title(f"System performance per arena ({M.shape[0]} converged seeds, "
                 f"50 episodes each)", loc="left")
    style(ax)
    save(fig, out, "v2_fig2_arenas")


def figD(runs, E1, out):
    """C3: la entropia colapsa ANTES de que el exito deje de subir.
    Sin ejes duales -- dos paneles compartiendo x."""
    ent = []
    import re
    for p in sorted(runs.glob("e1p_seed?.log")):
        v = [float(m) for m in re.findall(r"ent=(-?[\d.]+)", p.read_text(errors="ignore"))]
        ent.append(np.array(v))
    n = min(min(len(e) for e in ent), min(len(r[1]) for r in E1))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": .18})
    x = E1[0][1][:n][14:]
    S = np.stack([smooth(r[2][:n]) for r in E1])
    a1.fill_between(x, S.min(0), S.max(0), color=CAT[0], alpha=.16, lw=0)
    a1.plot(x, S.mean(0), color=CAT[0], lw=1.8)
    a1.set_ylabel("Success rate"); style(a1)
    E = np.stack([smooth(e[:n]) for e in ent])
    a2.fill_between(x, E.min(0), E.max(0), color=CAT[2], alpha=.16, lw=0)
    a2.plot(x, E.mean(0), color=CAT[2], lw=1.8)
    a2.axhline(0, color=INK3, lw=.7, ls=":")
    a2.set_ylabel("Policy entropy"); a2.set_xlabel("PPO iteration"); style(a2, pct=False)
    a1.set_title("Entropy collapses while success is still climbing", loc="left")
    save(fig, out, "v2_fig4_entropy_vs_success")



# ── fig 3: curvas de aprendizaje POR ARENA ───────────────────────────────────
def figC_per_arena(E1, out):
    """Una curva por arena, media +- 1 desv sobre las semillas CONVERGIDAS.

    Solo convergidas: incluir las dos que nunca despegaron ensancharia las bandas
    hasta ocultar la forma de cada curva, y la bimodalidad ya es el asunto de la
    fig. 1. Aqui la pregunta es otra -- a que ritmo y hasta donde aprende cada
    arena -- y para eso la mezcla de dos modos solo estorba.

    Etiquetado directo sobre la curva: la identidad de la arena no debe depender
    de cruzar la vista al recuadro de leyenda."""
    conv = [r for r in E1 if win_mean(r[1], r[2]) >= DEAD_T]
    n = min(len(r[1]) for r in conv)
    x = conv[0][1][:n][14:]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for t, c in zip(TRACKS, CAT):
        M = np.stack([smooth(np.nan_to_num(r[4][t][:n])) for r in conv if t in r[4]])
        m, sd = M.mean(0), M.std(0)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=.15, lw=0)
        ax.plot(x, m, color=c, lw=1.8, solid_capstyle="round")
        ax.annotate(t, (x[-1], m[-1]), xytext=(5, 0), textcoords="offset points",
                    color=c, fontsize=8, va="center", fontweight="bold")
    ax.axvspan(WIN[0], WIN[1] - 1, color=INK3, alpha=.07, lw=0)
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Success rate")
    ax.set_xlim(x[0], x[-1] * 1.13); ax.set_ylim(0, 1.02)
    ax.set_title(f"Per-arena learning ({len(conv)} converged seeds, "
                 f"shaded $\\pm$1 s.d.)", loc="left")
    style(ax)
    save(fig, out, "v2_fig3_per_arena_curves")


# ── fig E: el gate, sobre la tanda historica ─────────────────────────────────
def figE(old: Path, out):
    """E1 (con gate) vs E8 (sin gate), 3 semillas cada una. Sistema de agosto."""
    G = load(old, "seed?.log")
    N = load(old, "e8_seed?.log")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    xs, w = np.arange(len(TRACKS)), .36
    for R, c, lab, off in [(G, CAT[3], "With discrete gate", -w/2),
                           (N, CAT[0], "No gate (adopted)", +w/2)]:
        M = np.array([[win_mean(its, per[t], 200, 250) if t in per else np.nan
                       for t in TRACKS] for _, its, sr, ret, per in R])
        ax.bar(xs + off, np.nanmean(M, 0), w, yerr=np.nanstd(M, 0), capsize=2.5,
               color=c, label=lab, edgecolor="white", lw=.6,
               error_kw=dict(lw=.9, ecolor=INK2), zorder=2)
    ax.annotate("+18.0 pp\np = 0.012", xy=(2, .42), fontsize=7.5,
                color=INK2, ha="center", style="italic")
    ax.set_xticks(xs); ax.set_xticklabels(TRACKS)
    ax.set_ylabel("Success rate")
    ax.set_title("A discrete on/off gate for the flippers costs step traversal", loc="left")
    ax.legend(loc="upper right"); style(ax)
    save(fig, out, "v2_fig5_gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/paper_archive"))
    ap.add_argument("--old", type=Path, default=Path("runs/experiments"))
    ap.add_argument("--out", type=Path, default=Path("figs"))
    a = ap.parse_args()

    E1 = load(a.runs, "e1p_seed?.log")
    print(f"semillas -> {[r[0] for r in E1]}")
    figA(E1, a.out)
    figB_arenas(a.out)
    figC_per_arena(E1, a.out)
    figD(a.runs, E1, a.out)
    try:
        figE(a.old, a.out)
    except SystemExit as e:
        print(f"  (gate omitido: {e})")


if __name__ == "__main__":
    main()
