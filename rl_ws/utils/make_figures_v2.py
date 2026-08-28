#!/usr/bin/env python3
"""
make_figures_v2.py -- Figuras del paper SII sobre la tanda NUEVA (post-lattice).

Se separa de make_figures.py a proposito: aquel lee la tanda de agosto
(`seed?.log`, `e8_seed?.log`, ...) y esta lee E1'/E4' (`e1p_seed?.log`,
`e4p_seed?.log`), que son otro sistema. Mezclarlas en un script daria figuras
que promedian condiciones no comparables.

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
WIN = (250, 300)          # ventana de reporte: la meseta del sistema nuevo

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


# ── fig A: curvas por semilla, E1' vs E4' ────────────────────────────────────
def figA(E1, E4, out):
    """Curvas individuales. La semilla muerta de E1' SE VE: es un resultado,
    no un defecto que esconder promediando."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, (name, its, sr, *_ ) in enumerate(E1):
        x, y = its[14:], smooth(sr)
        dead = win_mean(its, sr) < .15
        ax.plot(x, y, color=CAT[0], lw=1.8 if not dead else 1.2,
                ls="-" if not dead else (0, (3, 2)), alpha=1 if not dead else .75,
                label="Flipper control" if i == 0 else None)
        if dead:
            j = len(x) // 2
            ax.annotate("1 of 3 seeds never took off", xy=(x[j], y[j]),
                        xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=7.5, color=DIV_NEG, style="italic")
    for i, (name, its, sr, *_) in enumerate(E4):
        ax.plot(its[14:], smooth(sr), color=CAT[1], lw=1.6,
                label="Flippers parked" if i == 0 else None)
    ax.axvspan(WIN[0], WIN[1] - 1, color=INK3, alpha=.07, lw=0)
    ax.text(WIN[0] + 2, .02, "reporting window", fontsize=7, color=INK2, style="italic")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Success rate")
    ax.set_title("Per-seed learning curves", loc="left")
    ax.legend(loc="upper left"); style(ax)
    save(fig, out, "v2_fig1_learning_curves")


# ── fig B: barras por pista con puntos de semilla ────────────────────────────
def figB(E1, E4, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    w, xs = .36, np.arange(len(TRACKS))
    for k, (R, c, lab, off) in enumerate([(E1, CAT[0], "Flipper control", -w/2),
                                          (E4, CAT[1], "Flippers parked", +w/2)]):
        M = np.array([[win_mean(its, per[t]) if t in per else np.nan
                       for t in TRACKS] for _, its, sr, ret, per in R])
        ax.bar(xs + off, np.nanmean(M, 0), w, color=c, label=lab,
               edgecolor="white", lw=.6, zorder=2)
        for j in range(len(TRACKS)):
            ax.scatter(np.full(M.shape[0], xs[j] + off), M[:, j], s=11,
                       color=INK, alpha=.7, zorder=3, linewidths=0)
    ax.set_xticks(xs); ax.set_xticklabels(TRACKS)
    ax.set_ylabel("Success rate")
    ax.set_title(f"Per-track success, iterations {WIN[0]}–{WIN[1]-1}", loc="left")
    ax.legend(loc="upper right"); style(ax)
    save(fig, out, "v2_fig2_per_track")


# ── fig C: steps1m, el efecto mas nitido ─────────────────────────────────────
def figC(E1, E4, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for R, c, lab in [(E1, CAT[0], "Flipper control"), (E4, CAT[1], "Flippers parked")]:
        for i, (_, its, sr, ret, per) in enumerate(R):
            if "steps1m" not in per:
                continue
            y = np.nan_to_num(per["steps1m"])
            ax.plot(its[14:], smooth(y), color=c, lw=1.6,
                    alpha=.9, label=lab if i == 0 else None)
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Success rate — steps1m")
    ax.set_title("Discrete steps: where the flippers pay for themselves", loc="left")
    ax.legend(loc="upper left"); style(ax)
    save(fig, out, "v2_fig3_steps1m")


# ── fig D: entropia vs exito, paneles separados ──────────────────────────────
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
    E4 = load(a.runs, "e4p_seed?.log")
    print(f"E1' -> {[r[0] for r in E1]}")
    print(f"E4' -> {[r[0] for r in E4]}")
    figA(E1, E4, a.out)
    figB(E1, E4, a.out)
    figC(E1, E4, a.out)
    figD(a.runs, E1, a.out)
    try:
        figE(a.old, a.out)
    except SystemExit as e:
        print(f"  (gate omitido: {e})")


if __name__ == "__main__":
    main()
