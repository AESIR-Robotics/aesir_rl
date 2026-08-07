#!/usr/bin/env python3
"""
make_figures.py -- Las nueve figuras del paper, a partir de los .log de runs/.

Todo en INGLES (destino: paper IEEE SII). Un fichero PDF + PNG por figura.
PDF porque LaTeX lo incrusta como vectorial; PNG para pegar en borradores.

Diseño (paleta de referencia validada, orden fijo de slots -- ver la guia de
dataviz): categorico slots 1-4 = azul / naranja / aqua / amarillo; divergente
azul<->rojo con gris neutro para los deltas, que son datos con POLARIDAD.
Sin ejes duales en ningun sitio: cuando hay dos magnitudes distintas van en
paneles separados compartiendo el eje x.

Uso:
    python3 rl_ws/utils/make_figures.py            # -> figs/
    python3 rl_ws/utils/make_figures.py --out DIR
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from analyze_seeds import parse

# ── paleta (slots en orden fijo; nunca ciclar) ───────────────────────────────
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # blue, orange, aqua, yellow
DIV_NEG, DIV_POS, NEUTRAL = "#e34948", "#2a78d6", "#f0efec"   # red <-> blue, gray mid
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"

TRACKS = ["flat", "ramps", "steps1m", "pallets"]
TRACK_C = dict(zip(TRACKS, CAT))
COND = {"E1": "Full system", "E3": "No terrain reward",
        "E4": "Flippers parked", "E8": "No gate"}
COND_C = {"E1": CAT[0], "E3": CAT[1], "E4": CAT[2], "E8": CAT[3]}
PAT = {"E1": "seed?.log", "E3": "e3_seed?.log",
       "E4": "e4_seed?.log", "E8": "e8_seed?.log"}

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
    """Rejilla recesiva, sin marco superior/derecho: los datos por delante."""
    ax.grid(True, color=INK3, alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def band(ax, x, M, color, label, lw=1.8):
    """Media +- 1 desv entre semillas. La banda ES la incertidumbre, no decoracion."""
    m, s = M.mean(0), M.std(0)
    ax.fill_between(x, m - s, m + s, color=color, alpha=.16, lw=0)
    ax.plot(x, m, color=color, lw=lw, label=label, solid_capstyle="round")


def load(runs: Path, pat: str, n=None):
    R = [parse(p) for p in sorted(runs.glob(pat))]
    R = [r for r in R if len(r[0])]
    if not R:
        raise SystemExit(f"sin logs para {pat} en {runs}")
    n = n or min(len(r[0]) for r in R)
    return R, n


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ── figuras ──────────────────────────────────────────────────────────────────
def fig1(R, n, out):
    """Global success, E1. La figura del aplanamiento."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = R[0][0][:n]
    band(ax, x, np.stack([r[1][:n] for r in R]), CAT[0], "Mean of 3 seeds")
    ax.axvspan(150, n - 1, color=INK3, alpha=.07, lw=0)
    ax.text(0.98, 0.06, "learning plateaus by ~150 iterations", transform=ax.transAxes,
            ha="right", color=INK2, fontsize=8, style="italic")
    ax.set(xlabel="Training iteration", ylabel="Success rate",
           title="Overall success rate (shaded: ±1 s.d. across seeds)", ylim=(0, .75))
    style(ax); ax.legend(loc="upper left")
    save(fig, out, "fig1_global_learning_curve")


def fig2(R, n, out):
    """Success por pista, E1. Etiquetado directo -- identidad no depende del color."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = R[0][0][:n]
    for t in TRACKS:
        M = np.stack([r[3][t][:n] for r in R])
        band(ax, x, M, TRACK_C[t], t)
        ax.annotate(t, (x[-1], M.mean(0)[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TRACK_C[t], fontsize=8, va="center", fontweight="bold")
    ax.set(xlabel="Training iteration", ylabel="Success rate",
           title="Per-track success rate (shaded: ±1 s.d. across seeds)",
           ylim=(0, 1.02), xlim=(0, n * 1.12))
    style(ax); ax.legend(loc="center right", ncol=1)
    save(fig, out, "fig2_per_track_learning_curves")


def fig3(R, n, out):
    """Retorno vs exito en PANELES separados -- nunca un eje dual."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 4.0), sharex=True)
    x = R[0][0][:n]
    Rt = np.stack([r[2][:n] for r in R])
    a1.fill_between(x, Rt.mean(0) - Rt.std(0), Rt.mean(0) + Rt.std(0),
                    color=CAT[1], alpha=.16, lw=0)
    a1.plot(x, Rt.mean(0), color=CAT[1], lw=1.8, label="Average episode return")
    a1.set(ylabel="Return", title="Return keeps rising while success plateaus")
    style(a1, pct=False); a1.legend(loc="lower right")
    band(a2, x, np.stack([r[1][:n] for r in R]), CAT[0], "Success rate")
    a2.set(xlabel="Training iteration", ylabel="Success rate", ylim=(0, .75))
    style(a2); a2.legend(loc="lower right")
    save(fig, out, "fig3_return_vs_success")


def fig4(C, out):
    """Ablacion: barras agrupadas por pista. Barras finas, hueco de 2px entre ellas."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w, keys = .20, ["E1", "E3", "E4", "E8"]
    xs = np.arange(len(TRACKS))
    for i, k in enumerate(keys):
        m = [np.mean(C[k][t]) for t in TRACKS]
        e = [np.std(C[k][t]) for t in TRACKS]
        ax.bar(xs + (i - 1.5) * (w + .012), m, w, yerr=e, capsize=2.5,
               color=COND_C[k], label=COND[k], zorder=3,
               error_kw=dict(ecolor=INK2, lw=.9))
    ax.set_xticks(xs); ax.set_xticklabels(TRACKS)
    ax.set(ylabel="Success rate", ylim=(0, 1.05),
           title="Ablation study (mean ± s.d., 3 seeds, iterations 200–250)")
    style(ax); ax.legend(loc="upper right")
    save(fig, out, "fig4_ablation_by_track")


def fig5(C, out):
    """Deltas respecto a la base: datos con POLARIDAD -> paleta divergente.

    UN PANEL POR CONDICION, no barras agrupadas: el color ya esta ocupado
    codificando el signo, asi que la condicion necesita su propio canal. Con
    barras agrupadas no habia forma de saber que barra era cual -- las etiquetas
    flotaban sin anclarse a su grupo.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    for ax, k in zip(axes, ["E3", "E4", "E8"]):
        d = [np.mean(C[k][t]) - np.mean(C["E1"][t]) for t in TRACKS]
        ax.bar(TRACKS, d, .55, color=[DIV_POS if v >= 0 else DIV_NEG for v in d],
               edgecolor="white", lw=.8, zorder=3)
        for xi, v in enumerate(d):
            ax.text(xi, v + (.008 if v >= 0 else -.008), f"{v:+.0%}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=.9)
        ax.set_title(f"Removed: {COND[k].lower()}", fontsize=9.5)
        ax.set_ylim(-.185, .165)
        style(ax)
    axes[0].set_ylabel("Change vs. full system")
    # En el panel derecho la leyenda pisaba la etiqueta -13%; aqui arriba a la
    # derecha del izquierdo el area esta vacia en ambas condiciones.
    axes[0].legend(handles=[Patch(facecolor=DIV_POS, label="Improves"),
                            Patch(facecolor=DIV_NEG, label="Degrades")],
                   loc="upper right")
    fig.suptitle("Effect of removing each component", y=1.03)
    save(fig, out, "fig5_ablation_deltas")


def fig6(runs, out):
    """steps1m en las 3 condiciones -- donde el efecto de los flippers es nitido."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k in ["E1", "E3", "E4", "E8"]:
        R = [parse(p) for p in sorted(runs.glob(PAT[k]))]
        # Una semilla de E8 perdio sus primeras 150 iteraciones (tee sin -a al
        # reanudar). Para la CURVA se usan solo las semillas con cobertura
        # completa; la TABLA sigue con las 3, que si comparten la ventana final.
        full = max(len(r[0]) for r in R)
        R = [r for r in R if len(r[0]) == full]
        n = min(full, 250)
        band(ax, R[0][0][:n], np.stack([r[3]["steps1m"][:n] for r in R]),
             COND_C[k], f"{COND[k]}" + ("" if len(R) == 3 else f" (n={len(R)})"))
    ax.set(xlabel="Training iteration", ylabel="Success rate",
           title="steps1m: removing the gate nearly doubles success", ylim=(0, .55))
    style(ax); ax.legend(loc="upper left")
    save(fig, out, "fig6_steps1m_ablation_curves")


def fig7(runs, out):
    """Modos de terminacion. Apiladas: hueco de 2px entre segmentos."""
    LAB = [("META alcanzada", "Goal reached", CAT[2]),
           ("atascado", "Stuck (no progress)", CAT[3]),
           ("inclinado", "Fall (tipped over)", CAT[1]),
           ("piso|bajo", "Fell off surface", CAT[0])]
    txt = "\n".join(p.read_text(errors="ignore") for p in sorted(runs.glob("seed?.log")))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    counts = {t: Counter() for t in TRACKS}
    for t in TRACKS:
        for line in re.findall(rf"\[{t}\] [^\n]*", txt):
            for pat, lab, _ in LAB:
                if re.search(pat, line):
                    counts[t][lab] += 1
                    break
    bottom = np.zeros(len(TRACKS))
    for _, lab, c in LAB:
        v = np.array([counts[t][lab] / max(sum(counts[t].values()), 1) for t in TRACKS])
        ax.bar(TRACKS, v, .55, bottom=bottom, color=c, label=lab,
               edgecolor="white", lw=1.2, zorder=3)
        for i, (b, vv) in enumerate(zip(bottom, v)):
            if vv > .07:
                ax.text(i, b + vv / 2, f"{vv:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom += v
    ax.set(ylabel="Share of episodes", ylim=(0, 1),
           title="How episodes end (full system, 3 seeds pooled)")
    style(ax); ax.legend(loc="center left", bbox_to_anchor=(1.01, .5))
    save(fig, out, "fig7_termination_modes")


def fig8(runs, out):
    """pallets en E4, semilla a semilla: honestidad sobre n=1."""
    R, n = load(runs, "e4_seed?.log"); n = min(n, 250)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, r in enumerate(R):
        ax.plot(r[0][:n], r[3]["pallets"][:n], color=CAT[i], lw=1.6, label=f"seed {i+1}")
    ax.set(xlabel="Training iteration", ylabel="Success rate",
           title="pallets with flippers parked: only 1 of 3 seeds learns", ylim=(0, .55))
    ax.text(.03, .93, "single-seed observation — not an established effect",
            transform=ax.transAxes, fontsize=8, color=INK2, style="italic", va="top")
    style(ax); ax.legend(loc="upper left", bbox_to_anchor=(.03, .85))
    save(fig, out, "fig8_pallets_seed_variance")


def fig9(runs, out):
    """Donde muere pallets a lo largo de su ruta de 73 waypoints."""
    txt = "\n".join(p.read_text(errors="ignore") for p in sorted(runs.glob("seed?.log")))
    wp = [int(m) for m in re.findall(r"\[pallets\][^\n]*wp=(\d+)", txt)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(wp, bins=range(0, 78, 4), color=CAT[0], edgecolor="white", lw=.8, zorder=3)
    ax.axvline(73, color=INK2, lw=1.1, ls="--")
    ax.text(72, ax.get_ylim()[1] * .93, "goal (wp 73) ", ha="right",
            fontsize=8, color=INK2)
    ax.set(xlabel="Waypoint reached at episode end", ylabel="Episodes",
           title=f"pallets: failures spread along the route (n={len(wp)})")
    style(ax, pct=False)
    save(fig, out, "fig9_pallets_failure_location")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("figs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    R1, n1 = load(a.runs, "seed?.log")
    print(f"E1: {len(R1)} semillas x {n1} iteraciones -> {a.out}/")

    # Ventana comun 200-250 para todas las condiciones. Seleccion por NUMERO de
    # iteracion, no por posicion: una semilla de E8 perdio sus primeras 150
    # iteraciones al reanudar, asi que un slice [200:250] cogeria iteraciones
    # distintas en cada corrida -- y sin avisar.
    C = {}
    for k in ["E1", "E3", "E4", "E8"]:
        R, _ = load(a.runs, PAT[k])
        C[k] = {t: [float(np.nanmean(r[3][t][(r[0] >= 200) & (r[0] < 250)])) for r in R]
                for t in TRACKS}

    fig1(R1, n1, a.out); fig2(R1, n1, a.out); fig3(R1, n1, a.out)
    fig4(C, a.out); fig5(C, a.out); fig6(a.runs, a.out)
    fig7(a.runs, a.out); fig8(a.runs, a.out); fig9(a.runs, a.out)


if __name__ == "__main__":
    main()
