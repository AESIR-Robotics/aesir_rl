#!/usr/bin/env python3
"""Curva de entrenamiento del agente de locomocion (base) para el paper ECoR:
retorno arriba, exito abajo, mismo formato que la figura del boton.
Banda = rango de las semillas convergidas; las que no despegaron, punteadas."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from analyze_seeds import parse

BLUE, RED, INK2, INK3 = "#2a78d6", "#e34948", "#52514e", "#8a8983"
plt.rcParams.update({"figure.dpi":160,"savefig.dpi":300,"savefig.bbox":"tight",
    "font.size":11,"axes.labelsize":11,"legend.fontsize":10,"xtick.labelsize":10,
    "ytick.labelsize":10,"axes.edgecolor":INK3,"axes.linewidth":.8,
    "xtick.color":INK2,"ytick.color":INK2,"legend.frameon":False})
sm = lambda y, w=15: np.convolve(y, np.ones(w)/w, mode="valid")

runs = [parse(p) for p in sorted(Path("runs/paper_archive").glob("e1p_seed?.log"))]
n = min(len(r[0]) for r in runs)
dead = [np.mean(r[1][250:300]) < .15 for r in runs]
x = runs[0][0][:n][14:]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True,
                             gridspec_kw={"hspace": .12})
for ax, idx, lab, pct in [(a1, 2, "Mean episodic return", False),
                          (a2, 1, "Success rate", True)]:
    C = np.stack([sm(r[idx][:n]) for r, d in zip(runs, dead) if not d])
    ax.fill_between(x, C.min(0), C.max(0), color=BLUE, alpha=.18, lw=0)
    ax.plot(x, C.mean(0), color=BLUE, lw=2.0,
            label=f"Converged ({len(C)} seeds)")
    first = True
    for r, d in zip(runs, dead):
        if d:
            ax.plot(x, sm(r[idx][:n]), color=RED, lw=1.3, ls=(0,(3,2)),
                    label=f"Never left initial policy ({sum(dead)})" if first else None)
            first = False
    ax.set_ylabel(lab)
    ax.grid(True, color=INK3, alpha=.22, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if pct: ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
a1.legend(loc="upper left")
a2.set_xlabel("PPO iteration")
fig.savefig("figs/ecor_fig_base_reward.png"); plt.close(fig)
print("figs/ecor_fig_base_reward.png")
