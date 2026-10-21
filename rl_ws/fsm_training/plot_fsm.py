#!/usr/bin/env python3
"""
plot_fsm.py — Plots for a train_fsm.py run.

Reads <run_dir>/metrics.csv (flushed every iteration by train_fsm.py, so it
can run WHILE training) and writes PNGs to <run_dir>/plots/:

  success.png   mission and per-stage success rate (rolling episode window)
  fails.png     episode outcomes within the rolling window
  returns.png   mean episode return per agent (env reward + task bonus)
  losses.png    policy / value loss, entropy and approximate KL per agent

Usage:
  .venv/bin/python3 rl_ws/fsm_training/plot_fsm.py checkpoints_fsm/<run>
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = [("sr_success", "mission success (final goal)"), ("sr_reached_tower", "reached pre-dock"),
          ("sr_docked", "docked"), ("sr_pressed", "button pressed")]
FAILS = ["SUCCESS", "base_to_tower", "dock", "arm", "base_to_goal", "plan_to_goal", "timeout", "nan"]
AGENTS = ("base", "arm")


def _num(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # The last line may be half-written when read during training.
    rows = [r for r in rows if None not in r.values()]
    return {k: [_num(r[k]) for r in rows] for k in (rows[0].keys() if rows else [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--x", choices=("iter", "env_steps", "wall_s"), default="iter")
    a = ap.parse_args()
    path = os.path.join(a.run_dir, "metrics.csv")
    if not os.path.isfile(path):
        print(f"{path} not found")
        return 1
    c = load(path)
    if not c:
        print("metrics.csv is empty")
        return 1
    x = c[a.x]
    xl = {"iter": "iteration", "env_steps": "control steps", "wall_s": "time (s)"}[a.x]
    out = os.path.join(a.run_dir, "plots")
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.run_dir))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, lab in STAGES:
        ax.plot(x, c[k], label=lab, lw=2.2 if k == "sr_success" else 1.3)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel(xl); ax.set_ylabel("rate (rolling window)")
    ax.set_title(f"{name}: success per stage"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "success.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for f in FAILS:
        k = f"fail_{f}"
        if k in c and any(v > 0 for v in c[k] if v == v):
            ax.plot(x, c[k], label=f)
    ax.set_xlabel(xl); ax.set_ylabel("episodes in window"); ax.grid(alpha=0.3)
    ax.set_title(f"{name}: episode outcome"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fails.png"), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ag in zip(axs, AGENTS):
        ax.plot(x, c[f"{ag}_ret_ep"]); ax.set_title(f"return/episode: {ag}")
        ax.set_xlabel(xl); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "returns.png"), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for row, ag in enumerate(AGENTS):
        for col, (k, lab) in enumerate((("pi", "policy loss"), ("v", "value loss"),
                                        ("ent", "entropy"), ("kl", "approx. KL"))):
            ax = axs[row, col]
            ax.plot(x, c.get(f"{ag}_{k}", [math.nan] * len(x)), ".-", ms=3)
            if k == "v":
                ax.plot(x, c.get(f"{ag}_v_warm", [math.nan] * len(x)), ".", ms=3, label="warm-up")
            ax.set_title(f"{ag}: {lab}", fontsize=9); ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel(xl)
    fig.tight_layout(); fig.savefig(os.path.join(out, "losses.png"), dpi=110); plt.close(fig)
    print(f"plots in {out} ({len(x)} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
