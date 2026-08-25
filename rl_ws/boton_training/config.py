#!/usr/bin/env python3
"""
config.py — Parametros del entrenamiento del BRAZO contra la torre de botones,
en un solo sitio. Mismo papel que rl_ws/base_training/config.py para la base.

Quien consume cada seccion:
  ../boton_env.py    → la TAREA (obs, reward, castigos, envolvente segura).
                       Esos parametros viven ALLI, no aqui: definen el problema,
                       no como se entrena.
  net.py             → arquitectura de la red
  vec_boton.py       → numero de envs y presupuesto por episodio
  train_boton.py     → hiperparametros de PPO, checkpoints, wandb

Los valores de PPO son los MISMOS que base_training/config.py salvo N_ENVS
(8 nucleos en esta maquina, no 14) y el proyecto de wandb. Se mantiene la
paridad a proposito: el algoritmo es identico al de la base, lo que cambia es
la tarea.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = Path(ROOT) / "checkpoints_boton"

# ── Envs paralelos ───────────────────────────────────────────────────────────
# mj_step libera el GIL, asi que los threads dan paralelismo real (ver
# mujoco_sim_base.VecMujocoEnv). 8 nucleos en esta maquina.
N_ENVS        = 8

# Botones a entrenar. None -> boton_env.TARGETS_ALCANZABLES ([0,1,2]).
# El 3 (cuspide) queda fuera porque casi todas sus poses superan el
# hard_stop_singularity_threshold de MoveIt Servo, y el 4 esta en la cara
# opuesta de la cupula. Ver la tabla en boton_env.TARGETS_ALCANZABLES.
TRAIN_TARGETS = None

# ── Red ──────────────────────────────────────────────────────────────────────
HIDDEN        = 256      # neuronas por capa del trunk (2 capas Tanh)
MAX_GRAD_NORM = 0.5

# ── Hiperparametros PPO ──────────────────────────────────────────────────────
STEPS_PER_ENV = 512      # pasos por env por rollout -> batch de T*N
ITERS         = 10000
PPO_EPOCHS    = 10
BATCH_SIZE    = 1024
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP          = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.005
LR            = 3e-4
SAVE_EVERY    = 50
DEVICE        = "auto"
WANDB_PROJECT = "AIDL-PPO-AESIR-BOTONES"
