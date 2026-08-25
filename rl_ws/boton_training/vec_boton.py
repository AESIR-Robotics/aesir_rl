#!/usr/bin/env python3
"""
vec_boton.py — N BotonArmEnv en paralelo. Espejo de
mujoco_sim_base.VecMujocoEnv, misma convencion y mismas metricas.

Threads, no procesos: mj_step libera el GIL, asi que dan paralelismo real sin
pagar serializacion entre procesos (el mismo razonamiento que documenta la Fase
6 del proyecto para la base).

Auto-reset con la convencion gym/SB3: cuando un env termina, la obs que devuelve
step() ya es la del episodio NUEVO; el reward/done terminal salen igual para el
buffer de PPO, y la obs terminal va en info["terminal_obs"].

Cada reset sortea un boton nuevo de `targets`, asi que agotar el presupuesto de
tiempo (boton_env.EPISODE_TIME_S) sin hundirlo cuenta como fallo y se pasa al
siguiente boton -- que es exactamente el comportamiento pedido.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from typing import Dict, List, Optional

import mujoco
import numpy as np

from . import config as C
from boton_env import (BotonArmEnv, BOTON_MAZE_REF, EPISODE_MAX_STEPS,
                       EPISODE_TIME_S, TARGETS_ALCANZABLES, XML_PATH)

_WINDOW = 100    # episodios de la ventana movil de las metricas


class VecBotonEnv:
    """N envs del brazo contra la torre, con auto-reset y metricas por boton."""

    def __init__(self, n_envs: int = C.N_ENVS,
                 targets: Optional[List[int]] = None,
                 max_steps: int = EPISODE_MAX_STEPS,
                 randomize_torre: bool = True,
                 seed: int = 0,
                 verbose: bool = True):
        self.n = n_envs
        self.targets = list(targets) if targets else list(TARGETS_ALCANZABLES)
        self.verbose = verbose

        # Compilar el XML una vez y repartir copias (una por env: MjModel es
        # mutable -- el jitter de la torre escribe en body_pos/body_quat).
        base_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.envs = [
            BotonArmEnv(model=copy.deepcopy(base_model),
                        max_steps=max_steps,
                        randomize_torre=randomize_torre,
                        targets=self.targets,
                        seed=seed + i)
            for i in range(n_envs)
        ]
        self.obs_dim = self.envs[0].obs_len
        self.act_dim = self.envs[0].act_len
        self._pool = ThreadPoolExecutor(max_workers=n_envs)

        self._ep_return = np.zeros(n_envs, dtype=np.float64)
        self._ep_history: List[float] = []
        self._success_history: List[bool] = []
        # Tasa de exito POR BOTON: la global esconde que uno facil compense a
        # uno dificil (o al reves). Mismo criterio que success_rate_by_track.
        self._success_by_target: Dict[int, List[bool]] = {t: [] for t in self.targets}
        self._steps_to_press: List[int] = []
        self._ajeno_history: List[bool] = []
        # Con TRES condiciones terminales distintas (boton ajeno, tocar con algo
        # que no es la garra, autocolision) mas el timeout, la tasa de exito sola
        # no dice donde esta el problema. Esto reparte los fallos por motivo.
        self._reasons: deque = deque(maxlen=_WINDOW)
        # Exito POR MODO. El desglose por indice de boton dejo de significar nada
        # al meter los botones sueltos (ahi el objetivo es siempre el indice 0) y
        # ocultaba justo lo que hay que ver: si la politica se refugia en la
        # torre, que ya sabia, y no aprende los sueltos.
        self._por_modo = {"torre": deque(maxlen=_WINDOW),
                          "sueltos": deque(maxlen=_WINDOW)}
        # ESCALONES INTERMEDIOS. Con exito raro, la tasa se queda en 0% mucho
        # rato y no dice si la politica se acerca o esta atascada. Estos tres si:
        # son los peldanos del reward y se mueven antes que el exito.
        self._touch_history: deque = deque(maxlen=_WINDOW)
        self._fracmax_history: deque = deque(maxlen=_WINDOW)
        self._hold_history: deque = deque(maxlen=_WINDOW)
        self._n_pressed = 0

        if verbose:
            etq = ", ".join(f"{t}({BOTON_MAZE_REF[t]})" for t in self.targets)
            print(f"[vec] n_envs={n_envs} | botones={etq} | "
                  f"presupuesto={EPISODE_TIME_S:.0f}s ({max_steps} pasos) por episodio")

    # ── API ────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        obs = list(self._pool.map(lambda e: e.reset(), self.envs))
        self._ep_return[:] = 0.0
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        results = list(self._pool.map(lambda i: self.envs[i].step(actions[i]),
                                      range(self.n)))
        obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        rews = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        infos: List[dict] = []

        for i, (o, r, d, inf) in enumerate(results):
            self._ep_return[i] += r
            if d:
                won = bool(inf.get("success"))
                tgt = int(inf.get("target", -1))
                self._push(self._ep_history, float(self._ep_return[i]))
                self._push(self._success_history, won)
                self._push(self._ajeno_history, bool(inf.get("fin_boton_ajeno")))
                self._reasons.append(inf.get("reason", ""))
                self._touch_history.append(bool(inf.get("tocado")))
                self._fracmax_history.append(float(inf.get("frac_max", 0.0)))
                self._hold_history.append(float(inf.get("hold_best_s", 0.0)))
                md = inf.get("modo")
                if md in self._por_modo:
                    self._por_modo[md].append(won)
                if tgt in self._success_by_target:
                    self._push(self._success_by_target[tgt], won)
                if won:
                    self._n_pressed += 1
                    self._push(self._steps_to_press, int(inf.get("steps", 0)))
                self._ep_return[i] = 0.0
                inf["terminal_obs"] = o
                # Agotar el tiempo es TRUNCAR, no terminar: el valor del estado
                # final no es cero (el brazo podria seguir), asi que GAE debe
                # bootstrapear ahi. Sin esta marca, PPO aprende que quedarse sin
                # tiempo "vale 0" y sesga el critico.
                inf["truncated"] = bool(inf.get("timeout"))
                o = self.envs[i].reset()          # auto-reset: sortea otro boton
            obs[i] = o
            rews[i] = r
            dones[i] = float(d)
            infos.append(inf)
        return obs, rews, dones, infos

    # ── metricas ───────────────────────────────────────────────────────────
    @staticmethod
    def _push(buf: list, v) -> None:
        buf.append(v)
        if len(buf) > _WINDOW:
            buf.pop(0)

    def avg_return(self) -> float:
        return float(np.mean(self._ep_history)) if self._ep_history else float("nan")

    def success_rate(self) -> float:
        """Fraccion de los ultimos 100 episodios TERMINADOS que hundieron el
        boton. Es la metrica de tarea: n_pressed es acumulado (siempre sube, no
        dice si mejora) y avg_return mezcla progreso con castigos."""
        return float(np.mean(self._success_history)) if self._success_history else float("nan")

    def success_rate_by_target(self) -> Dict[int, float]:
        return {t: (float(np.mean(h)) if h else float("nan"))
                for t, h in self._success_by_target.items()}

    def avg_press_time_s(self) -> float:
        """Segundos simulados medios hasta hundir el boton, en los exitos."""
        if not self._steps_to_press:
            return float("nan")
        return float(np.mean(self._steps_to_press)) * self.envs[0]._dt

    def progreso(self) -> Dict[str, float]:
        """toca / hunde / sostiene sobre los ultimos 100 episodios."""
        if not self._touch_history:
            return {}
        return dict(toca=float(np.mean(self._touch_history)),
                    hunde=float(np.mean(self._fracmax_history)),
                    sostiene=float(np.mean(self._hold_history)),
                    sostiene_max=float(np.max(self._hold_history)))

    def success_by_mode(self) -> Dict[str, float]:
        return {k: (float(np.mean(v)) if v else float("nan"))
                for k, v in self._por_modo.items()}

    def set_dificultad(self, x: float) -> None:
        for env in self.envs:
            env.dificultad = float(x)

    def reason_mix(self) -> Dict[str, float]:
        """Reparto de los ultimos 100 episodios por motivo de fin."""
        if not self._reasons:
            return {}
        n = len(self._reasons)
        return {k: v / n for k, v in Counter(self._reasons).most_common()}

    def wrong_button_rate(self) -> float:
        """Fraccion de los ultimos 100 episodios que acabaron por tocar un boton
        que no era el objetivo. Es un modo de fallo distinto del timeout y hay
        que verlo aparte: uno dice "no llego", el otro "llego al sitio
        equivocado"."""
        return float(np.mean(self._ajeno_history)) if self._ajeno_history else float("nan")

    @property
    def n_pressed(self) -> int:
        return self._n_pressed

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for e in self.envs:
            e.close()
