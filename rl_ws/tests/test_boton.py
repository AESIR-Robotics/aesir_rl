#!/usr/bin/env python3
"""
test_boton.py — Pipeline de prueba del brazo contra la TORRE de botones.

Modos:

  --diag      Diagnostico: la torre del maze (hub_N/door_N) y por que no es
              accionable alli, mas la geometria de la replica actuable.

  --reach     IK offline: para cada boton, ¿existe una pose del brazo que ponga
              el TCP en su cara CON la garra encarandolo? Separa "el entorno no
              da" de "el controlador no llega".

  (default)   Controlador SCRIPTED (sin red): servo cartesiano proporcional que
              lleva el TCP al boton y lo hunde. Sirve para dos cosas:
                1. Demostrar que la tarea es FISICAMENTE resoluble antes de
                   gastar horas de PPO en ella.
                2. Validar que el contrato de accion (twist cartesiano unitless
                   en arm_base_link) basta para la tarea — es el mismo que
                   consume MoveIt Servo, asi que si aqui funciona, el
                   despliegue por /servo_node/delta_twist_cmds tambien.

  --random    Politica aleatoria, como piso de comparacion.

ESTE SCRIPT NO ENTRENA. Solo corre episodios con una politica ya existente
(scripted, aleatoria o un checkpoint) y sale. Para entrenar:
    python3 -m boton_training.train_boton

  --policy CKPT   Carga una politica ENTRENADA y la corre en vez del scripted.
                  Funciona con TODOS los modos: --render para verla, --reward
                  para su desglose, --jitter/--repeat para medirla.

Uso:
    cd rl_ws
    python3 tests/test_boton.py --diag
    python3 tests/test_boton.py --reach
    python3 tests/test_boton.py                  # los 5 botones, scripted
    python3 tests/test_boton.py --render         # con viewer de MuJoCo
    python3 tests/test_boton.py --boton 0 --render
    python3 tests/test_boton.py --jitter --repeat 4
    python3 tests/test_boton.py --random

    # ver la politica entrenada  (--render se encarga solo de PYGLFW_LIBRARY_VARIANT)
    python3 tests/test_boton.py --policy ../checkpoints_boton/boton_best.pt --render
    # medirla (sin ruido de exploracion)
    python3 tests/test_boton.py --policy ../checkpoints_boton/boton_best.pt --jitter --repeat 10
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # rl_ws/ al path

import mujoco                                        # noqa: E402
from boton_env import (                              # noqa: E402
    BotonArmEnv, BOTON_MAZE_REF, JOINT_LIMIT_MARGIN, N_BOTONES, PRESS_TRAVEL,
    PRESS_THRESH_FRAC, REWARD_TERMS, SING_COND_STOP, SING_COND_WARN,
    TARGETS_ALCANZABLES, XML_PATH,
)

# La torre es una CUPULA: cada boton se hunde por SU normal, no por una comun.
# Se consulta con env.button_normal(); no hay una constante global de direccion.

PRESS_OVERSHOOT   = 0.030   # m detras de la cara del boton al que se apunta
HOLD_DEPTH        = 0.002   # m: lo justo para mantenerlo hundido sin empujar mas
KP_LIN            = 4.0
KP_ANG            = 3.0
# Eje +Y de link_6 = normal de la cara del taco que forman los dedos cerrados.
NUB_AXIS_L6       = np.array([0.0, 1.0, 0.0])


class ScriptedPresser:
    """Servo cartesiano proporcional en arm_base_link, por WAYPOINTS.

    La torre es una cupula: cada boton mira en una direccion distinta y el
    cuerpo de la torre se interpone. Un servo que apunta en linea recta desde
    la pose de reposo mete la muñeca contra la torre — medido: solo acertaba el
    boton de la cuspide (1 de 4). Asi que la aproximacion se hace SIEMPRE por el
    eje del boton, con tres puntos encadenados:

        P1 = cara + normal*d      lejos, ya alineado con el eje (d adaptativo)
        P2 = cara + normal*0.05   cerca, listo para empujar
        P3 = cara - normal*0.03   detras de la cara: el overshoot es la fuerza

    El avance de waypoint se LATCHEA (no se retrocede). Sin latcheo el TCP
    rebasaba el punto, volvia a verse lejos y el servo tiraba hacia atras:
    oscilaba sin llegar a tocar nunca el boton.

    Es un controlador de demostracion, no un planificador: sirve para probar que
    la tarea es resoluble y que el contrato de accion (twist cartesiano unitless
    en arm_base_link, el de MoveIt Servo) basta. Una politica aprendida no tiene
    por que imitarlo.
    """

    # El primer retroceso es ADAPTATIVO. Un valor fijo no vale para toda la
    # cupula: al boton frontal le conviene retroceder 20 cm (queda casi encima
    # del robot y hay que abrirse), pero al de la cuspide ese mismo punto le
    # cae a z~1.0 m, fuera del alcance, y el servo se clava ahi. Se coge el
    # retroceso MAS GRANDE cuyo punto siga dentro de REACH_SAFE del hombro.
    WP1_CANDIDATOS = (0.20, 0.17, 0.14, 0.11, 0.08)
    REACH_SAFE     = 0.80               # m desde el hombro (max medido: 1.079)
    SHOULDER_OFF   = np.array([0.0, 0.0, 0.1583])   # arm_base_link -> joint_1
    WP_NEAR    = 0.05                   # segundo waypoint, sobre el eje
    WP_TOL     = (0.05, 0.025)          # tolerancia para pasar de P1->P2 y P2->P3
    STALL_STEPS = 25                    # pasos sin acercarse antes de rendirse

    def __init__(self) -> None:
        self.wp = 0
        self._best = np.inf
        self._stall = 0

    def reset(self) -> None:
        self.wp = 0
        self._best = np.inf
        self._stall = 0

    def _wp1_offset(self, p_ab, R_ab, tgt, n) -> float:
        p_sh = p_ab + R_ab @ self.SHOULDER_OFF
        for off in self.WP1_CANDIDATOS:
            if np.linalg.norm((tgt + n * off) - p_sh) <= self.REACH_SAFE:
                return off
        return self.WP1_CANDIDATOS[-1]

    def __call__(self, env: BotonArmEnv) -> np.ndarray:
        p_ab, R_ab = env.arm_base_frame()
        tcp = env.tcp()
        tgt = env.target_pos()
        n = env.button_normal()          # eje de presion DE ESTE boton
        offsets = (self._wp1_offset(p_ab, R_ab, tgt, n),
                   self.WP_NEAR, -PRESS_OVERSHOOT)

        if self.wp < 2:
            aim_cur = tgt + n * offsets[self.wp]
            dist_wp = float(np.linalg.norm(aim_cur - tcp))
            if dist_wp < self._best - 1e-3:
                self._best, self._stall = dist_wp, 0
            else:
                self._stall += 1
            # Se avanza al llegar, o al estancarse: un waypoint intermedio puede
            # caer fuera del alcance (le pasa al boton de la cuspide, que queda
            # muy alto) y entonces el servo se quedaria clavado en el para
            # siempre. El waypoint es una guia, no un requisito.
            if dist_wp < self.WP_TOL[self.wp] or self._stall >= self.STALL_STEPS:
                self.wp += 1
                self._best, self._stall = np.inf, 0

        aim = tgt + n * offsets[self.wp]
        # Fase de SOSTENER: en cuanto el boton toca fondo, el objetivo pasa de
        # "3 cm por detras de la cara" a "justo en la cara". Seguir empujando
        # contra el tope mete el brazo entero hacia dentro y acaba tocando el
        # boton con la muñeca -- que es condicion de corte. Medido: 8 de 15
        # episodios morian asi.
        if env.press_frac() >= PRESS_THRESH_FRAC:
            aim = tgt - n * HOLD_DEPTH
        v_cmd = R_ab.T @ (KP_LIN * (aim - tcp))   # twist lineal en arm_base_link

        # Orientacion: la cara del taco de los dedos debe mirar CONTRA la normal
        # del boton. Solo se restringen 2 GDL (hacia donde MIRA la garra); el
        # giro sobre su propio eje se deja libre — para presionar da igual, y
        # pedir una pose de 6 GDL completa reduce el espacio de soluciones.
        R6 = env.data.xmat[env._b_l6].reshape(3, 3)
        n_cur = R6 @ NUB_AXIS_L6
        axis = np.cross(n_cur, -n)
        s_ang = np.linalg.norm(axis)
        if s_ang > 1e-6:
            ang = np.arctan2(s_ang, float(np.dot(n_cur, -n)))
            w_cmd = R_ab.T @ (KP_ANG * ang * (axis / s_ang))
        else:
            w_cmd = np.zeros(3)

        a = np.zeros(7, dtype=np.float32)
        a[:3] = np.clip(v_cmd, -1.0, 1.0)
        a[3:6] = np.clip(w_cmd, -1.0, 1.0)
        a[6] = 1.0                                # garra cerrada: puntas juntas
        return a


def cargar_politica(ckpt_path: str, obs_dim: int, act_dim: int):
    """Carga un checkpoint de boton_training/train_boton.py y devuelve una
    funcion accion(env) usable igual que ScriptedPresser.

    Se evalua con la MEDIA de la Beta, no muestreando: el ruido de exploracion
    es parte del entrenamiento, no de la politica. Muestreando, los numeros
    salen peores que la politica real y no son reproducibles.
    """
    import torch
    from boton_training.net import BetaActorCritic

    ck = torch.load(ckpt_path, map_location="cpu")
    d_obs = int(ck.get("obs_dim", obs_dim))
    d_act = int(ck.get("act_dim", act_dim))
    if (d_obs, d_act) != (obs_dim, act_dim):
        raise ValueError(
            f"El checkpoint es de obs={d_obs} act={d_act} y este env es "
            f"obs={obs_dim} act={act_dim}. Cambio la tarea desde que se entreno: "
            f"ese checkpoint no vale.")
    net = BetaActorCritic(d_obs, d_act)
    net.load_state_dict(ck["policy"])
    net.eval()
    print(f"politica cargada: {ckpt_path}")
    print(f"  iter={ck.get('iter','?')}  best_avg_ret={ck.get('best', float('nan')):.2f}  "
          f"entrenada sobre botones={ck.get('targets','?')}")

    def actuar(env):
        obs = env._observation()
        a, _, _, _ = net.act(obs, "cpu", deterministic=True)
        return a.astype(np.float32)
    return actuar


def reach_check(targets, nseeds: int = 40, iters: int = 300) -> None:
    """¿Es cada boton REALMENTE viable? Tres filtros, no uno.

    Busca poses del brazo que pongan el TCP en la cara del boton con la garra
    encarandola, y de las que lo consiguen reporta:

      1. ALCANCE      cuantas convergen (error < 10 mm)
      2. COLISION     cuantas de esas quedan libres de auto-colision y de tocar
                      la torre con algo que no sean los dedos
      3. CONDICION    cond(J) de esas poses, contra el hard_stop_singularity_
                      threshold=100 de servo_params.yaml

    Los tres hacen falta. Una version anterior de este chequeo solo miraba el
    alcance y daba por bueno el boton de la cuspide; al añadir el tercer filtro
    resulto que 50 de sus 51 poses estan por encima de 100, o sea que MoveIt
    Servo pararia el brazo en seco al intentarlas en el robot real.

    Separa ademas "el entorno no da" de "el controlador no llega": si hay poses
    viables, cualquier fallo del servo scripted es del servo."""
    env = BotonArmEnv(randomize_torre=False)
    env.reset()
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    from boton_env import TCP_OFFSET_L6

    def _colisiona(q, head_ok):
        d.qpos[env._q_arm] = q
        mujoco.mj_forward(m, d)
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            t1, t2 = g1 in env._g_torre, g2 in env._g_torre
            r1, r2 = g1 in env._g_robot, g2 in env._g_robot
            if t1 != t2 and (r1 or r2):
                g_rob = g2 if t1 else g1
                g_tor = g1 if t1 else g2
                if not (g_rob in env._g_finger and g_tor == head_ok):
                    return True
            elif r1 and r2:
                return True
        return False

    def _cond(q):
        d.qpos[env._q_arm] = q
        mujoco.mj_forward(m, d)
        R6 = d.xmat[env._b_l6].reshape(3, 3)
        tcp = d.xpos[env._b_l6] + R6 @ TCP_OFFSET_L6
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, tcp, env._b_l6)
        J = np.vstack([jp[:, env._v_arm], jr[:, env._v_arm]])
        sv = np.linalg.svd(J, compute_uv=False)
        return float(sv[0] / max(sv[-1], 1e-12))

    print(f"{'boton':>22} {'alcanzan':>9} {'sin colis':>10} {'cond min':>9} "
          f"{'cond med':>9} {'cond<100':>9}  veredicto")
    print("-" * 88)
    for idx, (label, target, normal) in enumerate(targets):
        target = np.asarray(target, dtype=float)
        want = -np.asarray(normal, dtype=float)   # la garra mira CONTRA la normal
        env.target = idx
        head_ok = env._g_head[idx]
        best = 1e9
        soluciones = []
        for s_i in range(nseeds):
            q = (env._joint_pos.copy() if s_i == 0
                 else rng.uniform(env._arm_lo * 0.95, env._arm_hi * 0.95))
            for _ in range(iters):
                d.qpos[env._q_arm] = q
                mujoco.mj_forward(m, d)
                R6 = d.xmat[env._b_l6].reshape(3, 3)
                tcp = d.xpos[env._b_l6] + R6 @ TCP_OFFSET_L6
                e_pos = target - tcp
                n_cur = R6 @ NUB_AXIS_L6
                ax = np.cross(n_cur, want)
                sa = np.linalg.norm(ax)
                e_ori = (np.arctan2(sa, float(np.dot(n_cur, want))) * (ax / sa)
                         if sa > 1e-9 else np.zeros(3))
                if np.linalg.norm(e_pos) < 0.004 and np.linalg.norm(e_ori) < 0.25:
                    break
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, tcp, env._b_l6)
                J = np.vstack([jp[:, env._v_arm], jr[:, env._v_arm] * 0.25])
                err = np.concatenate([e_pos, e_ori * 0.25])
                dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6), err)
                q = np.clip(q + 0.5 * dq, env._arm_lo, env._arm_hi)
            d.qpos[env._q_arm] = q
            mujoco.mj_forward(m, d)
            R6 = d.xmat[env._b_l6].reshape(3, 3)
            err = float(np.linalg.norm(target - (d.xpos[env._b_l6] + R6 @ TCP_OFFSET_L6)))
            best = min(best, err)
            if err < 0.010:
                soluciones.append(q.copy())

        libres = [q for q in soluciones if not _colisiona(q, head_ok)]
        conds = np.array([_cond(q) for q in libres]) if libres else np.array([])
        n_ok = int((conds < SING_COND_STOP).sum()) if conds.size else 0

        if not soluciones:
            veredicto = f"FUERA DE ALCANCE (mejor {best * 1000:.0f} mm)"
        elif not libres:
            veredicto = "SIEMPRE EN COLISION"
        elif n_ok == 0:
            veredicto = "SERVO LO PARARIA (cond > 100)"
        elif n_ok < 0.15 * len(conds):
            veredicto = "MARGINAL: pocas poses ejecutables"
        else:
            veredicto = "VIABLE"
        cmin = f"{conds.min():.0f}" if conds.size else "-"
        cmed = f"{np.median(conds):.0f}" if conds.size else "-"
        frac = f"{n_ok}/{len(conds)}" if conds.size else "-"
        print(f"{label:>22} {len(soluciones):>9} {len(libres):>10} {cmin:>9} "
              f"{cmed:>9} {frac:>9}  {veredicto}")
    print("-" * 88)
    print(f"cond<100 = poses por debajo de hard_stop_singularity_threshold "
          f"({SING_COND_STOP:.0f}) de servo_params.yaml.")
    print("Servo PARA EN SECO por encima de ese valor: esas poses no son")
    print("ejecutables en el robot real por muy buenas que se vean en MuJoCo.")
    env.close()


def reward_breakdown(policy: str, repeat: int, jitter: bool) -> None:
    """Desglose del reward por componente, como la tabla de test_base.py.

    Sirve para calibrar pesos: si un solo termino se lleva el 90% del total, o
    si un castigo dispara en trayectorias BUENAS, se ve aqui antes de gastar
    horas de PPO.
    """
    env = BotonArmEnv(randomize_torre=jitter, render=False, seed=0)
    filas = []
    for t in TARGETS_ALCANZABLES:
        for _ in range(repeat):
            env.fixed_target = t
            env.reset()
            presser = ScriptedPresser() if policy == "scripted" else None
            acc = {k: 0.0 for k in REWARD_TERMS}
            steps, info = 0, {}
            for _ in range(env.max_steps):
                if policy == "scripted":
                    a = presser(env)
                elif policy == "random":
                    a = np.random.uniform(-1, 1, env.act_len).astype(np.float32)
                else:
                    a = policy(env)
                _, _, done, info = env.step(a)
                steps += 1
                for k in REWARD_TERMS:
                    acc[k] += float(info[k])
                if done:
                    break
            filas.append((t, info.get("success", False), steps, acc))
    env.close()

    nombre = policy if isinstance(policy, str) else "entrenada"
    print(f"DESGLOSE DEL REWARD — politica={nombre}  jitter={jitter}  "
          f"episodios={len(filas)}")
    print()
    tot_abs = {k: sum(abs(f[3][k]) for f in filas) for k in REWARD_TERMS}
    suma_abs = sum(tot_abs.values()) or 1.0
    n_ok = sum(f[1] for f in filas)
    pasos = sum(f[2] for f in filas)
    print(f"{'termino':>14} {'total':>10} {'por paso':>10} {'% del |total|':>14}")
    print("-" * 52)
    for k in REWARD_TERMS:
        tot = sum(f[3][k] for f in filas)
        print(f"{k:>14} {tot:>10.2f} {tot / pasos:>10.4f} {100 * tot_abs[k] / suma_abs:>13.1f}%")
    print("-" * 52)
    gran = sum(sum(f[3][k] for f in filas) for k in REWARD_TERMS)
    print(f"{'TOTAL':>14} {gran:>10.2f} {gran / pasos:>10.4f}")
    print(f"\nexito {n_ok}/{len(filas)}   pasos totales {pasos}")
    print(f"\nEnvolvente segura vigente:")
    print(f"  vel. articular  <= {0.40 * 3.14:.2f} rad/s de 3.14 (joint_limits.yaml)")
    print(f"  acel. articular <= {0.50 * 10.0:.1f} rad/s^2 de 10.0 (ARM_JOINT_LIMITS)")
    print(f"  margen al tope de recorrido: {JOINT_LIMIT_MARGIN} rad")
    print(f"  cond. jacobiano: castigo de {SING_COND_WARN:.0f} a {SING_COND_STOP:.0f} "
          f"(servo_params.yaml: frena / para en seco)")


def run_episode(env: BotonArmEnv, target: int, policy: str, render: bool):
    env.fixed_target = target
    obs = env.reset()
    presser = ScriptedPresser() if policy == "scripted" else None
    best_frac, total_r, steps = 0.0, 0.0, 0
    min_dist = float("inf")
    info = {}
    while True:
        if policy == "scripted":
            a = presser(env)
        elif policy == "random":
            a = np.random.uniform(-1, 1, env.act_len).astype(np.float32)
        else:
            a = policy(env)                      # politica entrenada
        obs, r, done, info = env.step(a)
        total_r += r
        steps = info["steps"]
        best_frac = max(best_frac, info["frac"])
        min_dist = min(min_dist, info["dist"])
        if render:
            import time
            time.sleep(0.01)
        if done:
            break
    return dict(success=info.get("success", False), steps=steps, reward=total_r,
                best_frac=best_frac, min_dist=min_dist)


def diag() -> None:
    print("=" * 74)
    print("DIAGNOSTICO 1 — models/boton.xml (export CAD)  ->  ¿es un boton?")
    print("=" * 74)
    boton_xml = os.path.normpath(os.path.join(_HERE, "..", "..", "models", "boton.xml"))
    m = mujoco.MjModel.from_xml_path(boton_xml)
    d = mujoco.MjData(m)
    print(f"  joints={m.njnt}  actuadores={m.nu}  (un boton pasivo no deberia tener actuadores)")
    bad = 0
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lim = bool(m.jnt_limited[j])
        stf = float(m.jnt_stiffness[j])
        if not lim or stf == 0.0:
            bad += 1
        print(f"    {n:22s} limited={lim!s:5s} range={m.jnt_range[j]} stiffness={stf:.3f}")
    print(f"  -> {bad}/{m.njnt} joints sin recorrido definido y/o sin muelle de retorno.")

    adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j]
           for j in range(m.njnt)}
    slides = sorted(k for k in adr if "slide" in k)
    mujoco.mj_forward(m, d)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    print("\n  Sin tocar nada, tras 3 s de simulacion los vastagos se han MOVIDO SOLOS:")
    for k in slides:
        print(f"    {k:22s} qpos = {float(d.qpos[adr[k]]):+.4f} m")
    print("  -> arrancan enterrados en el casco convexo de su montura y la fisica")
    print("     los expulsa. No hay agujero: el casco convexo lo rellena.")

    print()
    print("=" * 74)
    print("DIAGNOSTICO 2 — la torre del maze  ->  ¿es accionable?")
    print("=" * 74)
    maze_path = os.path.normpath(os.path.join(_HERE, "..", "..", "models", "maze.xml"))
    src = open(maze_path).read()
    n_joints = src.count("<joint")
    hubs = sorted(set(re.findall(r'<body name="(hub_\d+)">', src)))
    doors = sorted(set(re.findall(r'<body name="(door_\d+)">', src)))
    print(f"  El maze SI tiene las torres, pero no se llaman 'boton': son pares")
    print(f"  hub_N / door_N. hubs={len(hubs)} doors={len(doors)}")
    print(f"    {hubs}")
    print(f"    {doors}")
    print(f"\n  joints en todo maze.xml: {n_joints}")
    print("  -> son cuerpos RIGIDOS pegados al mundo. Geometria sin mecanismo:")
    print("     no hay carrera, ni muelle, ni forma de detectar una pulsacion.")

    print("\n  Offset door-hub de cada boton (de los *_col de maze.xml):")
    cols = dict(re.findall(r'<geom name="((?:hub|door)_\d+)_col"[^>]*?pos="([^"]+)"', src, re.S))
    pares = [("hub_7", "door_7"), ("hub_8", "door_8"), ("hub_9", "door_9"),
             ("hub_10", "door_10"), ("hub_11", "door_11")]
    for h, dr in pares:
        if h in cols and dr in cols:
            ph = np.array([float(v) for v in cols[h].split()])
            pd = np.array([float(v) for v in cols[dr].split()])
            off = pd - ph
            nrm = off / (np.linalg.norm(off) + 1e-12)
            print(f"    {dr:8s} - {h:7s} = {np.round(off, 4)}   normal = {np.round(nrm, 3)}")
    print("  -> los cuatro laterales tienen offset DIAGONAL de igual modulo en dos")
    print("     ejes: estan a 45 grados hacia afuera-arriba. El quinto va recto a +Z.")
    print("     Es una CUPULA, no un panel plano. (Las cajas *_col del maze son")
    print("     axis-aligned: la AABB de una pieza girada, una aproximacion.)")

    print()
    print("=" * 74)
    print("DIAGNOSTICO 3 — replica actuable: aesir_arm_botones.xml + torre_botones.xml")
    print("=" * 74)
    env = BotonArmEnv(randomize_torre=False, freeze_base=True)
    env.reset()
    p_ab, R_ab = env.arm_base_frame()
    print(f"  arm_base_link (mundo)   = {np.round(p_ab, 4)}")
    print(f"  TCP en pose de reposo   = {np.round(env.tcp(), 4)}")
    print(f"  actuadores del modelo   = {env.model.nu}  (los botones no añaden ninguno)")
    print(f"  recorrido de cada boton = {PRESS_TRAVEL * 1000:.0f} mm"
          f"   (umbral de exito: {PRESS_THRESH_FRAC * 100:.0f}%)")
    print("\n  posicion de cada boton RELATIVA a arm_base_link (lo que ve la politica):")
    for i in range(N_BOTONES):
        env.target = i
        tgt = env.target_pos()
        rel = R_ab.T @ (tgt - p_ab)
        n_rel = R_ab.T @ env.button_normal()
        print(f"    boton_{i} (≙ {BOTON_MAZE_REF[i]:8s}) rel = {np.round(rel, 3)}"
              f"  normal = {np.round(n_rel, 3)}"
              f"  dist desde TCP = {np.linalg.norm(tgt - env.tcp()):.3f} m")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="solo diagnostico, sin correr episodios")
    ap.add_argument("--render", action="store_true", help="abre el viewer de MuJoCo")
    ap.add_argument("--random", action="store_true", help="politica aleatoria en vez de scripted")
    ap.add_argument("--policy", type=str, default="",
                    help="checkpoint de una politica entrenada (.pt)")
    ap.add_argument("--boton", type=int, default=None, help="probar solo este boton (0-4)")
    ap.add_argument("--all", action="store_true",
                    help="incluir tambien el boton del lado opuesto (no alcanzable con la base quieta)")
    ap.add_argument("--repeat", type=int, default=1, help="episodios por boton")
    ap.add_argument("--jitter", action="store_true",
                    help="aleatoriza posicion y giro de la torre")
    ap.add_argument("--reach", action="store_true",
                    help="IK offline: comprueba que los 5 botones son alcanzables")
    ap.add_argument("--reward", action="store_true",
                    help="tabla de desglose del reward por componente")
    args = ap.parse_args()

    if args.render and "PYGLFW_LIBRARY_VARIANT" not in os.environ:
        # En esta maquina (Wayland/XWayland + amdgpu) el viewer de MuJoCo
        # revienta con "Segmentation fault (core dumped)" al cerrar si GLFW usa
        # el backend de Wayland. El repo ya lo documenta en
        # base_training/watch_checkpoint.py, que se lanza a mano con esta
        # variable; aqui se pone sola. Hay que hacerlo ANTES de que se importe
        # glfw, o sea antes de crear el env con render=True.
        os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
        print("[viewer] PYGLFW_LIBRARY_VARIANT=x11 (evita el segfault de GLFW "
              "sobre Wayland al cerrar)")

    if args.diag:
        diag()
        return

    if args.reward:
        if args.policy:
            _e = BotonArmEnv(randomize_torre=False)
            pol = cargar_politica(args.policy, _e.obs_len, _e.act_len)
            _e.close()
        else:
            pol = "random" if args.random else "scripted"
        reward_breakdown(pol, args.repeat, args.jitter)
        return

    if args.reach:
        env = BotonArmEnv(randomize_torre=False)
        env.reset()
        from boton_env import BOTON_MAZE_REF
        targets = []
        for i in range(N_BOTONES):
            env.target = i
            targets.append((f"boton_{i} ({BOTON_MAZE_REF[i]})",
                            env.target_pos().copy(), env.button_normal().copy()))
        env.close()
        print("VIABILIDAD POR BOTON — alcance + colision + condicionamiento")
        print()
        reach_check(targets, nseeds=150)
        return

    if args.policy:
        _e = BotonArmEnv(randomize_torre=False)
        policy = cargar_politica(args.policy, _e.obs_len, _e.act_len)
        _e.close()
        etiqueta = f"entrenada ({os.path.basename(args.policy)})"
    else:
        policy = "random" if args.random else "scripted"
        etiqueta = policy
    if args.boton is not None:
        targets = [args.boton]
    elif args.all:
        targets = list(range(N_BOTONES))
    else:
        targets = list(TARGETS_ALCANZABLES)

    env = BotonArmEnv(randomize_torre=args.jitter, render=args.render, seed=0)
    print(f"modelo : {XML_PATH}")
    print(f"politica: {etiqueta}   torre_jitter={args.jitter}   episodios/boton={args.repeat}")
    print()
    print(f"objetivos: {targets}"
          f"{'   (el 4 esta en la cara opuesta: requiere mover la base)' if 4 in targets else ''}")
    print()
    print(f"{'boton':>6} {'ep':>3} {'exito':>6} {'pasos':>6} {'hundido':>8} "
          f"{'dist_min':>9} {'reward':>9}")
    print("-" * 56)

    rows = []
    for t in targets:
        for k in range(args.repeat):
            res = run_episode(env, t, policy, args.render)
            rows.append(res)
            print(f"{t:>6} {k:>3} {'SI' if res['success'] else 'no':>6} "
                  f"{res['steps']:>6} {res['best_frac'] * 100:>7.1f}% "
                  f"{res['min_dist']:>9.4f} {res['reward']:>9.2f}")
    env.close()

    n_ok = sum(r["success"] for r in rows)
    print("-" * 56)
    print(f"exito: {n_ok}/{len(rows)}  ({100.0 * n_ok / len(rows):.0f}%)   "
          f"pasos medios (exitosos): "
          f"{np.mean([r['steps'] for r in rows if r['success']]) if n_ok else float('nan'):.1f}")


if __name__ == "__main__":
    main()
