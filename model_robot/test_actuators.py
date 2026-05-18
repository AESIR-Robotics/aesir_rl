"""
Test manual de actuadores — aesir_complete.xml
Carga el modelo completo (robot + mapa) y aplica movimientos de prueba
a cada grupo de actuadores en secuencia mientras el viewer está abierto.

Uso:
    cd /home/<user>/aesir_rl/model_robot
    MUJOCO_GL=egl python3 test_actuators.py
"""

import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "../workspace/src/aesir_robot_description/launch/aesir_complete.xml"

# ── duración de cada prueba (segundos de sim-time) ──────────────────────────
PHASE_DURATION = 3.0   # segundos por fase
DT             = 0.002  # timestep del modelo


def _act_idx(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def run_phase(viewer, model, data, label: str, ctrl_fn, duration: float = PHASE_DURATION):
    """Ejecuta ctrl_fn(t, data) durante `duration` segundos de sim-time."""
    t0 = data.time
    print(f"\n{'─'*55}")
    print(f"  PROBANDO: {label}")
    print(f"{'─'*55}")
    while data.time - t0 < duration:
        ctrl_fn(data.time - t0, data)
        mujoco.mj_step(model, data)
        viewer.sync()
    # Resetea controles al terminar la fase
    data.ctrl[:] = 0.0


def main():
    print("Cargando modelo…")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # ── Imprime mapa de actuadores ──────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  ACTUADORES DISPONIBLES")
    print(f"{'═'*55}")
    for i in range(model.nu):
        name = model.actuator(i).name
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  [{i:2d}] {name:<30} rango [{lo:+.2f}, {hi:+.2f}]")
    print(f"{'═'*55}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20

        # ── 1. RUEDAS DE TRACCIÓN (diff-drive) ─────────────────────────────
        l1 = _act_idx(model, "vel_drive_l_1")
        l2 = _act_idx(model, "vel_drive_l_2")
        l3 = _act_idx(model, "vel_drive_l_3")
        r1 = _act_idx(model, "vel_drive_r_1")
        r2 = _act_idx(model, "vel_drive_r_2")
        r3 = _act_idx(model, "vel_drive_r_3")

        def drive_forward(t, d):
            vel = 15.0
            for idx in [l1, l2, l3, r1, r2, r3]:
                d.ctrl[idx] = vel

        def drive_turn(t, d):
            for idx in [l1, l2, l3]:
                d.ctrl[idx] =  15.0
            for idx in [r1, r2, r3]:
                d.ctrl[idx] = -15.0

        run_phase(viewer, model, data, "RUEDAS — avance recto", drive_forward)
        run_phase(viewer, model, data, "RUEDAS — giro en sitio (rápido)", drive_turn)

        # ── 2. FLIPPERS (posición) ──────────────────────────────────────────
        fp = [_act_idx(model, f"pos_flipper_{i}") for i in range(1, 5)]

        def flipper_up(t, d):
            angle = np.sin(2 * np.pi * t / PHASE_DURATION) * 1.2
            for idx in fp:
                d.ctrl[idx] = angle

        run_phase(viewer, model, data, "FLIPPERS — barrido sinusoidal", flipper_up)

        # ── 3. BRAZO — joints 1-6 (posición) ───────────────────────────────
        arm = [_act_idx(model, f"pos_joint_{i}") for i in range(1, 7)]
        AMPLITUDES = [1.0, 0.8, 0.8, 0.6, 0.6, 0.4]

        def arm_wave(t, d):
            for k, idx in enumerate(arm):
                phase = k * np.pi / 3
                d.ctrl[idx] = AMPLITUDES[k] * np.sin(2 * np.pi * t / PHASE_DURATION + phase)

        run_phase(viewer, model, data, "BRAZO — onda en todos los joints", arm_wave, duration=5.0)

        # ── 4. GRIPPER ──────────────────────────────────────────────────────
        lf = _act_idx(model, "pos_left_finger")
        rf = _act_idx(model, "pos_right_finger")

        def gripper_open_close(t, d):
            pos = 0.015 * (1 + np.sin(2 * np.pi * t / PHASE_DURATION))
            d.ctrl[lf] = pos
            d.ctrl[rf] = pos

        run_phase(viewer, model, data, "GRIPPER — abrir y cerrar", gripper_open_close)

        # ── 5. LIDAR spin ───────────────────────────────────────────────────
        ls = _act_idx(model, "vel_lidar_spin")

        def lidar_spin(t, d):
            d.ctrl[ls] = 30.0

        run_phase(viewer, model, data, "LIDAR — rotación", lidar_spin)

        # ── 6. FLIPPER WHEELS 
        fw_names = [
            "vel_flip1_back", "vel_flip1_front",
            "vel_flip2_back", "vel_flip2_front",
            "vel_flip3_back", "vel_flip3_front",
            "vel_flip4_back", "vel_flip4_front",
        ]
        fw = [_act_idx(model, n) for n in fw_names]

        def flipper_to_ground(_t, d):
            for idx in fp:
                d.ctrl[idx] = 0.0

        run_phase(viewer, model, data,
                  "FLIPPERS — bajando a nivel del suelo (0 rad)",
                  flipper_to_ground, duration=2.0)

        def flipper_wheels(t, d):
            for idx in fp:        # mantiene flippers en suelo
                d.ctrl[idx] = 1
            for idx in fw:
                d.ctrl[idx] = 0.8

        run_phase(viewer, model, data, "FLIPPER WHEELS — rodillos en punta de flippers", flipper_wheels)

        # ── 7. BRAZO — posición "arm up" para ver la garra ─────────────────
        # Mueve el brazo suavemente a una pose vertical con la garra visible.
        # joint_1=0 (yaw neutro), joint_2 eleva el primer segmento,
        # joint_3 extiende hacia arriba, joints 4-6 orientan la muñeca.
        ARM_UP_TARGET = np.array([0.0, 1.4, -1.2, 0.0, 1.57, 0.0])
        INTERP_DURATION = 4.0

        def arm_to_up(t, d):
            alpha = min(t / INTERP_DURATION, 1.0)  # 0→1 suave
            for k, idx in enumerate(arm):
                d.ctrl[idx] = ARM_UP_TARGET[k] * alpha

        run_phase(viewer, model, data,
                  "BRAZO — moviendo a posición ARM-UP (garra visible)",
                  arm_to_up, duration=INTERP_DURATION + 1.0)

        # Mantiene la pose final con gripper abierto
        for k, idx in enumerate(arm):
            data.ctrl[idx] = ARM_UP_TARGET[k]
        data.ctrl[lf] = 0.03
        data.ctrl[rf] = 0.03

        print("\n✓ Todas las pruebas completadas.")
        print("  El brazo queda en posición ARM-UP con gripper abierto.")
        print("  Cierra el viewer para salir.\n")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
