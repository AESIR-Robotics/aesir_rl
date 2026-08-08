# Evolución del proyecto AESIR-RL

## Fase 0 — Setup y modelo del robot

Arranque del repo: estructura del proyecto, modelo URDF/MuJoCo del robot (base con
4 orugas + 4 flippers + brazo de 6 DOF), colisionadores, mapas de pallets. Boceto
inicial MoveIt+MuJoCo.

## Fase 1 — Brazo: PPO + MoveIt + MuJoCo

Primer pipeline de RL: cinemática del brazo funcionando, controller manager del
gripper, conexión PPO ↔ MoveIt ↔ MuJoCo por ROS2. Bridge MuJoCo-ROS2 con interfaz
de hardware realista (límites de velocidad/fuerza).

## Fase 2 — Navegación global: A\* + vortex APF

`global_navigator.py` nace aquí: A\* 2D sobre grilla + campo de potencial vortex
para el seguimiento local. Primera versión del mapa de pallets.

## Fase 3 — Pipeline base completo (ROS)

Conexión política↔navegación↔bridge MuJoCo-ROS2 armada de punta a punta. Primer
entrenamiento real de la base. Reorganización grande de tests/utils/modelo.
Correcciones de reward y de las oscilaciones del vortex.

## Fase 4 — Pipeline sin ROS (el que se volvió `base_training/`)

Salto importante: entrenamiento **directo en MuJoCo**, sin el bridge ROS2 de por
medio → paralelización por cores de CPU, mucho más rápido que tiempo real.
Métricas y plot en vivo del mapa con datos del robot durante el entrenamiento
(precursor del `LiveTrajectoryPlot` retomado más adelante).

## Fase 5 — Navegación multi-pista y arquitectura de acción rica

Esta fase (reflejada también en `docs/paper_sii.md`) es la más sofisticada del
historial:

- Corrección del bug de "robot atascado siguiendo waypoints" y de A\* considerando
  obstáculos desde el inicio (Dijkstra/A\* con obstáculos previos, no post-hoc).
- **Multi-pista**: entrenamiento sobre varias pistas (`tracks/`), mapas de
  ocupación generados por pista.
- **Mapa de elevación robot-céntrico** (heatmap) + reward de flippers
  **condicionado al terreno**: criterio geométrico basado en `d_edge`/`h_edge`
  (la punta del flipper debe superar el borde en alcance y altura, para subir o
  bajar).
- **Espacio de acción heterogéneo**: Normal recortada para orugas, **Beta** para
  flippers (soporte acotado exacto, sin desajuste clip/log-prob), y un **gate
  Bernoulli** para decidir si usar los flippers o "aparcarlos" en terreno fácil.
- Experimento con **SAC** en paralelo a PPO (solo de prueba).
- **Value normalization** (`RunningMeanStd`) — corrigió que el gradiente del
  crítico ahogaba al de la política a través del trunk compartido (~630× más
  gradiente a la cabeza de flippers tras el fix).
- Métricas de éxito por pista + generales, barrido de semillas
  (`--seed`/`--ckpt-dir`) y analizador de logs.

## Fase 6 — Pipeline "etapa 2" (rápido, ROS-compatible)

Rama más simple y desplegable del pipeline base, en paralelo a la Fase 5:

1. **`base_training/`** nuevo: `base_env.py` (world_base, lógica pura de tarea),
   `robot_control.py` (rampas AVR446 sin ROS), `mujoco_sim_base.py`
   (`VecMujocoEnv` por threads, ~5× paralelismo real porque `mj_step` libera el
   GIL), `train_fast.py` (PPO vectorizado en GPU).
2. **Fidelidad de despliegue**: mismas rampas de actuador, mismo `world_base`,
   mismo 20 Hz de control en el backend rápido y en `base_ros_env.py` (ROS) — la
   política entrenada rápido es la misma que corre en el bridge.
3. **Reward iterado varias veces** hasta encontrar y cerrar un exploit: el robot
   descubrió que oscilar (avanzar y retroceder) farmeaba recompensa densa sin
   avanzar. Se cerró con un reward simple: velocidad deseada = **distancia al
   punto-guía del vortex** (lejos→rápido, cerca→lento) + castigo directo por
   retroceder.
4. **Lookahead de trayectoria**: la observación pasó de 15→30 dims, agregando 5
   puntos futuros muestreados sobre la ruta A\* (no sobre el vortex, que zigzaguea
   por el swirl).
5. **Límites de flipper** por software, asimétricos `[-1.3, 3.14159]` rad.
6. **Castigo por aceleraciones fuertes** del chasis (integridad del robot en
   terreno difícil).
7. **Herramientas de depuración en `test_base.py`**: desglose del reward por
   componente (tabla con % y /paso), y una ventana matplotlib **en vivo** que
   reutiliza `draw_map` de `plot_path_vortex.py` para mostrar la pista real +
   trayectoria del robot + guía inmediata + lookahead mientras corre el episodio.

## Estado actual

`docs/paper_sii.md` describe la arquitectura rica de la Fase 5 (heatmap,
Beta+gate, reward de flippers por terreno). El pipeline de la Fase 6
(`base_training/`) es la misma línea de trabajo, no una rama paralela:
**ambas convergen** — `base_training/` es la base desplegable (world_base,
rampas AVR446, 20 Hz, fiel al bridge ROS) sobre la que se espera reintroducir
progresivamente heatmap, Beta+gate y el reward de flippers por terreno de la
Fase 5, en vez de mantenerlas como dos arquitecturas separadas.
