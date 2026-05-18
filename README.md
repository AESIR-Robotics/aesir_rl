# Aesir — Entorno MuJoCo + PPO

Robot de rescate **Aesir** (orugas + 4 flippers + brazo 6-DOF + gripper) navegando una pista con puerta articulada en MuJoCo, entrenado con PPO en PyTorch.Se requiere instalación de ROS2 previa.

## Estructura

```
aesir/
├── aesir_scene.xml         # Escena unificada: robot + puerta + cámaras + LiDAR
├── adapters/
│   ├── __init__.py
│   ├── action_adapter.py   # ActionAdapter base (12-dim [-1,1] -> RobotAction)
│   ├── mujoco_adapter.py   # Bridge genérico ActionAdapter -> data.ctrl
│   └── aesir_adapter.py    # Subclase que maneja las 3 ruedas por lado
├── aesir_env.py            # Clase Env con reward de travesía de puerta
├── train_ppo.py            # Loop PPO + W&B
├── meshes/                 # STL placeholders (reemplazar por los reales)
└── README.md
```

## Instalación

```bash
pip install mujoco torch wandb mediapy matplotlib numpy
```

## Mallas (importante)

`aesir_scene.xml` usa `meshdir="meshes"` (relativo). Tienes dos opciones:

1. **Copia tus STLs reales** en `aesir/meshes/`, reemplazando los placeholders:
   ```
   base_link.stl, flipper_1_1.stl, flipper_2_1.stl, flipper_3_1.stl,
   flipper_4_1.stl, lidar.stl, logitech_c920.stl, tracked_1.stl, tracked_2.stl
   ```

2. **Apunta a tu directorio original** editando la línea `<compiler meshdir="..."/>` en `aesir_scene.xml` (p. ej. `/home/kfcnef/AESIR/arm/workspace/src/aesir_robot_description/meshes`).

Las mallas son **visuales** (`contype=0 conaffinity=0`); la simulación física usa primitivas (boxes/cylinders/spheres) así que la dinámica funciona aún con los placeholders.

## Escena (aesir_scene.xml)

**Robot:** `nq=29`, `nv=28`, `nu=22`. Layout `qpos`:
```
[0:7]   base_freejoint (x,y,z, qw,qx,qy,qz)
[7]     flipper_joint_1 (trasera-izq)
[8:10]  drive_flip1_back / drive_flip1_front (mini-ruedas en flipper 1)
[10]    flipper_joint_2 (trasera-der)
[11]    flipper_joint_3 (delantera-der)
[12]    flipper_joint_4 (delantera-izq)
[13:19] joint_1..joint_6 (brazo)
[19:21] left_finger_joint, right_finger_joint
[21:24] drive_l_1..3 (ruedas izquierdas)
[24:27] drive_r_1..3 (ruedas derechas)
[27]    door_hinge   (bisagra de la puerta)
[28]    handle_hinge (manija)
```

**Cámaras (3):**
* `cam_overview` — bird's-eye fijo, ve toda la escena
* `cam_door` — apunta al robot desde el marco de la puerta (`mode="targetbody"`)
* `cam_chase` — anclada al robot, lo persigue **(default para video)**

**LiDAR (17 rangefinders):**
* `lidar_distant` — rayo largo definido en el `<site lidar_origin>` del robot original
* `lidar_ring_00..15` — anillo de 16 rayos cada 22.5° (escaneo 2D)

Las lecturas se concatenan en `data.sensordata` (vector de 17 elementos).
Valor `-1` (sin impacto) se reemplaza por `LIDAR_MAX = 10.0 m` en `Env._get_state()`.

**Puerta:** marco fijo (postes + dintel + paredes laterales) y una hoja con bisagra Z en `x=3.0 m`, frente al robot. Manija decorativa con su propia bisagra X.

## Adapters

`AesirMujocoAdapter(MujocoAdapter)` — subclase para Aesir:
* Sobrescribe `_cache_indices()` para guardar **listas** de IDs (`wheels_left`, `wheels_right`) con los 3 actuadores velocity de cada lado (`vel_drive_l_1/2/3`, `vel_drive_r_1/2/3`).
* Sobrescribe `_apply_base()` para escribir la misma `omega` a las 3 ruedas del lado correspondiente.
* `track_width=0.40 m`, `wheel_radius=0.10 m` (ajustar si tu geometría real difiere).
* Hereda `_apply_flippers` y `_apply_arm` del adapter base (los actuadores `flipper_0_act..3` y `joint_1_act..6` ya están definidos en la escena con la convención esperada).

## Función de recompensa (`aesir_env.py`)

Por substep (5 substeps por env step):
* `+5.0 * progress` — reducción de distancia a la puerta (mientras `x < DOOR_X`)
* `+2.0 * max(vx, 0)` — velocidad hacia adelante después de cruzar
* `+0.5 * |door_hinge|` — bono por abrir la puerta

Por env step (no por substep, para no saturar):
* `-0.02 * (1 - cos(yaw))` — penalización ligera por desorientación
* `-1e-5 * ||qvel||²` — penalización mínima de energía

Terminal:
* `-100` si cae (`z < 0.05`) o vuelca (componente Z del eje-Z del cuerpo en el mundo `< 0.3`)
* `+200` y `done` si cruza (`x > 4.0`)
* `done` al cumplirse `DURATION = 8 s`

Valores de prueba del reward landscape:
| Política              | x final | reward total |
|-----------------------|---------|--------------|
| Forward (a[0]=+1)     | 2.11    | +6.5         |
| Idle (zeros)          | 0.00    | ~0           |
| Turn-only (a[1]=+1)   | -0.02   | -0.4         |
| Backward (a[0]=-1)    | -3.19   | -16.8        |

El gradiente del reward favorece avanzar hacia la puerta como se espera.

## PPO (`train_ppo.py`)

Cambios sobre la versión ANYmal:
* `OBS_LEN = 74` (antes 37), `ACT_LEN = 12` (igual).
* `compute_action()` **sin** los clamps duros del ANYmal:
  ```python
  # REMOVED:
  # action_clamped[0][0] = action_clamped[0][0]*0.6-0.1
  # action_clamped[0][3] = action_clamped[0][3]*0.6+0.1
  # ...
  ```
  Ahora regresa `tanh(action)` limpio en `[-1, 1]`, lo cual deja que `ActionAdapter.from_policy_output()` aplique sus propios escalamientos físicos (`MAX_LINEAR_VEL`, `MAX_FLIPPER_POS`, etc).
* `transition` dtype reformateado con shape `(74,)` para `s` y `s_`.
* `PROJECT = "AIDL-PPO-AESIR"`.
* Umbral de guardado `SAVE_REWARD = 50.0` (reward total típico del éxito).
* `target_reward = 300.0` para detener antes de los 15000 episodios.

Hiperparámetros base (`gamma=0.99`, `lr=1e-5`, `clip=0.1`, `ppo_epoch=48`, `batch_size=128`, `c1=1.0`, `c2=0.001`, `std_init=1.0 → std_min=0.6` lineal) se conservaron.

## Ejecutar

```bash
# Login en W&B (una sola vez por máquina)
wandb login

# Entrenamiento estándar
cd aesir
python train_ppo.py

# Sweep bayesiano (descomentar el bloque al final de train_ppo.py)
```

Mientras entrena, W&B registra:
* `avg_reward`, `policy_loss`, `value_loss`, `avg_entropy`, `ratio`
* Cuando `running_reward > SAVE_REWARD`: guarda `.pt` del policy + optimizer y graba un video `video_<episode>_<reward>.mp4` desde `cam_chase`.

## Notas para producción

* Si tienes GPU, descomenta la línea `device = torch.device(...)` y mueve `policy` y los tensores con `.to(device)`.
* Para visualizar la escena en MuJoCo Viewer: `python -m mujoco.viewer --mjcf aesir_scene.xml`
* `wheel_radius` y `track_width` en `AesirMujocoAdapter.__init__` deben coincidir con tu robot físico real para que `MAX_LINEAR_VEL=0.5 m/s` del adapter mapee correctamente.
* La pista actual es **una sola puerta**. Para agregar la pista completa (`MAZE.xacro`, `PALLETS.xacro`, etc.), inserta nuevos `<body>` en `aesir_scene.xml` después del bloque `door_frame`.
