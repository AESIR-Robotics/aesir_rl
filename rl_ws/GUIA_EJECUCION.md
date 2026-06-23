# Aesir RL — Guía definitiva de archivos y ejecución

## Archivos que necesitas

Todos van en la misma carpeta (`model_robot/`). Solo estos 6:

```
model_robot/
├── aesir_mujoco.xml          ← ya tienes este
├── ppo_conv_train.py         ← entrenamiento completo (cámaras + lidar + todo)
├── base_env.py               ← env solo para la base (opcional, para entrenar separado)
├── arm_env.py                ← env solo para el brazo (opcional, para entrenar separado)
├── combined_env.py           ← env combinado (opcional, para joint/separate/hierarchical)
└── train_base.py             ← trainer MLP rápido para la base (opcional)
```

**Si solo quieres empezar a entrenar ya:** solo necesitas `aesir_mujoco.xml` + `ppo_conv_train.py`.

---

## Dependencias entre archivos

```
ppo_conv_train.py   → no importa nada local (autónomo)
train_base.py       → importa base_env.py
combined_env.py     → importa base_env.py + arm_env.py
arm_env.py          → no importa nada local
base_env.py         → no importa nada local
```

---

## Instalación de librerías (una sola vez)

```bash
pip install mujoco torch numpy
pip install wandb   # opcional, para logging
```

---

## Opción A — Entrenamiento completo con cámaras (recomendado)

Un solo archivo, todo incluido. Usa las 3 cámaras + lidar + joint states.
Acciones: 14 valores semánticos con capa differential drive y brazo integrado.

```bash
cd model_robot

# Con render (viewer abierto, más lento):
MUJOCO_GL=egl python3 ppo_conv_train.py

# Sin render (puro entrenamiento, mucho más rápido):
MUJOCO_GL=egl python3 -c "from ppo_conv_train import train; train(render=False)"

# Reanudar desde el mejor checkpoint guardado:
MUJOCO_GL=egl python3 -c "
from ppo_conv_train import train
train(render=False, resume_from='checkpoints/ppo_conv_best.pt')
"
```

Los checkpoints se guardan automáticamente en `model_robot/checkpoints/`.

---

## Opción B — Entrenar solo la base (más rápido, sin cámaras)

Red MLP simple, solo lidar + estado. Converge mucho antes que con cámaras.
Útil para preentrenar la navegación antes de añadir el brazo.

```bash
cd model_robot

# Necesitas: base_env.py + train_base.py

python3 train_base.py

# O sin render:
python3 -c "from train_base import train; train(render=False)"
```

Los checkpoints se guardan en `model_robot/checkpoints_base/`.

---

## Opción C — Env combinado (joint / separate / hierarchical)

Para usar `combined_env.py` necesitas también `base_env.py` y `arm_env.py` en la misma carpeta.

```python
# Dentro de tu propio script de entrenamiento:
from combined_env import CombinedMuJoCoEnv

# Modo joint (una sola red, 14 acciones, reward compartida):
env = CombinedMuJoCoEnv("aesir_mujoco.xml", mode="joint", render=True)
obs = env.reset()            # np.ndarray (55,)
obs, rew, done, _ = env.step(np.zeros(14))

# Modo separate (dos redes independientes, mismo sim):
env = CombinedMuJoCoEnv("aesir_mujoco.xml", mode="separate")
obs = env.reset()            # {"base": ndarray(24,), "arm": ndarray(31,)}
obs, rew, done, _ = env.step({"base": np.zeros(6), "arm": np.zeros(8)})
# rew = {"base": float, "arm": float}

# Modo jerárquico (congelar base, entrenar solo brazo):
env = CombinedMuJoCoEnv("aesir_mujoco.xml", mode="joint", freeze_base=True)
```

---

## Qué hace cada acción

`ppo_conv_train.py` y `combined_env.py` usan 14 acciones en `[-1, 1]`:

| Índice | Qué controla | Cómo se aplica |
|--------|-------------|----------------|
| 0 | velocidad lineal base | differential drive → 6 ruedas vel_drive_* |
| 1 | velocidad angular base | differential drive → 6 ruedas vel_drive_* |
| 2..5 | posición flippers 1..4 | actuadores pos_flipper_* + sinc. rueditas |
| 6..11 | vel. articular joint_1..6 | integrador → pos_joint_* |
| 12 | dedo izquierdo | pos_left_finger |
| 13 | dedo derecho | pos_right_finger |

---

## Ajuste obligatorio antes de entrenar

En `ppo_conv_train.py` (y `base_env.py`), verifica estas dos constantes:

```python
TRACK_HALF_WIDTH = 0.21   # m — mide en tu XML: distancia_Y_entre_tracked_1_y_tracked_2 / 2
WHEEL_RADIUS     = 0.05   # m — radio de las ruedas vel_drive_* en MuJoCo
```

Para encontrar el valor correcto en tu XML:
- Busca el body `tracked_1` y `tracked_2` → anota su posición Y
- `TRACK_HALF_WIDTH = abs(Y_tracked_1 - Y_tracked_2) / 2`

Si estos valores están mal, el robot girará cuando debería ir recto.

---

## Errores comunes

**`ValueError: Actuador no encontrado: 'pos_flipper_1'`**
El nombre del actuador en tu XML es diferente. Busca en `aesir_mujoco.xml` los nombres
reales de tus actuadores y ajusta las listas `FLIPPERS`, `DRIVE_LEFT`, etc. al inicio del archivo.

**`ValueError: Sensor lidar_0 no encontrado`**
Igual, verifica que en tu XML los sensores se llaman `lidar_0` .. `lidar_6`.

**`MUJOCO_GL=egl` no funciona (sin GPU)**
Usa `MUJOCO_GL=osmesa` en su lugar, o simplemente `render=False` para no necesitar GL.

**El robot se mueve en circulos en lugar de recto**
`TRACK_HALF_WIDTH` o `WHEEL_RADIUS` incorrectos. Ver ajuste obligatorio arriba.

---

## Flujo recomendado de entrenamiento

```
1. Opción B: entrenar base sola (train_base.py)
   → rápido, converge en horas, aprende navegación básica

2. Opción A: entrenar todo junto (ppo_conv_train.py)
   → inicializar pesos del trunk de la base si quieres
   → aprende navegación + brazo + cámaras

3. Opcional: finetunear con combined_env.py modo jerárquico
   → freeze_base=True para perfeccionar el brazo sin romper la navegación
```
