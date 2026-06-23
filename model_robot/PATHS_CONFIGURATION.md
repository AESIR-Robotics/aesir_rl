# Rutas Relativas - Configuración del Proyecto

## Cambios Realizados

Se han corregido todos los paths hardcoded (específicos del usuario) a paths relativos usando un sistema de utilidades centralizado.

### Archivos Modificados

1. **`model_robot/path_utils.py`** (NUEVO)
   - Módulo centralizado para manejar todas las rutas del proyecto
   - Proporciona funciones como:
     - `get_project_root()`: Directorio raíz del proyecto
     - `get_xml_path()`: Ruta al archivo XML de la escena MuJoCo
     - `get_checkpoint_dir()`: Directorio de checkpoints
     - `get_meshes_dir()`: Directorio de meshes
     - `get_urdf_path()`: Ruta al archivo URDF
     - `validate_paths()`: Valida que todos los paths existan

2. **`model_robot/ppo_conv_train.py`**
   - Antes: `XML_PATH = "../workspace/src/aesir_robot_description/launch/aesir_complete.xml"`
   - Después: `XML_PATH = str(get_xml_path())`
   - Antes: `CHECKPOINT_DIR = Path("./checkpoints")`
   - Después: `CHECKPOINT_DIR = get_checkpoint_dir()`

3. **`model_robot/compilation.py`**
   - Antes: `ruta_absoluta_meshes = "/home/kfcnef/AESIR/arm/workspace/src/aesir_robot_description/meshes/"`
   - Después: `meshes_dir = get_meshes_dir()`
   - Antes: `with open('aesir_puro.urdf', 'r') as f:`
   - Después: `urdf_path = get_urdf_path()` y `with open(urdf_path, 'r') as f:`

4. **`model_robot/load.py`**
   - Antes: `modelo = mujoco.MjModel.from_xml_path('aesir_mujoco.xml')`
   - Después: `modelo = mujoco.MjModel.from_xml_path(str(get_xml_model_path()))`

5. **`model_robot/sensors.py`**
   - Antes: `model = mujoco.MjModel.from_xml_path("aesir_mujoco.xml")`
   - Después: `model = mujoco.MjModel.from_xml_path(str(get_xml_model_path()))`

6. **`model_robot/test_actuators.py`**
   - Antes: `XML_PATH = "../workspace/src/aesir_robot_description/launch/aesir_complete.xml"`
   - Después: `XML_PATH = str(get_xml_path())`

7. **`workspace/src/rl_trainer/rl_trainer/path_utils.py`** (NUEVO)
   - Módulo similar para la carpeta rl_trainer
   - Proporciona las mismas funciones ajustadas para su ubicación

8. **`workspace/src/rl_trainer/rl_trainer/train_ppo.py`**
   - Antes: `XML_PATH = "../workspace/src/aesir_robot_description/launch/aesir_complete.xml"`
   - Después: `XML_PATH = str(get_xml_path())`
   - Antes: `CHECKPOINT_DIR = Path("./checkpoints")`
   - Después: `CHECKPOINT_DIR = get_checkpoint_dir()`

## Cómo Ejecutar el Proyecto

### 1. Verificación Inicial
```bash
cd ./model_robot
python3 verify_setup.py
```

### 2. Ejecutar los Scripts

#### Desde model_robot/:
```bash
cd ./model_robot

# Entrenar el modelo PPO
MUJOCO_GL=egl python3 ppo_conv_train.py

# Cargar un modelo MuJoCo
python3 load.py

# Compilar modelo desde URDF
python3 compilation.py

# Probar actuadores
MUJOCO_GL=egl python3 test_actuators.py

# Leer sensores
python3 sensors.py
```

#### Desde workspace/src/rl_trainer/rl_trainer/:
```bash
cd ./workspace/src/rl_trainer

# Entrenar usando el módulo rl_trainer
MUJOCO_GL=egl python3 -m rl_trainer.train_ppo
```

## Ventajas del Sistema de Paths Relativo

✓ **Portabilidad**: El proyecto funciona desde cualquier ubicación
✓ **Flexibilidad**: No depende de usuarios específicos (`/home/kfcnef/`)
✓ **Mantenibilidad**: Cambios centralizados en `path_utils.py`
✓ **Validación**: Función `validate_paths()` para debugging
✓ **Consistencia**: Todos los módulos usan el mismo sistema

## Validación

Se ha creado el script `verify_setup.py` que verifica:
- Que todos los paths existan
- Que los módulos puedan importarse
- Que el proyecto esté listo para ejecutarse

Run it anytime to ensure everything is configured correctly:
```bash
python3 ./model_robot/verify_setup.py
```

## Estructura de Rutas

```
./
├── model_robot/
│   ├── path_utils.py          ← Configuración de paths
│   ├── ppo_conv_train.py      ✓ Actualizado
│   ├── compilation.py          ✓ Actualizado
│   ├── load.py                 ✓ Actualizado
│   ├── sensors.py              ✓ Actualizado
│   ├── test_actuators.py       ✓ Actualizado
│   ├── verify_setup.py         (nuevo)
│   ├── checkpoints/            (generado automáticamente)
│   └── aesir_mujoco.xml
├── workspace/
│   └── src/
│       ├── aesir_robot_description/
│       │   ├── launch/
│       │   │   └── aesir_complete.xml
│       │   └── meshes/
│       └── rl_trainer/
│           └── rl_trainer/
│               ├── path_utils.py    ✓ (nuevo)
│               └── train_ppo.py     ✓ Actualizado
└── ...
```
