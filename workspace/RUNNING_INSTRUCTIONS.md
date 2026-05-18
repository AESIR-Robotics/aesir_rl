# Instrucciones de Ejecución (Run Project)

Sigue estos pasos para compilar, configurar el entorno y ejecutar la simulación y el entrenamiento PPO de los agentes concurrentes de RL para el Aesir Robot.

## 1. Requisitos y Preparación Inicial

Asegúrate de que estás activando tu entorno virtual de Python. Los paquetes ROS están diseñados bajo la estructura estándar de `colcon`:

```bash
# 1. Activación del virtualenv con las dependencias Python requeridas (PyTorch, Mujoco, etc.)
source /home/didier/aesir_rl/.venv/bin/activate

# 2. Navegar al entorno de trabajo e invocar setup base 
cd /home/didier/aesir_rl/workspace
source /opt/ros/humble/setup.bash  # (o versión de ROS respectiva a tu distribución)
```

## 2. Construcción (Build) de Paquetes ROS 2

Para compilar los paquetes (como `rl_agent_env` y `rl_trainer` y cualquier descriptor del robot):

```bash
colcon build --symlink-install
```

Si es exitoso, asegúrate de activar el setup generado:

```bash
source install/setup.bash
```

## 3. Ejecución del Entrenamiento

Actualmente, puedes inicializar el trainer directamente. El script principal de entrenamiento en `rl_trainer/train_ppo.py` instancia `HybridAesirEnv`, que puede levantar internamente la simulación MuJoCo y enlazar las publicaciones asíncronamente con ROS (ver el argumento lógico `use_ros`).

Tienes dos formas habituales de correr el proyecto:

### A) Lanzamiento Completo Directo (Recomendado para Debug de RL)

Puedes ejecutar el script principal de Python de PPO y dejar que éste instancie todo. Solo asegúrate de estar parado donde el script lo encuentra o instanciarlo mediante su módulo:

```bash
# Correr el entrenador directamente (Asegúrate de que dependencias como ROS y wandb si está activo, estén corriendo o loggeadas).
python3 src/rl_trainer/rl_trainer/train_ppo.py
```

### B) Vía ROS 2 Launch Files

Existen archivos `.launch.py` provistos en el paquete `rl_agent_env` o `aesir_robot_description` para prearrancar infraestructura auxiliar (RViz, URDF state publishers) requerida si deseas inspeccionar temas desde el exterior o validarlo con MoveIt.

```bash
# Ejemplo, correr algún launcher existente de ambiente y luego el agente:
ros2 launch rl_agent_env train_agents.launch.py
```
*(Nota: Si usas ROS Launch, verifica que los paths hacia pesos `.pt` de modelo en checkpoints sean relativos al directorio de arranque, que tiende a ser la raíz del workspace `~/aesir_rl/workspace/`).*

## Detección Práctica de Errores

* **Sensor X no encontrado (`lidar_0`)**: El modelo XML de MuJoCo (`aesir_complete.xml` en `aesir_robot_description`) necesita coincidir estructuralmente con las solicitudes del entorno `HybridAesirEnv`. Verifica que las definiciones base estén presentes en el xml.
* **Topic Warnings (`No se recibió /clock`)**: Al entrenar con `ros_bridge`, el reloj necesita fluir apropiadamente. Si MuJoCo no hace el forward del tiempo de simulación hacia ROS, tus controladores pueden colapsar o congelarse.