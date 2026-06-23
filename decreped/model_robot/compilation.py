import mujoco
import mujoco.viewer  # Importante: Importar el visor gráfico

from path_utils import get_urdf_path, get_meshes_dir

# 1. Leer el archivo URDF como texto
urdf_path = get_urdf_path()
with open(urdf_path, 'r') as f:
    urdf_string = f.read()

# Reemplaza los paths de meshes con rutas correctas
meshes_dir = get_meshes_dir()
urdf_string = urdf_string.replace('package://aesir_robot_description/meshes/', str(meshes_dir) + '/')

# 2. Compilar el modelo
modelo = mujoco.MjModel.from_xml_string(urdf_string)
print("¡Modelo cargado con éxito!")

# --- LO NUEVO EMPIEZA AQUÍ ---

# 3. Crear la estructura de Datos (estado actual de la simulación)
data = mujoco.MjData(modelo)

# Avanzar la simulación un paso inicial para que todo se acomode
mujoco.mj_step(modelo, data)

print("Lanzando visor 3D. Presiona ESC en la ventana para salir.")

# 4. Lanzar el visor interactivo nativo de MuJoCo
mujoco.viewer.launch(modelo, data)