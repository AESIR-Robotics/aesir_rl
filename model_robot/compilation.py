import mujoco
import mujoco.viewer  # Importante: Importar el visor gráfico

# 1. Leer el archivo URDF como texto
with open('model_robot/assets/aesir_puro.urdf', 'r') as f:
    urdf_string = f.read()

# Reemplaza esto con tu ruta real (lo que ya habías hecho)
ruta_absoluta_meshes = "model_robot/assets/meshes/"
urdf_string = urdf_string.replace('package://aesir_robot_description/meshes/', ruta_absoluta_meshes)

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