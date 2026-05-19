import mujoco
import mujoco.viewer

# 1. Cargar el modelo directamente desde el archivo XML (.xml o .mjb)
# Cambia 'aesir_mujoco.xml' por la ruta de tu archivo si está en otra carpeta
modelo = mujoco.MjModel.from_xml_path('aesir_mujoco.xml')

# 2. Crear la estructura de datos de la simulación
data = mujoco.MjData(modelo)

print("¡Modelo MJCF cargado con éxito!")
print("Lanzando visor 3D. Presiona ESC en la ventana para salir.")

# 3. Lanzar el visor interactivo nativo de MuJoCo
mujoco.viewer.launch(modelo, data)