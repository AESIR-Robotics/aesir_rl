import mujoco
import mujoco.viewer

from path_utils import get_xml_model_path

# 1. Cargar el modelo directamente desde el archivo XML (.xml o .mjb)
xml_path = get_xml_model_path()
modelo = mujoco.MjModel.from_xml_path(str(xml_path))

# 2. Crear la estructura de datos de la simulación
data = mujoco.MjData(modelo)

print("¡Modelo MJCF cargado con éxito!")
print("Lanzando visor 3D. Presiona ESC en la ventana para salir.")

# 3. Lanzar el visor interactivo nativo de MuJoCo
mujoco.viewer.launch(modelo, data)