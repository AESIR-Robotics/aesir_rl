import mujoco
import mujoco.viewer

modelo = mujoco.MjModel.from_xml_path('model_robot/assets/aesir_mujoco.xml')

data = mujoco.MjData(modelo)

print("¡Modelo MJCF cargado con éxito!")
print("Lanzando visor 3D. Presiona ESC en la ventana para salir.")

mujoco.viewer.launch(modelo, data)