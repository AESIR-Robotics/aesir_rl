import mujoco
import mujoco.viewer
import cv2
import numpy as np
import time

# 1. Cargar el modelo
model = mujoco.MjModel.from_xml_path("aesir_mujoco.xml")
data = mujoco.MjData(model)

# 2. Crear el Renderizador para las cámaras
renderer = mujoco.Renderer(model, height=480, width=640)

# Configuración de FPS
FPS_DESEADO = 30
intervalo_render = 1.0 / FPS_DESEADO
ultimo_tiempo_render = 0.0
ultimo_tiempo_print = 0.0

# Encender el motor del Lidar
id_motor_lidar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel_lidar_spin")
data.ctrl[id_motor_lidar] = 20.0

# Obtener cutoff (usamos el rayo 0 de referencia)
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "lidar_0")
range_cutoff = model.sensor_cutoff[sensor_id] if sensor_id != -1 else 15.0

# Lanzar el visor pasivo de MuJoCo
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Activar la visualización de los rayos láser en el visor 3D
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    
    while viewer.is_running():
        # Avanzar las físicas
        mujoco.mj_step(model, data)
        tiempo_actual_sim = data.time
        
        # ---------------------------------------------------------
        # CONTROL DE TASA: Renderizar solo a 30 FPS
        # ---------------------------------------------------------
        if tiempo_actual_sim - ultimo_tiempo_render >= intervalo_render:
            ultimo_tiempo_render = tiempo_actual_sim

            # 1. Cámara de la Garra
            renderer.update_scene(data, camera="cam_gripper")
            img_gripper = cv2.cvtColor(cv2.flip(renderer.render(), -1), cv2.COLOR_RGB2BGR)

            # 2. Cámara Frontal
            renderer.update_scene(data, camera="cam_oakd")
            img_oakd = cv2.cvtColor(cv2.flip(renderer.render(), -1), cv2.COLOR_RGB2BGR)
            
            # 3. Cámara Trasera
            renderer.update_scene(data, camera="cam_back")
            img_back = cv2.cvtColor(cv2.flip(renderer.render(), -1), cv2.COLOR_RGB2BGR)

            # Visualización OpenCV
            cv2.imshow("Camara Gripper (Logitech)", img_gripper)
            cv2.imshow("Camara Frontal (OAK-D)", img_oakd)
            cv2.imshow("Camara Trasera (Logitech)", img_back)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            viewer.sync()

        # ---------------------------------------------------------
        # LECTURA DEL LIDAR (Matriz 4D)
        # ---------------------------------------------------------
        if tiempo_actual_sim - ultimo_tiempo_print >= 0.1:
            ultimo_tiempo_print = tiempo_actual_sim
            
            # Leer los 7 rayos
            lecturas = []
            for i in range(7):
                dist_raw = data.sensor(f"lidar_{i}").data[0]
                lecturas.append(dist_raw if 0 < dist_raw < range_cutoff else float('inf'))
            
            # lecturas[0] es el techo (+90°), lecturas[6] es el suelo (-6°)
            techo = lecturas[0]
            frente = lecturas[5] # +10° suele golpear obstáculos frontales a cierta distancia
            suelo = lecturas[6]
            
            # Formateo seguro para imprimir infinitos
            str_techo = f"{techo:.2f}m" if techo != float('inf') else "LIBRE"
            str_frente = f"{frente:.2f}m" if frente != float('inf') else "LIBRE"
            str_suelo = f"{suelo:.2f}m" if suelo != float('inf') else "LIBRE"

            print(f"L2 | Techo: {str_techo: >7} | Frente: {str_frente: >7} | Suelo: {str_suelo: >7}      ", end="\r")

cv2.destroyAllWindows()