"""
test_base.py — Visualiza el modelo entrenado del robot base.
"""
import torch
import time
import numpy as np
from pathlib import Path

# Importamos tu entorno y la arquitectura de la red
from base_env import BaseMuJoCoEnv
from train_base import ConvActorCritic, obs_to_tensor, XML_PATH

# ── Configuración ───────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints_base/base_best.pt"  # El mejor modelo guardado
DETERMINISTIC = True  # True = usa el promedio exacto, False = añade ruido/exploración

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Iniciar el entorno CON renderizado
    env = BaseMuJoCoEnv(XML_PATH, render=True)
    
    # 2. Reconstruir la red neuronal (con las mismas dimensiones)
    policy = ConvActorCritic(
        image_shape=env.image_shape,
        lidar_dim=env.num_lidar,
        joint_dim=env.joint_len,
        act_dim=env.act_len,
    ).to(device)

    # 3. Cargar los pesos entrenados
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ No se encontró el archivo: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    print(f"✅ Modelo cargado exitosamente desde el episodio (Iter: {checkpoint.get('iter', '?')}) con Recompensa Promedio: {checkpoint.get('avg_ep_r', '?'):.2f}")

    # Poner la red en modo evaluación (desactiva dropout, etc.)
    policy.eval()

    obs = env.reset()
    ep_reward = 0.0

    print("🎮 Iniciando simulación... (Presiona Ctrl+C en la terminal para salir)")
    try:
        while True:
            # Convertir observaciones a tensores de PyTorch
            obs_t = obs_to_tensor(obs)
            
            with torch.no_grad():
                images = obs_t["images"].unsqueeze(0).to(device)
                lidar  = obs_t["lidar"].unsqueeze(0).to(device)
                joints = obs_t["joint_states"].unsqueeze(0).to(device)
                
                # Obtener la acción de la red
                mu, std, _ = policy(images, lidar, joints)
                
                if DETERMINISTIC:
                    # Toma la decisión más segura (sin ruido)
                    action = mu.squeeze(0).cpu().numpy()
                else:
                    # Muestrea de la distribución (igual que en entrenamiento)
                    dist = torch.distributions.Normal(mu, std)
                    action = dist.sample().squeeze(0).cpu().numpy()

            # Ejecutar la acción en MuJoCo
            obs, rew, done, _ = env.step(action)
            ep_reward += rew

            # Pausa pequeña para que tus ojos puedan seguir el movimiento a tiempo real
            time.sleep(0.02) 

            if done:
                print(f"🏁 Episodio terminado. Recompensa total: {ep_reward:.2f}")
                obs = env.reset()
                ep_reward = 0.0

    except KeyboardInterrupt:
        print("\n🛑 Simulación detenida por el usuario.")
    finally:
        env.close()

if __name__ == "__main__":
    test()