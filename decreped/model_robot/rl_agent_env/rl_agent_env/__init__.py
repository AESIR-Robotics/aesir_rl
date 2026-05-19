"""rl_agent_env: sync/async bridge between PPO (PyTorch) and ROS 2 / MuJoCo."""

from rl_agent_env.ros_bridge import RosCommunicationNode
from rl_agent_env.rl_env import Env

__all__ = ["RosCommunicationNode", "Env"]

# Partes de tu robot que pueden chocar
piezas_robot = [
    "base_link", "tracked_1", "tracked_2", 
    "flipper_1_1", "flipper_2_1", "flipper_3_1", "flipper_4_1"
]

# Diccionario de misiones y colisiones (Banderas True/False)
estado_mision = {
    "puerta_desbloqueada": False,
    "paso_chicana_1": False,
    "paso_chicana_2": False,
    "choco_pallet_1": False
}