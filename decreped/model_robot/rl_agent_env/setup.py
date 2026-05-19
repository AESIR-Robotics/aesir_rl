"""ament_python setup for the rl_agent_env package.

Installs:
    * The `rl_agent_env` Python module (ros_bridge.py, rl_env.py).
    * The launch files under `launch/` so they are reachable via
      `ros2 launch rl_agent_env <file>.launch.py`.
    * Optional config files under `config/`.
"""
import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rl_agent_env"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="didier",
    maintainer_email="edidier.iew@gmail.com",
    description=(
        "ROS 2 bridge between a synchronous PPO trainer and an asynchronous "
        "MuJoCo simulation driven by mujoco_ros2_control + MoveIt Servo."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Launches the Env standalone (useful for smoke tests / launch file).
            "rl_env = rl_agent_env.rl_env:main",
        ],
    },
)
