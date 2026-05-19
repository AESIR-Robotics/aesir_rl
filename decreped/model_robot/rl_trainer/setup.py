"""ament_python setup for the rl_trainer package."""
from setuptools import find_packages, setup

package_name = "rl_trainer"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="didier",
    maintainer_email="edidier11@outlook.com",
    description="PPO trainer (PyTorch) for the Aesir robot via rl_agent_env.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_ppo = rl_trainer.train_ppo:main",
        ],
    },
)
