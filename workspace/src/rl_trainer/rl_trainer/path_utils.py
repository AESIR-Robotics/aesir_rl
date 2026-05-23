"""
Utility module for managing project paths in rl_trainer.
Resolves paths relative to the project root directory (aesir_rl/).
"""

from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory (aesir_rl/)."""
    # This script is in workspace/src/rl_trainer/rl_trainer/
    # Go up 4 levels to reach aesir_rl/
    return Path(__file__).parent.parent.parent.parent.parent

def get_xml_path() -> Path:
    """Get the path to the MuJoCo XML scene file."""
    return get_project_root() / "workspace" / "src" / "aesir_robot_description" / "launch" / "aesir_complete.xml"

def get_checkpoint_dir() -> Path:
    """Get the checkpoints directory in model_robot."""
    ckpt_dir = get_project_root() / "model_robot" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir

if __name__ == "__main__":
    print("Project root:", get_project_root())
    print("XML path:", get_xml_path())
    print("Checkpoint dir:", get_checkpoint_dir())
