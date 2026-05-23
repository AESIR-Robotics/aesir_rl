"""
Utility module for managing project paths consistently.
Resolves paths relative to the project root directory.
"""

from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory (aesir_rl/)."""
    # This script is in model_robot/, so go up one level
    return Path(__file__).parent.parent

def get_model_robot_dir() -> Path:
    """Get the model_robot directory."""
    return Path(__file__).parent

def get_checkpoint_dir() -> Path:
    """Get the checkpoints directory."""
    ckpt_dir = get_model_robot_dir() / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir

def get_xml_path() -> Path:
    """Get the path to the MuJoCo XML scene file."""
    return get_project_root() / "workspace" / "src" / "aesir_robot_description" / "launch" / "aesir_complete.xml"

def get_meshes_dir() -> Path:
    """Get the path to the meshes directory."""
    return get_project_root() / "workspace" / "src" / "aesir_robot_description" / "meshes"

def get_urdf_path() -> Path:
    """Get the path to the URDF file."""
    return get_model_robot_dir() / "aesir_puro.urdf"

def get_xml_model_path() -> Path:
    """Get the path to the aesir_mujoco.xml file."""
    return get_model_robot_dir() / "aesir_mujoco.xml"

def validate_paths() -> bool:
    """Validate that all critical paths exist."""
    paths = [
        ("XML Scene", get_xml_path()),
        ("Meshes Directory", get_meshes_dir()),
        ("URDF File", get_urdf_path()),
        ("XML Model", get_xml_model_path()),
    ]
    
    all_exist = True
    for name, path in paths:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("Project paths validation:")
    print(f"Project root: {get_project_root()}")
    print(f"Model robot dir: {get_model_robot_dir()}")
    print(f"Checkpoint dir: {get_checkpoint_dir()}")
    print()
    validate_paths()
