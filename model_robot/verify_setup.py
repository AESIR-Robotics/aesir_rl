#!/usr/bin/env python3
"""
Quick verification script to ensure all modules can be imported successfully
and paths are correctly configured.
"""

import sys
from pathlib import Path

# Add model_robot to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("VERIFICATION: Checking all module imports and paths")
print("=" * 60)

try:
    print("\n✓ Importing path_utils...")
    from path_utils import (
        get_project_root,
        get_model_robot_dir,
        get_checkpoint_dir,
        get_xml_path,
        get_meshes_dir,
        get_urdf_path,
        get_xml_model_path,
        validate_paths,
    )
    print("  Success!")

    print("\n✓ Validating all paths...")
    all_valid = validate_paths()
    
    if not all_valid:
        print("\n⚠ Warning: Some paths are missing!")
        sys.exit(1)
    
    print("\n✓ All paths are valid and accessible!")
    print(f"\n  Project root: {get_project_root()}")
    print(f"  Model robot dir: {get_model_robot_dir()}")
    print(f"  Checkpoint dir: {get_checkpoint_dir()}")

    print("\n" + "=" * 60)
    print("✓ All checks passed! Project is ready to run.")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error during verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
