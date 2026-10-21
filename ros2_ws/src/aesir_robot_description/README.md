# aesir_robot_description

## What it is

ROS2 package (`ament_cmake`) with the URDF/xacro description of the AESIR
rescue robot: tracked chassis, 4 flippers, 6-DOF arm with a two-finger
gripper, and the sensors mounted on the chassis and the arm's wrist (LiDAR,
OAK-D stereo camera, two Logitech C920 cameras). It contains no nodes of its
own: it exposes the `.xacro` files, the `.stl` meshes, and a couple of
convenience `launch` files to visualize the model with
`robot_state_publisher` + `joint_state_publisher_gui` + RViz2. The model is
also reused from `robot_moveit_config` (via `xacro:include`) to add the
`ros2_control` tags needed for MoveIt2.

## Package layout

```
aesir_robot_description/
  urdf/       rescue_robot.urdf.xacro, rescue_robot_v2.urdf.xacro,
              rescue_robot_foxglove.urdf.xacro, arm.urdf.xacro
  meshes/     *.stl for the chassis, flippers and sensors
  launch/     robot.launch.py, robot_arm.launch.py
  CMakeLists.txt, package.xml
```

`CMakeLists.txt` only installs the `urdf`, `meshes` and `launch` folders
into `share/aesir_robot_description` (`install(DIRECTORY urdf meshes launch
...)`); it doesn't build any executable. `find_package(ament_cmake
REQUIRED)` is the only declared dependency — see [Notes](#notes).

## Model variants (urdf/*.xacro)

| File | Use | Confirmed in code |
|---|---|---|
| `rescue_robot_v2.urdf.xacro` | **Current/canonical** model: loaded by `robot.launch.py` and imported by `robot_moveit_config/config/rescue_robot.urdf.xacro` via `xacro:include` to add `ros2_control`. Chassis with track meshes (`tracked_1.stl`/`tracked_2.stl`), flippers as `continuous` joints, real OAK-D mesh. | Yes |
| `rescue_robot.urdf.xacro` | Version predating v2 (by commit order: v2 came later, adding track meshes, a real OAK-D mesh, and more simulated track wheels — commit `2bdf7c27`). Flippers as `revolute` with `±3.14` limits, no track meshes, OAK-D as a placeholder box. No launch file in this package uses it. | Partial (v1→v2 relation inferred from git history, not an explicit comment) |
| `rescue_robot_foxglove.urdf.xacro` | Presumably meant for exporting/visualizing in Foxglove Studio (by name), but **no launch file or script in the repo references it**. References meshes `base.stl`, `flipper_1.stl`…`flipper_4.stl` with `file://$(find aesir_robot_description)/meshes/...` syntax — those filenames **don't exist** in `meshes/` (which only has `base_link.stl`, `flipper_1_1.stl`…`flipper_4_1.stl`); as it stands, loading it would fail to resolve the meshes. Includes no sensors (no lidar/OAK-D/cameras). | Yes (mesh names verified against the actual contents of `meshes/`) |
| `arm.urdf.xacro` | **Standalone** 6-DOF arm, with its own `world` link and a cylindrical `base_link` independent of the chassis (no flippers, chassis, or a finished gripper — it does have an empty `tool_link` at the end). Same kinematic tree (`joint_1`…`joint_6`, same dimensions `H_BASE`, `L_UPPER`, `L_FORE`, etc.) as the arm section of `rescue_robot*.urdf.xacro`, but without the gripper. Loaded by `robot_arm.launch.py`. | Yes |

No macros (`xacro:macro`) are used in any file: each variant repeats the
full link/joint tree, with `xacro:property` for the arm's shared dimensions.
There's no `xacro:arg` in any of the four files.

## Main links / joints

Based on `rescue_robot_v2.urdf.xacro` (the canonical model):

- **Chassis**: `base_footprint` (fixed, `-0.03` m Z offset relative to
  `base_link`) → `base_link` (mesh `base_link.stl`, mass 25 kg, with
  explicit inertia). It has no `world` link, unlike `rescue_robot.urdf.xacro`
  and `rescue_robot_foxglove.urdf.xacro`, which do define `world` +
  `world_to_base` as a fixed root.
- **Tracks**: `tracked_1_link` and `tracked_2_link`, attached to
  `base_link` with fixed joints named `Rigid 1` and `Rigid 2` (names with a
  space, as they appear in the xacro).
- **Flippers**: 4 links (`flipper_1_link`…`flipper_4_link`) attached to
  `base_link` via `continuous` joints (`flipper_1_joint`…`flipper_4_joint`,
  axes on Y). **Anomaly found in the xacro**: `flipper_3_joint` has
  `flipper_4_link` as its `child`, and `flipper_4_joint` has
  `flipper_3_link` as its `child` — flipper 3/4's joint and link names are
  swapped. Whether this is intentional or a naming bug isn't confirmed;
  reported as it appears in the code.
- **Arm (6 DOF)**: mounted on `base_link` via `arm_mount_joint` (fixed) →
  `arm_base_link` → `joint_1`…`joint_6` (all `revolute`, limits `±3.14`
  rad; `joint_1`-`joint_3` with `effort=150`, `joint_4`-`joint_6` with
  `effort=50`; all with `velocity=3.14`) → `link_1`…`link_6` →
  `tool_joint` (fixed) → `tool_link`.
- **Gripper**: `gripper_base_link` (fixed to `link_6`) with two fingers
  `left_finger_link`/`right_finger_link` on `prismatic` joints
  (`left_finger_joint`/`right_finger_joint`, range `0`–`0.03` m,
  `effort=50`, `velocity=0.5`).
- **Sensors** (fixed to `base_link`, except the arm camera):
  - `lidar_link` (mesh `lidar.stl`), at `+0.305, 0, +0.112` m with
    `pitch=0.785` rad (45°).
  - `oakd_link` (mesh `OAK-D-parts-14sep2020.stl` in v2; in v1 it's a
    mesh-less box `0.02×0.1×0.03`), at `+0.257, 0, +0.19` m.
  - `arm_camera_link` (mesh `logitech_c920.stl`), fixed to **`link_6`** —
    mounted on the arm's wrist, not the chassis.
  - `back_camera_link` (mesh `logitech_c920.stl`), fixed to `base_link`,
    facing backward (`yaw=3.14159`).

`rescue_robot.urdf.xacro` shares this same arm/gripper structure and sensor
list, with slightly different offsets and `revolute` flippers instead of
`continuous`. `rescue_robot_foxglove.urdf.xacro` only has chassis + 4
flippers + arm + gripper, with no sensors.

## Meshes

| STL file | Link that uses it (confirmed in xacro) |
|---|---|
| `base_link.stl` | `base_link` (`rescue_robot.urdf.xacro`, `rescue_robot_v2.urdf.xacro`) |
| `flipper_1_1.stl` | `flipper_1_link` (both variants) |
| `flipper_2_1.stl` | `flipper_2_link`, **only in `rescue_robot.urdf.xacro`** — not referenced in `rescue_robot_v2.urdf.xacro` (the canonical model), which instead uses `flipper_4_1.stl` for both `flipper_2_link` and `flipper_4_link` |
| `flipper_3_1.stl` | `flipper_3_link` (both variants) |
| `flipper_4_1.stl` | `flipper_4_link` (both variants); also reused for `flipper_2_link` in v2 (see above) |
| `lidar.stl` | `lidar_link` (both variants) |
| `logitech_c920.stl` | `arm_camera_link` and `back_camera_link` (both variants) |
| `OAK-D-parts-14sep2020.stl` | `oakd_link`, only in `rescue_robot_v2.urdf.xacro` (in v1 `oakd_link` is a mesh-less box) |
| `tracked_1.stl` | `tracked_1_link`, only in `rescue_robot_v2.urdf.xacro` |
| `tracked_2.stl` | `tracked_2_link`, only in `rescue_robot_v2.urdf.xacro` |

`rescue_robot_foxglove.urdf.xacro` references `base.stl` and
`flipper_1.stl`…`flipper_4.stl`, which don't exist in `meshes/` (see the
variants table above).

## Launch files

| Launch | What it does | Use |
|---|---|---|
| `robot.launch.py` | Processes `urdf/rescue_robot_v2.urdf.xacro` with `xacro.process_file(...).toxml()` and publishes it as the `robot_description` parameter. Launches `robot_state_publisher` (package `robot_state_publisher`), `joint_state_publisher_gui` (manual sliders, doesn't read from real hardware), and `rviz2` (no `.rviz` config file, opens with the default config). | `ros2 launch aesir_robot_description robot.launch.py` |
| `robot_arm.launch.py` | Same as above but loads `urdf/arm.urdf.xacro` (standalone arm). The `rviz2` node is defined in the script but **commented out** in the `LaunchDescription` (`#node_rviz`), so this launch only brings up `robot_state_publisher` and `joint_state_publisher_gui`; RViz has to be opened separately if needed. | `ros2 launch aesir_robot_description robot_arm.launch.py` |

Neither launch declares any `LaunchArgument`s (not for choosing the
`.xacro`, not `use_sim_time`, not an RViz config) — the file to load is
hardcoded in each script's `os.path.join(...)`.

## Notes

- `package.xml` declares no runtime dependencies (`exec_depend`) even
  though both launch files need `robot_state_publisher`,
  `joint_state_publisher_gui`, `rviz2` and the Python `xacro` module at
  runtime. It only declares `ament_cmake` (buildtool) and
  `ament_lint_auto`/`ament_lint_common` (test). It also still has the
  placeholders `<description>TODO: Package description</description>` and
  `<license>TODO: License declaration</license>` unfinished.
- `rescue_robot_foxglove.urdf.xacro` isn't linked from any launch file or
  script in the repo, and its mesh paths (`base.stl`, `flipper_N.stl` via
  `file://$(find aesir_robot_description)/...`) don't match the `.stl`
  files present in `meshes/`. It looks like a leftover from a Foxglove
  Studio experiment that went stale after commit `2bdf7c27` (which
  renamed/added meshes for v2) — not confirmed, only inferred from the
  filename diff.
- The cross-wiring between `flipper_3_joint`/`flipper_4_joint` and their
  child links in `rescue_robot_v2.urdf.xacro` (see the links/joints
  section) couldn't be confirmed as intentional; it may simply be a swapped
  naming with no real functional effect (the joint is still valid, only
  which visual mesh corresponds to which joint name changes).
- `robot_moveit_config/config/rescue_robot.urdf.xacro` (outside this
  package) does a `xacro:include` of `rescue_robot_v2.urdf.xacro` and adds
  the `ros2_control` macros to it — the most direct confirmation that v2 is
  the model used in production (visualization and MoveIt2), not v1 or the
  foxglove variant.
- No reusable `robot.xacro` or macro was found across variants: all four
  copy the arm/gripper tree almost verbatim, with small offset differences
  — any arm dimension change today has to be replicated by hand in each
  file.
