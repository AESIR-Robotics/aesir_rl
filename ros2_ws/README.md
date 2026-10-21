# ros2_ws — AESIR robot ROS2 workspace

## What's here

`colcon` workspace with the 3 ROS2 packages that describe and control the
physical AESIR robot: the geometry (URDF/xacro), the low-level driver that
speaks I2C with the microcontroller, and the arm's MoveIt2 configuration.

- **`aesir_robot_description`** defines where everything comes from: the
  `xacro` files with the robot's geometry (tracks, 4 flippers, arm, sensors)
  and their meshes. The other packages — and also `models/` in MuJoCo, see
  below — build on this description.
- **`hardware`** is the C++ driver that translates ROS2 commands (`cmd_vel`,
  `JointControl`) into I2C frames for the flipper steppers and the track DC
  motors, and publishes their feedback back.
- **`robot_moveit_config`** provides the arm's MoveIt2 configuration
  (planning, collisions, real-time servo) and a `ros2_control` System
  Interface (`TopicBridgeHardware`) that packages arm trajectories as
  `hardware/JointControl` messages over topics — the bridge between MoveIt
  and `hardware`.

Overall: `aesir_robot_description` supplies the model, `hardware` moves the
real motors, and `robot_moveit_config` plans the arm's motion and translates
it into the topics `hardware` understands. `robot_moveit_config` explicitly
depends on `aesir_robot_description` (`exec_depend` in its `package.xml`);
`hardware` is independent at the build level — it relates to the other two
only by topic convention (`/hardware_node/*`, `/commands_hardware`), not
package dependency.

## Packages

| Package | What it is | Docs |
|---|---|---|
| `aesir_robot_description` | URDF/xacro of the complete robot (tracks + 4 flippers + arm + sensors) and of the standalone arm, STL meshes, and RViz visualization launch files | [README](src/aesir_robot_description/README.md) |
| `hardware` | ROS2↔I2C driver: 4 steppers (flippers) + 2 DC motors (differential drive) over a custom binary protocol with CRC-8, plus a custom `JointControl` message | [README](src/hardware/README.md) |
| `robot_moveit_config` | MoveIt2 configuration for the 6-DOF arm: planning, MoveIt Servo, `ros2_control`, and the `TopicBridgeHardware` bridge toward `hardware` | [README](src/robot_moveit_config/README.md) |

## How they relate (typical launch order)

There's no single launch file that starts all three packages together —
they have to be composed by hand. From the existing launch files, the
following can be inferred, but **it isn't encoded anywhere as an "official"
pipeline**:

1. **Real hardware driver** (optional if working purely in simulation):
   `ros2 run hardware dc_motors --ros-args -p ...` (see
   [hardware/README.md](src/hardware/README.md)) — exposes
   `/hardware_node/joint_states`, `/hardware_node/state_vel`, etc.
   `robot_moveit_config`'s `hardware_loopback.launch.py` (instant echo, no
   physics) or `rl_ws/base_training/mujoco_sim_rosbridge.py` (real MuJoCo
   physics) can be used instead — the three are alternatives to each other
   and publish on the same topics.
2. **MoveIt2 + arm control**: `robot_moveit_config`'s `bringup.launch.py`
   (or `bringup_controllers.launch.py`) brings up MoveIt, RViz and (after a
   3 s delay) MoveIt Servo. Internally it builds `robot_description` from
   `config/rescue_robot.urdf.xacro`, which directly includes
   `aesir_robot_description`'s xacro — it **does not** invoke
   `aesir_robot_description`'s own launch files
   (`robot.launch.py`/`robot_arm.launch.py`), which are independent RViz
   visualizers and aren't part of this chain.

In short: `aesir_robot_description` participates by *xacro inclusion*, not
by chained launch; `hardware` and `robot_moveit_config` do connect at
runtime via topics, but which of the three hardware "backends" to use
(real, loopback, or MuJoCo bridge) is a manual decision each session, not
something a single command resolves.

## Build

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

`aesir_robot_description` declares no build dependencies beyond
`ament_cmake` (its `package.xml` is unfinished: description and license are
still "TODO"). `robot_moveit_config` does depend on
`aesir_robot_description`, plus MoveIt2 and `ros2_control`; see `bash
setup.bash` at the repo root to install everything at once (auto-detects
Humble/Jazzy, including `ros2_control`, `moveit`, `moveit_servo`, etc.).

## Relation to the simulation (rl_ws/)

`rl_ws/base_training/mujoco_sim_rosbridge.py` is an alternative backend to
the real `hardware` driver: it listens on `/commands_hardware` and
`hardware_node/cmd_vel` just like the real hardware, but instead of speaking
I2C it applies the commands to a MuJoCo model with real physics (mass,
gravity, collisions) and publishes the resulting state on
`/hardware_node/joint_states` — allowing MoveIt/RViz/servo to run against a
faithful simulation without touching the real robot. It's explicitly
incompatible with `hardware_loopback.py`: both publish on the same topic and
must not be launched together.

## Notes

- **Supported distro**: Humble or Jazzy (auto-detected by `setup.bash` from
  the Ubuntu version, 22.04→Humble / 24.04→Jazzy). `robot_moveit_config`'s
  README mentions also being tested on Iron for `hardware`.
- **Which full-robot xacro is current**: `aesir_robot_description` has
  three variants (`rescue_robot.urdf.xacro`, `rescue_robot_v2.urdf.xacro`,
  `rescue_robot_foxglove.urdf.xacro`). By code, **`rescue_robot_v2.urdf.xacro`
  is the one actively in use**: it's included both by
  `aesir_robot_description/launch/robot.launch.py` and by
  `robot_moveit_config/config/rescue_robot.urdf.xacro` (the actual source
  MoveIt uses). `rescue_robot.urdf.xacro` (no suffix) and
  `rescue_robot_foxglove.urdf.xacro` aren't referenced by any active launch
  file in `ros2_ws/` — they look like variants or earlier versions; their
  exact purpose or whether they're still needed couldn't be confirmed.
- `arm.urdf.xacro` describes the standalone arm (no tracks or flippers) and
  is only used by `aesir_robot_description/launch/robot_arm.launch.py`, an
  independent RViz visualizer — it isn't the xacro `robot_moveit_config`
  consumes.
- There's no workspace-level launch file that starts all three packages
  together; see the "How they relate" section above for what can be
  inferred from the code.
