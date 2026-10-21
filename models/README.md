# models — MuJoCo models (robot + terrains)

## What's here

The MJCF models (MuJoCo XML) of the AESIR robot and the training terrains,
plus their meshes (`meshes/`). Everything `rl_ws/base_training/config.py`
references in `TRACK_DEFS` comes from here: each training track is a
"complete" XML (robot + terrain) loaded with
`mujoco.MjModel.from_xml_path(...)`, and each robot-less terrain XML is also
used to generate that track's octomap (see below).

General pattern: the `aesir_complete_*.xml` files are marked in their own
comment as `GENERATED` and assembled with `<include file="...">` from
`aesir_mujoco_robot.xml` (the robot) + a terrain XML. They don't define their
own geometry, only shared `compiler`/`option`/`visual` blocks (radians,
`timestep=0.002`) and the `<include>`s.

## Robot

`aesir_mujoco_robot.xml` (model `rescue_robot`) is the robot alone, meant to
be included by the `aesir_complete_*.xml` files (it has no terrain or floor
of its own beyond an invisible reference plane `__mjv_default_floor`). It
defines:

- **Chassis**: `footprint_link` with a `freejoint` (floating base),
  `base_link` mesh + tracks (`tracked_1`, `tracked_2`).
- **4 flippers**: `flipper_1_link` .. `flipper_4_link`, each with its wheels
  (`wheel_flipperN_*`); excluded from contact against the chassis
  (`<contact><exclude .../></contact>`) to avoid spurious collisions.
- **6-DOF arm**: `link_1` .. `link_6` + gripper (`left_finger_link`,
  `right_finger_link`) mounted on `logitech_gripper_assembly`.
- **Drive tracks**: `wheel_l_1..6` / `wheel_r_1..6` (left/right).
- **Mounted sensors**: `sensor_lidar` (with 64 `ray_0..ray_63` sites in a
  fan), `camera_oakd_link`, `camera_back_link`.
- **Actuators** (`<actuator>`): `position` with `kp=200` for the 4 flippers
  (`pos_flipper_1..4`) and the arm's 6 joints (`pos_joint_1..6`), `position`
  with `kp=50` for the gripper; `velocity` with `kv=15` and `forcerange`
  ±140 for the tracks (`vel_track_left`/`vel_track_right`, over joints
  `drive_l_1`/`drive_r_1`); `velocity` for the lidar spin (`vel_lidar_spin`).
- **MuJoCo sensors** (`<sensor>`): 64 `rangefinder` (`lidar_0..lidar_63`),
  one per lidar site, `cutoff=30.0`, `noise=0.02`.

There's an equivalent URDF/Xacro model for ROS2/MoveIt2 at
[`ros2_ws/src/aesir_robot_description/`](../ros2_ws/src/aesir_robot_description/)
(documented in its own README) — a separate package, not detailed here.

## Terrains / tracks

| Terrain XML (no robot) | Terrain | Used by (track in `config.py`) |
|---|---|---|
| `plataform.xml` | Flat 20×20 m platform over lethal ground (floor = episode end) | `flat` |
| `track_steps.xml` (generated from `random_stepfields.xml`) | Field of random steps/cubes (`step_*` boxes 15/30 cm tall) | `steps` |
| `track_ramps.xml` (generated from `pistas_temporal/ramps/mapa_con_rampas.xml`) | Field of mesh ramps (`rampa_mesh`), ~1190 instances laid out in a grid | `ramps` |
| `aesir_pallets.xml` + `aesir_puerta.xml` | Pallets/sticks track (classic mission, obstacle JSON) + a hinged door | `pallets` |
| `track_pallets.xml` | Wrapper that just `<include>`s `aesir_pallets.xml` + `aesir_puerta.xml`, no robot | `pallets` (used to generate its octomap, see below) |

Notes on these files:

- `plataform.xml` (model `plataforma_gigante_aesir`) carries its own
  comment: *"LETHAL ground: in the pipeline, any 'plane'-type geom counts as
  floor (floor_contact -> penalty and episode end)"*. The `plane` sits below
  the raised platform and is only touched if the robot falls off the edge.
- `random_stepfields.xml` is in **degrees** (`compiler angle="degree"`) and
  is a standalone model (has its own `compiler`/`option`). `track_steps.xml`
  is its conversion to **radians** (comment: *"GENERATED from
  random_stepfields.xml... eulers converted to RADIANS... no
  compiler/option: set by the aesir_complete_* that includes it"*) — this is
  the one actually included/used.
- `track_ramps.xml` comes from `pistas_temporal/ramps/mapa_con_rampas.xml`
  (per its header comment), also converted to radians, with the mesh
  pointing at `meshes/ramps/rampa.stl`.
- `aesir_pallets.xml` (model `aesir_mapa_pallets`) defines the pallets
  track: a floor (`muerte_suelo_penalizacion`), the visual pallet mesh
  (`meshes/pallets/base_link.stl`), and 18 `fatal_pallet N` collision boxes
  (target/waypoint positions) + 18 `fatal_stick N` (thin stick-like
  obstacles) — names that `rl_ws/map_extractor.py` explicitly recognizes to
  classify/export the mission to JSON.
- `aesir_puerta.xml` (model `aesir_puerta`) is a door with a fixed frame
  (`door_frame`), a hinged leaf (`door_hinge`, range `-1.5..1.5`) and a
  handle with its own hinge (`handle_hinge`) — included alongside the
  pallets both in `aesir_complete_pallets.xml` and `track_pallets.xml`.

## "Complete" models (robot + terrain)

| XML | Includes (real `<include file="...">`) |
|---|---|
| `aesir_complete_flat.xml` | `aesir_mujoco_robot.xml` + `plataform.xml` |
| `aesir_complete_pallets.xml` | `aesir_mujoco_robot.xml` + `aesir_pallets.xml` + `aesir_puerta.xml` |
| `aesir_complete_ramps.xml` | `aesir_mujoco_robot.xml` + `track_ramps.xml` |
| `aesir_complete_steps.xml` | `aesir_mujoco_robot.xml` + `track_steps.xml` |
| `aesir_complete_flat_tower.xml` | `aesir_mujoco_robot.xml` + `plataform.xml` + `torre_botones.xml` + `botones_sueltos.xml` + inactive `freeze_base` weld — single world for the base+arm mission (`rl_ws/fsm_training/`); not a `TRACK_DEFS` entry |

The four track models share the same `compiler`/`option`/`visual` block (comment: *"Same
compiler/option/visual as aesir_complete.xml"* — see the note below about
that name). These are the XMLs that `TRACK_DEFS` in
`rl_ws/base_training/config.py` loads as each track's `xml=`.

## Other XML files

- **`boton.xml`** (model `boton`) — a button/mechanism with 5 pairs of
  `Component1_N`/`Component2_N` parts (fixed `mount` base + moving pin with
  a `slide`+`hinge` joint), meshes in `meshes/boton/`. Its `<compiler
  meshdir=...>` points to an absolute path
  `/home/aesir/aesir_rl/models/meshes/boton/` (not repo-relative). **No
  reference to `boton.xml` was found from any other XML or Python script in
  the repo** — it looks like an isolated component (possibly a piece for a
  future "press the button" mission, or test material); its exact purpose
  couldn't be confirmed from the code.
- **`plataform.xml`** — see the terrain table above (`flat` track).
- **`random_stepfields.xml`** — see the terrain table above (degrees-based
  source for `track_steps.xml`).

## Meshes

- `meshes/robot/` — chassis, tracks, flippers, lidar and cameras (Logitech
  C920, OAK-D) of the robot.
- `meshes/pallets/` — visual mesh for the pallets track (`base_link.stl`).
- `meshes/ramps/` — a single ramp mesh (`rampa.stl`), instanced ~1190 times
  in `track_ramps.xml`.
- `meshes/door/` — frame, leaf and handle of the door (`aesir_puerta.xml`).
- `meshes/boton/` — button parts (`boton.xml`): base and 5 pairs of
  moving/fixed components.

## Relation to rl_ws/tracks/ (octomaps)

`rl_ws/octomap_from_mujoco.py` builds, for each track, a 3D octomap (`.bt`)
from the collision geoms of a **robot-less** XML (so the rays used for mesh
rasterization don't hit the chassis). The XML → track mapping is hardcoded
in `_TRACK_XMLS` inside that script:

| Track | Octomap source XML |
|---|---|
| `flat` | `models/plataform.xml` |
| `steps` | `models/track_steps.xml` |
| `ramps` | `models/track_ramps.xml` |
| `pallets` | `models/track_pallets.xml` |

Output per track, in `rl_ws/tracks/<track>/`:
- `occupied_map.bt` — "raw" octomap (only voxels occupied by collision
  geoms: boxes rasterized directly, meshes via vertical raycast).
- `inflated_map.bt` — the same map dilated by a sphere of radius
  `--inflate-radius` (default 0.25 m ≈ robot radius + margin), ready as a
  costmap for `global_navigator.py`.
- `pallets/` additionally has `obstacles.json` (exported with
  `rl_ws/map_extractor.py --export-json`, not by `octomap_from_mujoco.py`)
  — the pallets/sticks map used by `kind="pallets"` navigation.

Regenerating a track's `.bt` files (after editing its XML):

```bash
python3 rl_ws/octomap_from_mujoco.py --track <track>   # track ∈ {flat, steps, ramps, pallets}
```

`rl_ws/map_extractor.py` is a complementary script (doesn't generate `.bt`
files): it renders a top-down view (`--save fig.png`) and/or exports to JSON
(`--export-json`) the layout of obstacles/pallets/sticks/robot in an XML,
classifying each geom by geom/body name prefix (`fatal_`, `muerte_`,
`ROBOT_BODIES`/`ARM_BODIES` lists, etc.).

## Notes

- The comment in the `aesir_complete_*.xml` files says *"Same
  compiler/option/visual as aesir_complete.xml"*, and `rl_ws/map_extractor.py`
  / `rl_ws/ppo_pallets_train.py` / `rl_ws/ppo_conv_train.py` also
  default-reference a `models/aesir_complete.xml` (or
  `aesir_robot_description/launch/aesir_complete.xml`) — **that file no
  longer exists** in the current repo. It looks like the single-track XML
  that predates the multi-track refactor (see commit `c89e0ca3`, "Multiple
  tracks training... tracks re-org"); those scripts likely went stale
  against the current `TRACK_DEFS`-based pipeline
  (`rl_ws/base_training/config.py` + `train_fast.py`/`train_base.py`).
  Whether they're still in use couldn't be confirmed.
- `boton.xml`'s exact purpose (not referenced from any other XML or script)
  couldn't be determined with certainty — see the "Other XML files" section.
- `aesir_pallets.xml` has no `<compiler>` of its own; it inherits whichever
  one is set by the XML that includes it (`aesir_complete_pallets.xml`, or,
  robot-less, `track_pallets.xml`).
