# boton_training — Arm button-pressing task

6-DOF arm (base frozen) trained with PPO to press buttons on a dome-shaped tower. Task is precision manipulation under singularity constraints.

## Overview

The arm's job: reach a target button (one of 5 on a dome), press it fully (98% of travel), and hold it pressed for 2 seconds. Two modes mix during training — 35% fixed tower (replica of the maze's button tower), 65% random buttons scattered in reachable volume — to prevent overfitting to geometry.

This is a **sim-only training pipeline** (no ROS during iteration). Deployment to real robot uses the same observation/action contract through a separate ROS2 wrapper.

## Core files

| File | Role |
|---|---|
| `../boton_env.py` | `BotonArmEnv`: the task. Observation (28 dims), action (7 dims), reward shaping, collision/singularity detection, anti-windup torque limiting |
| `config.py` | Tunable params: N_ENVS=8, network size, PPO hyperparams (same as base training) |
| `net.py` | `BetaActorCritic`: network architecture. Beta distribution for all 7 continuous actions (Cartesian twist + grip) |
| `train_boton.py` | Main script. Orchestrates `VecBotonEnv` + `BetaActorCritic` + PPO update (imported from `base_training.ppo`) |
| `vec_boton.py` | `VecBotonEnv`: N=8 environments in parallel threads (MuJoCo's `mj_step` releases GIL) |

## Task design

### Observation (28 values)
```
 0:6   Arm joint positions (joint_1..6)
 6:12  Arm joint velocities
12:14  Finger positions
14:16  Finger velocities
16:19  Target button position (relative to arm_base_link)  ← task data
19:22  TCP → button vector (relative to arm_base_link)
22:25  Button normal (outward-facing direction, relative to arm_base_link)
  25   TCP distance to button
  26   Button press depth fraction (0=loose, 1=fully pressed)
  27   Jacobian condition number / 100, capped at 2.0
       └─ Singularity metric (1.0 = warning, >1.0 = hard stop in real MoveIt Servo)
```

**Why all relative to `arm_base_link`?** That's the frame MoveIt Servo receives twist commands in. Policy sees and acts in the same reference frame as the real robot.

### Action (7 values)
```
[0:3]  Linear TCP velocity (vx, vy, vz) in arm_base_link frame
[3:6]  Angular TCP velocity (wx, wy, wz) in arm_base_link frame
  [6]  Gripper opening (−1 = open, +1 = closed)
```

All bounded in [−1,1], then scaled:
- Linear: up to 0.4 m/s (precision over speed)
- Angular: up to 0.6 rad/s

**IK conversion:** Action → Jacobian pseudo-inverse (damped least-squares, same as MoveIt Servo) → joint velocities → integrated into position command.

### Reward shaping
```
progress     Potential on distance to button     ~+15 typical episode
touch        One-time bonus (grasper first contacts button face)  +5
press        Potential on press depth (best reached, not current)  ~+20
hold         Potential on hold duration best streak               ~+40
bonus        Terminal reward when HOLD_TIME_S reached            +250
alive        Per-step cost (encourages speed)                    −0.01
collisions   No positive reward collected in steps with collisions
TCP_speed    Penalty for |TCP velocity| > 3.0 m/s               −2.0 (capped)
joint_vel    Per-joint velocity limits (weak joints: j4/5/6)     −0.5 (capped)
joint_acc    Ramped acceleration (anti-jerk)                     −0.3 (capped)
joint_limits Soft margin enforcement (0.15 rad before stops)     −1.0 (capped)
singular     Jacobian condition > 80 (Servo warning)             −1.0 (capped)
wrong_button Terminal: touched a non-target button               −25
auto_col     Terminal: arm self-collision                        −25
```

**Anti-farming measure:** Shaping terms (touch, press, hold, progress) are **one-time or best-reached only**. Episode continues even after a bonus to prevent the policy from harvesting the same reward multiple times.

## Training

```bash
cd rl_ws
python3 -m boton_training.train_boton                    # default (10k iters)
python3 -m boton_training.train_boton --iters 200       # shorter run
python3 -m boton_training.train_boton --resume ../checkpoints_boton/boton_best.pt
python3 -m boton_training.train_boton --viewer          # live MuJoCo viewer on env 0
```

**Hyperparameters:**
- N_ENVS = 8 (parallel threads)
- STEPS_PER_ENV = 512
- PPO_EPOCHS = 10, BATCH_SIZE = 1024
- γ = 0.99, GAE λ = 0.95
- LR = 3e-4

**Checkpoints:** Saved to `checkpoints_boton/` every 50 iterations.

**WandB:** Logs to project `AIDL-PPO-AESIR-BOTONES` (set in `config.py`).

### Why not SAC for the arm?

SAC (Soft Actor-Critic) was added for the base in this branch because PPO plateaus (7–9% success, signal is there). For the arm: the task space is tighter (28 obs, 7 act, fewer parallel envs), and PPO works. Revisit if you need off-policy for arm later.

## Validation / Sim2Real

### Within MuJoCo

```bash
python3 -m boton_training.train_boton --viewer
```

Watch the policy train live. It's exploratory (noisy), so it looks less smooth than the final policy.

### Against real MoveIt Servo stack

Use `tests/test_boton.py` (not yet included in this branch's docs, but `boton_env.py` supports it via `render=True` or external viewer). The env can be used standalone:

```python
from boton_env import BotonArmEnv
env = BotonArmEnv(render=True)
obs = env.reset()
for _ in range(100):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

For real deployment: wrap policy loading in a ROS2 node that listens on `/servo_node/delta_twist_cmds` (same topic as real Servo). The observation format and action scales are already matched.

## Hardware assumptions validated

This task **validated real-robot constraints** that pure navigation doesn't exercise:

1. **TCP point accuracy:** Real gripper's pressing point is `tool_link` (0.12 m from link_6), not the gripper_assembly body origin (19.1 cm away). The env computes Jacobian at tool_link.
2. **Per-joint velocity limits:** Not a single global 3.14 rad/s, but per-joint (j4/5/6 are weaker, limited to 0.8–1.0). Policy learns this is a hard constraint.
3. **Singularity threshold:** cond(J) ≥ 100 = hard stop (real MoveIt Servo behavior). Penalty teaches the policy to avoid it.
4. **Soft joint limits:** Arm never commands within 0.15 rad of physical stops (motor damage prevention). Enforced as a hard constraint, not a penalty.
5. **Anti-windup:** When gripper is blocked (pressing button), the position integrator can't run away. Without it, shoulder keeps pushing against wrist and breaks it. Verified by testing with/without — critical for real deployment.

## Task modes

### Mode: `torre` (35% of episodes)

Fixed dome tower (replica of maze's button tower), 5 buttons at fixed compass positions:
- Button 0 (−X, 45°) — facing robot
- Button 1 (+Y, 45°)
- Button 2 (−Y, 45°)
- Button 3 (+Z, vertical, dome top)
- Button 4 (+X, 45°) — back side (not reachable)

Only buttons 0, 1, 2 are trainable by default (3 has Jacobian condition > 100 from almost all poses; 4 is unreachable). See `TARGETS_ALCANZABLES` in `boton_env.py`.

Tower jitters slightly (XY translation, yaw rotation) each reset to add variation without changing the underlying geometry.

### Mode: `sueltos` (65% of episodes)

1–4 random buttons scattered in arm's reachable volume (around shoulder joint_1). Curriculum: starts with 1 button (no distractors), gradually adds more as success rate rises. By end of training: ~4 random buttons, some of which are decoys (non-target).

Generated in 0.45–0.58 m radius, elevation ±10°–75° from horizontal, up to ±46° azimuth from robot's facing direction. Each button is tested for:
- Reachability (IK from rest pose)
- No collision with robot when placed
- Minimum separation from other buttons (0.16 m face-to-face)

## Files & architecture

```
rl_ws/
├── boton_env.py              # BotonArmEnv class
└── boton_training/
    ├── __init__.py
    ├── README.md             # this file
    ├── config.py             # all tunable params
    ├── net.py                # BetaActorCritic network
    ├── train_boton.py        # main training loop
    └── vec_boton.py          # VecBotonEnv (parallel envs)
models/
└── aesir_arm_botones.xml     # MuJoCo model: arm + button tower
```

## State machine (done conditions)

| Condition | Type | Effect |
|---|---|---|
| Hold target button 2 s | **Success** | episode ends, reward +250 |
| Time limit (35 s = 700 steps) | **Timeout** (truncate, not terminal) | episode ends, final obs fed to value net via bootstrap |
| Wrong button touched | **Failure** | episode ends |
| Arm self-collision | **Failure** | episode ends |
| Jacobian condition > 100 | **Freeze** (MoveIt Servo hard stop simulation) | policy can command, but arm won't move; strong incentive to avoid |

**Why truncate vs terminal on timeout?** The policy doesn't see time remaining, so "game ends after 35 s" makes the problem non-Markovian (same obs is worth different values mid-episode vs. at end). Standard fix: treat timeout as truncation (gamma * V(terminal_obs) added to reward), not terminal. Applies to both PPO and SAC.

## Comparison: arm_env vs. boton_env

| Aspect | `arm_env.ArmMuJoCoEnv` | `boton_env.BotonArmEnv` |
|---|---|---|
| Use | Reach arbitrary targets, Cartesian control | Press buttons, precision manipulation |
| Obs | 31 dims (no target info) | 28 dims (target relative, normal, press depth) |
| Reward | Simple distance-based | Rich shaping (touch→press→hold cascade) |
| Action | Twist, open/close | Twist, open/close (same) |
| Backend | ROS2/MoveIt Servo (slow, real-faithful) | Pure MuJoCo (fast, training) |
| IK | Jacobian at body origin | Jacobian at TCP (tool_link) |
| Anti-windup | No | Yes (critical for this task) |
| Training pipeline | Not packaged (manual) | Packaged: `train_boton.py`, parallel envs |

## Gotchas

- **Joint limits:** The `ARM_SOFT_MARGIN_RAD = 0.15` is not arbitrary. It's the largest margin that keeps the canonical rest pose in range. Increasing it would make the rest pose unreachable.
- **Singularity metric:** `cond(J) / 100` in observation is capped at 2.0. Why divided by 100? To match Servo's hard-stop threshold (100 = stop). Capping prevents outliers in gradient computation.
- **Grip action:** Single value [−1, +1], scaled to [0, 0.03 m]. Not two independent fingers. Policy learns the strategy naturally (e.g., close just before contact, open after).
- **Task is NOT solvable from all poses.** ~50% of random placements are unreachable or singular. The env filters at episode start (via `_alcanzable()` check), so every episode is solvable but not trivial.
- **Distractors:** Random buttons are **not gripped** during evaluation. They're just geometric obstacles. Touching them fails the episode, teaching precision.

## Next steps / open questions

1. **Deploy to real robot:** Wrap policy in ROS2 node, feed real arm state/Servo feedback, validate sim2real transfer.
2. **Behavior tree:** Once arm + base policies exist, need a state machine to coordinate (navigate → stop → press button sequence). Python-first, wrap in ROS2 later.
3. **Closed-loop vision:** Current obs is no camera (only kinematics). Add camera for visual feedback? Depends on sim2real strategy.
4. **Generalization:** Train on random button positions (done), test on tower unseen during training (not yet).
