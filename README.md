Two cooperative agents share sensory observations (cameras + lidar) but control
disjoint subsystems:

  Agent A — Arm & Gripper
    Obs joints  : joint_1..6, left_finger_joint, right_finger_joint  (qpos+qvel → 16 dims)
    Net output  : 7 dims  [j1..j6 ∈ [-1,1],  gripper_open ∈ [-1,1]]
    Adapter     : joints  → [-π, π] rad  (like ROS joint_states)
                  gripper_open → both fingers symmetrically in [0, 0.03] m
    Actuators   : pos_joint_1..6, pos_left_finger, pos_right_finger

  Agent B — Tracks & Flippers
    Obs joints  : flipper_joint_1..4, drive_l_{1..3}, drive_r_{1..3} (qpos+qvel → 20 dims)
    Net output  : 6 dims  [v_linear ∈ [-1,1], v_angular ∈ [-1,1], flip1..flip4 ∈ [-1,1]]
    Adapter     : twist → differential drive  (3 wheels per side treated as a single stack)
                  flipper wheels mirror their side's track velocity (scaled to [-1,1])
                  flippers  → [-π, π] rad  (joint positions)
    Actuators   : vel_drive_{l,r}_{1..3}, pos_flipper_{1..4},
                  vel_flip{1..4}_{back,front}