<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dex Retargeting Configuration Files

This directory contains example configuration files for the `DexHandRetargeter`.

## Files

- **`hand_left_config.yml`** - Configuration for left hand retargeting (from IsaacLab Unitree G1 Inspire Hand)
- **`hand_right_config.yml`** - Configuration for right hand retargeting (from IsaacLab Unitree G1 Inspire Hand)
- **`left_robot_hand.urdf`**, **`right_robot_hand.urdf`** - Placeholder URDF files (REPLACE WITH YOUR ROBOT'S URDF)

## Using These Files

### For Testing/Demo
The placeholder URDFs (`left_robot_hand.urdf`, `right_robot_hand.urdf`) match the YAML config structure but give their joints no offsets, so moving a joint moves no fingertip. The retargeter therefore outputs the middle of each joint's range (0.785 for most joints) whatever the hand does, and 0 while no hand is tracked.

### For Real Robot Use
You need to provide your robot-specific URDF and YAML config:

1. **Get your robot's URDF**
   - From robot manufacturer
   - From IsaacLab Nucleus assets (if using standard robots)
   - Generate from CAD using tools like phobos

2. **Update the YAML config**
   - Set `urdf_path` to point to your URDF
   - Update `target_joint_names` to match your robot's joint names
   - Update `finger_tip_link_names` to match your robot's fingertip link names
   - Adjust `scaling_factor` and `low_pass_alpha` as needed

## YAML Configuration Structure

```yaml
retargeting:
  # Link names for fingertips (used by retargeting optimizer)
  finger_tip_link_names:
  - L_thumb_tip
  - L_index_tip
  - L_middle_tip
  - L_ring_tip
  - L_pinky_tip

  # Low-pass filter alpha (0-1, higher = less filtering)
  low_pass_alpha: 0.2

  # Scaling factor for hand size differences
  scaling_factor: 1.2

  # Target joint names (output from retargeting)
  target_joint_names:
  - L_thumb_proximal_yaw_joint
  - L_thumb_proximal_pitch_joint
  - L_index_proximal_joint
  - L_middle_proximal_joint
  - L_ring_proximal_joint
  - L_pinky_proximal_joint

  # Retargeting method (DexPilot is recommended)
  type: DexPilot

  # Path to robot hand URDF
  urdf_path: left_robot_hand.urdf

  # Wrist link name (base of the hand)
  wrist_link_name: L_hand_base_link
```

## Example Robot URDFs

Real robot URDFs can be obtained from:

### IsaacLab Nucleus (requires Isaac Sim)
- Unitree G1 Inspire Hand: `{ISAACLAB_NUCLEUS_DIR}/Mimic/G1_inspire_assets/retarget_inspire_white_left_hand.urdf`
- Fourier GR1-T2 Hand: Similar path in Nucleus

### Direct Sources
- **Shadow Hand**: https://github.com/shadow-robot/sr_common
- **Allegro Hand**: https://github.com/simlabrobotics/allegro_hand_ros
- **ReFlex Hand**: https://github.com/RightHandRobotics/reflex-ros-pkg
- **Custom robots**: Check your manufacturer's documentation

## Testing Without Real URDF

The placeholder URDFs are sufficient for:
- Testing the retargeting pipeline
- Verifying hand tracking data is being received (outputs change from 0 to 0.785)

They are **NOT** suitable for:
- Checking joint angle outputs (they do not follow the hand)
- Actual robot control
- Production use
- Accurate hand pose replication

## More Information

Check the dex-retargeting library documentation:
- GitHub: https://github.com/dexsuite/dex-retargeting
- Paper: https://arxiv.org/abs/2202.00448
