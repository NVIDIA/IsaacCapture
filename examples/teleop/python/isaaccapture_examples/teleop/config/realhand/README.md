<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand Configuration for the Official Dex Hand Retargeter

These L6, O6, and L20 configurations are consumed directly by IsaacCapture's
`DexHandRetargeter`. Both optical OpenXR hand tracking and the RealHand FFG
Glove plugin supply the same standard 26-joint `HandInput`.

Each model has a left/right YAML vector objective and a mesh-free, hand-only
URDF. The URDF joint origins, axes, limits, and existing mimic relationships
come from the released P7 assemblies at immutable revision
`e20d1f7dfe27d84784270012f291ae16f4762b0c` of
[RealHand Teleop](https://huggingface.co/realhandinc/realhand-teleop).
Fixed fingertip frames use the mean of the distal visual mesh's terminal
0.5 mm cap. They are geometric reference points, not contact sensors.
Only kinematics are included; no meshes, CAD sources, or personal calibration.

L6 and O6 fit wrist-to-tip, inter-tip, and finger segment vectors with a 0.75
length scale. L20 uses inter-tip and finger segment vectors with a 0.9 scale;
its long base-to-knuckle offsets are not treated as human wrist-to-knuckle
lengths. Individual phalanx vectors constrain L20's additional degrees of
freedom without requiring a specific gesture.

L6/O6 use `normal_delta: 0.001` and `huber_delta: 0.02`. L20 uses
`normal_delta: 0.016` and `huber_delta: 0.005` to reduce thumb solution changes
while retaining sensitivity to small position errors. All models use the
official joint-output filter `low_pass_alpha: 0.4`. It advances 40% toward each
solved target per update; its delay depends on the update rate. These settings
apply equally to FFG and optical tracking, not to controller trigger mapping.

There is no gesture classifier, pinch projection, contact threshold, or pose
anchor. These are model-level starting configurations, not a guarantee of exact
contact: limited robot degrees of freedom and estimated glove geometry can
prevent exact reproduction of human finger positions. Validate stationary noise,
slow bidirectional human-space trajectories at several sampling densities, and
endpoint poses. Inspect unfiltered joint solutions, not just rendered output;
reachable-robot round trips alone do not establish glove accuracy or continuity.
Human calibration is independent of these robot settings and does not need to
be repeated when changing models.

The example passes an explicit row-vector coordinate transform: OpenXR dorsal
`+Y` maps to robot `-X`, and OpenXR fingertip `-Z` maps to robot `+Z`.
Output order comes from `get_realhand_profile()`; the official optimizer owns
joint-limit handling and URDF mimic adaptation.

Complete robot assemblies for simulation remain available through
`python -m isaaccapture_examples.teleop.fetch_realhand_assets`.

To replay a local human calibration through the native skeleton reconstruction
and test L20 continuity, add `--ffg-human-calibration /path/to/human.yml` to the
RealHand pytest suite along with `--ffg-skeleton-executable /path/to/ffg_test`.
The additional tests use 1%, 0.5%, and 0.25% trajectory steps and synthetic noise.
They require no live glove. Recordings remain local and are not part of these
robot configurations; passing replay tests is not a guarantee for all motions.
