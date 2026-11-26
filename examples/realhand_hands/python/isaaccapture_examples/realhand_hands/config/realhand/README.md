<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand hand models

Each model has left/right mesh-free URDF kinematics and vector-retargeting YAML.
Use the official `DexHandRetargeter` through `hand_config.get_hand_config`.
Joint limits and active-joint names come from these files; mimic joints remain
part of FK rather than independent output commands. No glove measurements or
calibration files are included. These are not visual or collision models.
