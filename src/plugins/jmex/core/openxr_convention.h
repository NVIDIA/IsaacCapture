// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/jmex/body_joint_mapping.hpp"

namespace plugins
{
namespace jmex
{

/*!
 * @brief Re-express one MOXI-space pose in the axis convention this channel's consumers expect:
 *        OpenXR's +X right, +Y up, -Z forward, left and right being the operator's.
 *
 * MOXI's global frame is +X to the operator's left, +Y behind them, +Z up, right-handed.
 *
 * @param moxi_position         Joint position in MOXI space, meters.
 * @param moxi_orientation_wxyz Joint orientation in MOXI space, scalar first, as the SDK answers.
 */
JointPose moxi_pose_to_openxr(const float moxi_position[3], const float moxi_orientation_wxyz[4]);

} // namespace jmex
} // namespace plugins
