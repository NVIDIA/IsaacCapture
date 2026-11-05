// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <revohuman_plugin/types.hpp>

#include <array>
#include <revohuman.h>

namespace plugins
{
namespace revohuman
{

//! Converts one SDK glove frame. joints_rad comes from rh_joints_to_radians for the same frame;
//! the plugin never maps degrees itself. Throws std::invalid_argument when struct_size is smaller
//! than this build's RhGloveFrame, the hand is unknown, or the tactile point count is out of range.
//! Kinematics are left empty; the KinematicsProvider fills them in.
GloveSample from_rh_frame(const RhGloveFrame& frame, const std::array<double, kGloveJointCount>& joints_rad);

} // namespace revohuman
} // namespace plugins
