// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! JointStateOutput with the 21 encoder joints in SDK order (not sorted by name).
//! position is static_cast<float>(joints_rad[i]); valid is bit i of joints_valid_mask.
//! Throws std::length_error above kJointStateMaxBytes.
std::vector<std::uint8_t> serialize_joint_state(const std::string& device_id,
                                                const std::array<std::string, kGloveJointCount>& names,
                                                const std::array<double, kGloveJointCount>& joints_rad,
                                                std::uint32_t joints_valid_mask);

//! TactileDataArray, one force vector per taxel in TactileMetadata order.
//! Throws std::length_error above kTactileMaxBytes.
std::vector<std::uint8_t> serialize_tactile(const std::vector<std::array<float, 3>>& forces);

} // namespace revohuman
} // namespace plugins
