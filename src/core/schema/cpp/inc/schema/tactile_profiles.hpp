// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace core
{

// Provisional IsaacCapture registry identifiers. The payloads use external standards where
// named, but the profiles themselves are not Khronos or ROS standards.
inline constexpr std::string_view TACTILE_OPENXR_HAND_PROFILE_ID = "com.nvidia.isaaccapture.tactile.openxr-hand-binding";
inline constexpr std::string_view TACTILE_URDF_PROFILE_ID = "com.nvidia.isaaccapture.tactile.urdf-binding";
inline constexpr std::string_view TACTILE_ISAACCAPTURE_POSE_PROFILE_ID =
    "com.nvidia.isaaccapture.tactile.pose-stream-binding";
inline constexpr std::string_view TACTILE_BINDING_PROFILE_SCHEMA = "core.TactileBindingProfile";

} // namespace core
