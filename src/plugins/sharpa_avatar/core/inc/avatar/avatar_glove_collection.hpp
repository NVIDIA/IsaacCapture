// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace plugins
{
namespace avatar
{

// Teleop's PushTensorHapticDevice and this plugin identify the bidirectional
// haptic stream by this collection id. The producer sends one five-element
// normalized-power vector for each endpoint ("left" and "right").
inline constexpr const char* AVATAR_GLOVE_HAPTIC_COLLECTION_ID = "avatar_glove_haptic";

// Avatar RAW and ROBOT frames are exported as JointStateOutput tensors. The
// `joint_state` tensor identifier lets the standard JointStateSource discover
// them without an Avatar-specific Python adapter.
inline constexpr const char* AVATAR_RAW_LEFT_COLLECTION_ID = "avatar_raw_left";
inline constexpr const char* AVATAR_RAW_RIGHT_COLLECTION_ID = "avatar_raw_right";
inline constexpr const char* AVATAR_ROBOT_LEFT_COLLECTION_ID = "avatar_robot_left";
inline constexpr const char* AVATAR_ROBOT_RIGHT_COLLECTION_ID = "avatar_robot_right";

} // namespace avatar
} // namespace plugins
