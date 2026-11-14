// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <array>
#include <cstdint>
#include <variant>

struct omniglove_post_data;

namespace plugins::omniglove
{

// SDK STREAM_DATA_V1.md defines these POST_DATA bits; its C header has no names.
inline constexpr uint32_t kPostIkConverged = 1U << 0U;
inline constexpr uint32_t kPostBoneModelValid = 1U << 2U;

enum class ConversionStatus
{
    NotMapped,
    IkNotConverged,
    BoneModelInvalid,
    InvalidPose,
};

using HandJoints = std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT>;
using ConversionResult = std::variant<HandJoints, ConversionStatus>;

ConversionResult convert_post_data(const omniglove_post_data& post, bool is_left);

const char* conversion_status_name(ConversionStatus status) noexcept;

} // namespace plugins::omniglove
