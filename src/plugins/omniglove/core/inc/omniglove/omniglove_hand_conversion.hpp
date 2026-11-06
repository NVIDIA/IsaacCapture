// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omniglove/omniglove.h>
#include <openxr/openxr.h>

#include <array>

namespace plugins::omniglove
{

enum class ConversionStatus
{
    Success,
    NotMapped,
    IkNotConverged,
    BoneModelInvalid,
    InvalidPose,
};

ConversionStatus convert_post_data(const omniglove_post_data_t& post,
                                   bool is_left,
                                   std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT>& joints);

const char* conversion_status_name(ConversionStatus status) noexcept;

} // namespace plugins::omniglove
