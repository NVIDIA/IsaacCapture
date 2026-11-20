// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "openxr_hand_tracking_push_channel.hpp"

#include <stdexcept>

namespace plugin_utils
{

OpenXRHandTrackingPushChannel::OpenXRHandTrackingPushChannel(const core::OpenXRSessionHandles& handles,
                                                             const core::XrTimeConverter& time_converter,
                                                             XrHandEXT hand)
    : time_converter_(time_converter), injector_(handles.instance, handles.session, hand, handles.space)
{
}

void OpenXRHandTrackingPushChannel::push(const XrHandJointLocationEXT* joint_locations,
                                         int64_t sample_time_local_common_clock_ns)
{
    if (joint_locations == nullptr)
    {
        throw std::invalid_argument("Hand tracking push requires joint locations");
    }

    injector_.push(joint_locations, time_converter_.convert_monotonic_ns_to_xrtime(sample_time_local_common_clock_ns));
}

} // namespace plugin_utils
