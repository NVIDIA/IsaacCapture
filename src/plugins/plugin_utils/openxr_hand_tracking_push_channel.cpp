// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "openxr_hand_tracking_push_channel.hpp"

#include "inc/plugin_utils/hand_injector.hpp"

#include <oxr_utils/oxr_time.hpp>

#include <stdexcept>

namespace plugin_utils
{

namespace
{

class OpenXRHandTrackingPushChannel final : public core::IHandTrackingPushChannel
{
public:
    OpenXRHandTrackingPushChannel(const core::OpenXRSessionHandles& handles, XrHandEXT hand)
        : time_converter_(handles), injector_(handles.instance, handles.session, hand, handles.space)
    {
    }

    void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns) override
    {
        if (joint_locations == nullptr)
        {
            throw std::invalid_argument("Hand tracking push requires joint locations");
        }

        injector_.push(
            joint_locations, time_converter_.convert_monotonic_ns_to_xrtime(sample_time_local_common_clock_ns));
    }

private:
    core::XrTimeConverter time_converter_;
    HandInjector injector_;
};

} // namespace

std::unique_ptr<core::IHandTrackingPushChannel> make_openxr_hand_tracking_push_channel(
    const core::OpenXRSessionHandles& handles, XrHandEXT hand)
{
    return std::make_unique<OpenXRHandTrackingPushChannel>(handles, hand);
}

} // namespace plugin_utils
