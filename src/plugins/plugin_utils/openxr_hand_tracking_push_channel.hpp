// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/plugin_utils/hand_injector.hpp"

#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <pusherio/hand_tracking_push_channel.hpp>

namespace plugin_utils
{

class OpenXRHandTrackingPushChannel final : public core::IHandTrackingPushChannel
{
public:
    OpenXRHandTrackingPushChannel(const core::OpenXRSessionHandles& handles,
                                  const core::XrTimeConverter& time_converter,
                                  XrHandEXT hand);

    void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns) override;

private:
    const core::XrTimeConverter& time_converter_;
    HandInjector injector_;
};

} // namespace plugin_utils
