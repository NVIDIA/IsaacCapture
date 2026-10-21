// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pusherio/hand_tracking_pusher.hpp"

#include <stdexcept>
#include <utility>

namespace core
{

HandTrackingPusher::HandTrackingPusher(std::unique_ptr<IHandTrackingPushChannel> channel) : channel_(std::move(channel))
{
    if (!channel_)
    {
        throw std::invalid_argument("HandTrackingPusher requires a channel");
    }
}

HandTrackingPusher::~HandTrackingPusher() = default;

void HandTrackingPusher::push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns)
{
    if (!joint_locations)
    {
        throw std::invalid_argument("HandTrackingPusher requires joint locations");
    }

    channel_->push(joint_locations, sample_time_local_common_clock_ns);
}

} // namespace core
