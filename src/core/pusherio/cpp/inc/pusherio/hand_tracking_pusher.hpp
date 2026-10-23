// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hand_tracking_push_channel.hpp"

#include <cstdint>
#include <memory>

namespace core
{

/*!
 * @brief Publishes one hand's joint locations through a transport-owned channel.
 *
 * This is the plugin-facing facade. OpenXR and remote transports provide the
 * channel implementation while the pusher preserves the existing hand data
 * contract used by plugins.
 */
class HandTrackingPusher
{
public:
    explicit HandTrackingPusher(std::unique_ptr<IHandTrackingPushChannel> channel);
    ~HandTrackingPusher();

    HandTrackingPusher(const HandTrackingPusher&) = delete;
    HandTrackingPusher& operator=(const HandTrackingPusher&) = delete;
    HandTrackingPusher(HandTrackingPusher&&) = delete;
    HandTrackingPusher& operator=(HandTrackingPusher&&) = delete;

    // joint_locations contains XR_HAND_JOINT_COUNT_EXT entries and is borrowed
    // for this call only.
    void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns);

private:
    std::unique_ptr<IHandTrackingPushChannel> channel_;
};

} // namespace core
