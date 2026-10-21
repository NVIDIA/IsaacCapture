// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <cstdint>

namespace core
{

/*!
 * @brief Transport-owned channel for publishing one hand's tracking data.
 *
 * OpenXR value types define the established hand data shape. Runtime handles,
 * function pointers, and calls remain private to concrete channel implementations.
 */
class IHandTrackingPushChannel
{
public:
    // Orderly destruction closes the logical hand stream and makes it inactive
    // at the receiver. Remote implementations must also expire active state
    // after an unexpected transport disconnect.
    virtual ~IHandTrackingPushChannel() = default;

    // joint_locations contains XR_HAND_JOINT_COUNT_EXT entries and is borrowed
    // for this call only. Asynchronous transports must copy it before returning.
    virtual void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns) = 0;
};

} // namespace core
