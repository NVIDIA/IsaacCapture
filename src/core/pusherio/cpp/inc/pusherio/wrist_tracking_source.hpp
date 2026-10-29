// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <cstdint>

namespace core
{

enum class WristTrackingSourceMode
{
    Auto,
    HandTracking,
    Controller,
};

struct WristTrackingSourceConfig
{
    WristTrackingSourceMode mode = WristTrackingSourceMode::Auto;
    XrPosef left_aim_to_wrist{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    XrPosef right_aim_to_wrist{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
};

struct WristTrackingSample
{
    XrPosef pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    bool valid = false; //!< Pose is usable, including a cached last-good pose.
    bool tracked = false; //!< Source is actively tracked for this sample.
};

/*! @brief Transport-neutral source for a hand's wrist pose in the session base space. */
class IWristTrackingSource
{
public:
    virtual ~IWristTrackingSource() = default;

    //! Queries the pose corresponding to a timestamp in the local common-clock domain.
    virtual WristTrackingSample query(bool is_left, int64_t sample_time_local_common_clock_ns) = 0;
};

} // namespace core
