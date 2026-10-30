// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <cstdint>

namespace core
{

/** @brief Which device provides the wrist pose. */
enum class WristTrackingSourceMode
{
    Auto, //!< Optical hand tracking preferred, controller aim pose as fallback.
    HandTracking, //!< Optical hand tracking only.
    Controller, //!< Controller aim pose (plus the configured rigid offset) only.
};

/** @brief Source selection plus per-device tuning. */
struct WristTrackingSourceConfig
{
    WristTrackingSourceMode mode = WristTrackingSourceMode::Auto;
    // Rigid transform from the controller aim pose to the wrist joint, per
    // hand. Depends on the controller model and how it is mounted on the
    // glove, so each plugin supplies its own calibrated pair.
    XrPosef left_aim_to_wrist;
    XrPosef right_aim_to_wrist;
};

/** @brief One wrist query result, in the session base space. */
struct WristTrackingSample
{
    XrPosef pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    bool valid = false; //!< Pose usable (may be the cached last good pose).
    bool tracked = false; //!< Source actively tracked this frame.
};

/*! @brief Transport-neutral source for a hand's wrist pose in the session base space. */
class IWristTrackingSource
{
public:
    virtual ~IWristTrackingSource() = default;

    //! Queries the pose corresponding to a timestamp in the local common-clock domain.
    virtual WristTrackingSample query(XrHandEXT hand, int64_t sample_time_local_common_clock_ns) = 0;
};

} // namespace core
