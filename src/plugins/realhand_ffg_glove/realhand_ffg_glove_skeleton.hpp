// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_protocol.hpp"

#include <openxr/openxr.h>
#include <plugin_utils/wrist_pose_source.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace plugins::realhand_ffg_glove
{

struct ThumbSensorMapping
{
    std::array<float, 4> origin, scale, minimum, maximum;
    std::array<std::array<float, 15>, 4> coefficients;
    std::array<float, 3> base, lengths;
};

struct HumanHandCalibration
{
    std::array<float, kNumSensors> open;
    std::array<float, kNumSensors> fist;
    std::array<float, kNumSensors> thumb_curl;
    std::array<float, kNumSensors> thumb_opposition;
    std::array<float, kNumSensors> spread;
    std::optional<ThumbSensorMapping> thumb_mapping = std::nullopt;
};

using HandJoints = std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT>;

// Device-side human FK only: no robot joint names, URDFs, limits, or target poses.
class HumanHandSkeleton
{
public:
    explicit HumanHandSkeleton(const std::string& calibration_path = {}, float hand_scale = 1.0f);
    bool calibrated(HandSide side) const;
    std::optional<HandJoints> reconstruct(const std::array<float, kNumSensors>& sensors, HandSide side) const;

private:
    std::array<std::optional<HumanHandCalibration>, 2> calibrations_;
    float hand_scale_;
};

// VALID describes usable reconstruction; TRACKED is added only with a live wrist source.
HandJoints place_hand(HandJoints joints, const XrPosef& wrist, bool wrist_valid, bool wrist_tracked);

class WristPoseContinuity
{
public:
    bool accept(const plugin_utils::WristSample& wrist, std::int64_t time_ns);

private:
    std::optional<XrPosef> previous_;
    std::int64_t previous_time_ns_ = 0;
    unsigned stable_samples_ = 0;
    bool recovering_ = false;
};

} // namespace plugins::realhand_ffg_glove
