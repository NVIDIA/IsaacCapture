// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <avatar_sdk/AvatarSDK.h>
#include <openxr/openxr.h>

#include <array>
#include <cstddef>
#include <string_view>

namespace plugins
{
namespace avatar
{

// DeviceIO names bilateral hardware through per-side methods (get_left_hand /
// get_right_hand), but every call site here needs one side as a value: to index
// per-side state, to label logs, and to pick an SDK handle function. This is that
// value. Keep the []-indexable order in sync with kDeviceSides.
enum class DeviceSide : size_t
{
    LEFT = 0,
    RIGHT = 1,
};

inline constexpr std::array<DeviceSide, 2> kDeviceSides{ DeviceSide::LEFT, DeviceSide::RIGHT };

// XR_HAND_LEFT_EXT / XR_HAND_RIGHT_EXT feed both XR_EXT_hand_tracking and the
// push device, so the side-to-OpenXR mapping lives here once.
inline constexpr XrHandEXT to_xr_hand(DeviceSide side)
{
    return side == DeviceSide::LEFT ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
}

// Operator-facing label; "left"/"right" are also the HapticCommand endpoint keys.
inline constexpr std::string_view to_string(DeviceSide side)
{
    return side == DeviceSide::LEFT ? "left" : "right";
}

// The Avatar SDK's own side enum. A conversion rather than a using-alias so the
// two enums cannot be passed for one another by accident.
inline constexpr ::avatar::DeviceSide to_sdk_side(DeviceSide side)
{
    return side == DeviceSide::LEFT ? ::avatar::DeviceSide::LEFT : ::avatar::DeviceSide::RIGHT;
}

// What the SDK reports when asked for one stream of a device: avatar::DeviceDataCategery.
// Ordered so the enum value doubles as an index into per-category state; the
// order itself is free (the SDK value is sent through to_sdk_category), the
// contiguity is not.
enum class DeviceDataCategory : size_t
{
    RAW = 0, //!< 22-DOF joint angles straight from the glove.
    ROBOT = 1, //!< Robot-oriented joint frame produced by the device retarget worker.
    HUMAN = 2, //!< Human hand skeleton (forward-kinematics landmarks).
};

//! Categories that carry joint names/positions; HUMAN reports landmarks instead.
inline constexpr std::array<DeviceDataCategory, 2> kJointDataCategories{ DeviceDataCategory::RAW,
                                                                         DeviceDataCategory::ROBOT };
inline constexpr size_t kDeviceDataCategoryCount = 3;

inline constexpr bool is_joint_category(DeviceDataCategory category)
{
    return category != DeviceDataCategory::HUMAN;
}

inline constexpr ::avatar::DeviceDataCategery to_sdk_category(DeviceDataCategory category)
{
    switch (category)
    {
    case DeviceDataCategory::RAW:
        return ::avatar::DeviceDataCategery::RAW;
    case DeviceDataCategory::ROBOT:
        return ::avatar::DeviceDataCategery::ROBOT;
    case DeviceDataCategory::HUMAN:
        return ::avatar::DeviceDataCategery::HUMAN;
    }
    return ::avatar::DeviceDataCategery::RAW;
}

// The SDK tags each frame with the payload it last carried; a frame is only
// usable when this tag matches the category we asked for.
inline constexpr ::avatar::AvatarDataFrame::PayloadCase payload_case_of(DeviceDataCategory category)
{
    switch (category)
    {
    case DeviceDataCategory::RAW:
        return ::avatar::AvatarDataFrame::kRaw;
    case DeviceDataCategory::ROBOT:
        return ::avatar::AvatarDataFrame::kRobot;
    case DeviceDataCategory::HUMAN:
        return ::avatar::AvatarDataFrame::kSkeleton;
    }
    return ::avatar::AvatarDataFrame::PAYLOAD_NOT_SET;
}

// Short uppercase label for one-line logs ("RAW", "ROBOT", "HUMAN").
inline constexpr std::string_view to_string(DeviceDataCategory category)
{
    switch (category)
    {
    case DeviceDataCategory::RAW:
        return "RAW";
    case DeviceDataCategory::ROBOT:
        return "ROBOT";
    case DeviceDataCategory::HUMAN:
        return "HUMAN";
    }
    return "UNKNOWN";
}

// The Hand payload a category reads/writes inside an AvatarDataFrame. RAW and
// ROBOT alias their own member; HUMAN is presented as raw too (the same slot
// the SDK copies skeleton data through), so callers can use one code path and
// branch only where landmark semantics matter.
inline constexpr ::avatar::Hand& hand_payload(::avatar::AvatarDataFrame& frame, DeviceDataCategory category)
{
    return category == DeviceDataCategory::ROBOT ? frame.robot : frame.raw;
}

inline constexpr const ::avatar::Hand& hand_payload(const ::avatar::AvatarDataFrame& frame, DeviceDataCategory category)
{
    return category == DeviceDataCategory::ROBOT ? frame.robot : frame.raw;
}

} // namespace avatar
} // namespace plugins
