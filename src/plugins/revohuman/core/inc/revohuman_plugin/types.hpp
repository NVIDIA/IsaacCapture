// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! Encoder joints per glove, in SDK order (RH_JOINT_COUNT).
constexpr std::size_t kGloveJointCount = 21;
//! OpenXR hand joints (XR_HAND_JOINT_COUNT_EXT); PALM = 0, WRIST = 1.
constexpr std::size_t kHandJointCount = 26;
constexpr std::uint32_t kPalmJoint = 0;
constexpr std::uint32_t kWristJoint = 1;
constexpr std::uint32_t kMappedHandMask =
    ((1u << kHandJointCount) - 1u) & ~((1u << 6) | (1u << 11) | (1u << 16) | (1u << 21));
//! Protocol upper bound on taxels per glove (RH_TACTILE_MAX_POINTS).
constexpr std::size_t kMaxTaxels = 350;

enum class Hand : std::uint8_t
{
    Left = 0,
    Right = 1,
};

const char* hand_name(Hand hand);

//! Rigid transform. q is (x, y, z, w), unit length.
struct Pose
{
    std::array<double, 3> p{ 0.0, 0.0, 0.0 };
    std::array<double, 4> q{ 0.0, 0.0, 0.0, 1.0 };
};

//! The 26 OpenXR hand joints in the WRIST frame (joints[kWristJoint] is identity), with radii.
//! Produced by the selected pose provider, after mapping and reviewed corrections.
struct HandKinematics
{
    std::array<Pose, kHandJointCount> joints{};
    std::array<float, kHandJointCount> radius{};
    std::uint32_t position_valid_mask = 0;
    std::uint32_t orientation_valid_mask = 0;
};

struct TactileSample
{
    std::uint32_t device_tick = 0; //!< Tactile sampling tick, 1 ms, wraps at 2^32.
    std::int64_t sample_mono_ns = 0; //!< CLOCK_MONOTONIC sampling time; valid when has_sample_time.
    bool has_sample_time = false;
    bool valid = false;
    bool calibrated = false;
    std::vector<std::uint16_t> values; //!< The SDK reading, one per taxel.
};

struct GloveSample
{
    Hand hand = Hand::Left;
    std::uint32_t generation = 0; //!< Changes on every (re)start of the device stream.
    std::uint32_t device_tick = 0; //!< Device send tick, 1 ms, wraps at 2^32; not a sampling time.
    std::uint32_t joints_valid_mask = 0; //!< Bit i set when joints_rad[i] is valid.
    bool has_sample_time = false;
    std::int64_t sample_mono_ns = 0; //!< CLOCK_MONOTONIC encoder sampling time.
    std::array<double, kGloveJointCount> joints_rad{};
    std::optional<TactileSample> tactile;
    //! Joint poses at sample_mono_ns, in the WRIST frame.
    std::optional<HandKinematics> kinematics;
};

struct HandMeta
{
    Hand hand = Hand::Left;
    std::string product_code;
    std::string device_id;
    std::array<std::string, kGloveJointCount> joint_names;
    //! The product has tactile sensors. A P1 glove has none: it streams joints only.
    bool has_tactile = true;
};

} // namespace revohuman
} // namespace plugins
