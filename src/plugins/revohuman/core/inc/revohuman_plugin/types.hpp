// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/pose_generated.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
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
//! The four non-thumb metacarpals are placeholders and never valid.
constexpr std::uint32_t kMappedHandMask =
    ((1u << kHandJointCount) - 1u) & ~((1u << 6) | (1u << 11) | (1u << 16) | (1u << 21));
//! Protocol upper bound on taxels per glove (RH_TACTILE_MAX_POINTS).
constexpr std::size_t kMaxTaxels = 350;

enum class Hand : std::uint8_t
{
    Left = 0,
    Right = 1,
};

constexpr std::array<Hand, 2> kHands = { Hand::Left, Hand::Right };

constexpr std::size_t index_of(Hand hand)
{
    return static_cast<std::size_t>(hand);
}

constexpr const char* hand_name(Hand hand)
{
    return hand == Hand::Left ? "left" : "right";
}

//! The OpenXR hand joints with per-joint validity.
struct HandJointPoses
{
    std::array<core::Pose, kHandJointCount> joints{};
    std::uint32_t position_valid_mask = 0;
    std::uint32_t orientation_valid_mask = 0;
};

struct TactileSample
{
    std::uint32_t device_tick = 0; //!< Tactile sampling tick, 1 ms, wraps at 2^32.
    std::optional<std::int64_t> sample_mono_ns;
    bool valid = false;
    bool calibrated = false; //!< The SDK reports forces, not raw counts.
    std::span<const float> values; //!< Newtons per taxel when calibrated; a view valid for the call it is given to.
};

struct GloveSample
{
    std::uint32_t generation = 0; //!< Changes on every (re)start of the device stream.
    std::int64_t sample_mono_ns = 0; //!< Encoder sampling time.
    std::uint32_t joints_valid_mask = 0;
    std::array<double, kGloveJointCount> joints_rad{};
    std::optional<TactileSample> tactile;
    HandJointPoses joint_poses; //!< WRIST frame.
};

struct HandMeta
{
    std::string product_code;
    std::string device_id;
    std::array<std::string, kGloveJointCount> joint_names;
    //! The product has tactile sensors (a P1 glove has none).
    bool has_tactile = false;
    //! Serialized TactileMetadata; absent when the SDK has no taxel layout for the glove.
    std::optional<std::vector<std::uint8_t>> tactile_metadata;
    std::uint32_t taxel_count = 0;
};

} // namespace revohuman
} // namespace plugins
