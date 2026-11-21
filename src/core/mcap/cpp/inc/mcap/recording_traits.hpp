// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

namespace core
{

/**
 * @brief Compile-time MCAP recording metadata per tracker type.
 *
 * Centralizes the default channel names used for MCAP recording and replay. Each tracker
 * impl's create_mcap_channels references these instead of embedding string literals. The
 * schema name is not here: flatc derives it from the .fbs as RecordT::GetFullyQualifiedName(),
 * which is what both McapTrackerChannels and McapTrackerViewers use.
 */

struct HeadRecordingTraits
{
    static constexpr std::array recording_channels = { "head" };
    static constexpr std::array replay_channels = { "head" };
};

struct HandRecordingTraits
{
    static constexpr std::array recording_channels = { "left_hand", "right_hand" };
    static constexpr std::array replay_channels = { "left_hand", "right_hand" };
};

struct ControllerRecordingTraits
{
    static constexpr std::array recording_channels = { "left_controller", "right_controller" };
    static constexpr std::array replay_channels = { "left_controller", "right_controller" };
};

struct FullBodyRecordingTraits
{
    static constexpr std::array recording_channels = { "full_body" };
    static constexpr std::array replay_channels = { "full_body" };
};

// Deprecated alias for the renamed FullBodyRecordingTraits (was
// FullBodyPicoRecordingTraits before the vendor-agnostic rename). Retained so source
// referencing the old type name keeps compiling (with a deprecation warning); prefer
// FullBodyRecordingTraits.
using FullBodyPicoRecordingTraits [[deprecated("renamed to core::FullBodyRecordingTraits")]] = FullBodyRecordingTraits;

struct MessageChannelRecordingTraits
{
    static constexpr std::array channels = { "message_channel" };
};

struct KeyboardRecordingTraits
{
    static constexpr std::array recording_channels = { "keyboard" };
    static constexpr std::array replay_channels = { "keyboard" };
};

// Traits for trackers declared in deviceio_trackers/trackers.toml, emitted from their
// channel manifest key. Add traits above by hand only for hand-written trackers.
#include "generated_recording_traits.inc"

} // namespace core
