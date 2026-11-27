// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Records that cross the IPoseEngine boundary. Times are int64_t nanoseconds.
#pragma once

#include "engine_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace pose_provider
{

/**
 * @brief A prepared image in host memory.
 *
 * Readiness: submit only fully written storage, so that an engine never waits for it at admission. Do not change
 * the storage while any copy of `hold` exists. Releasing the last copy of `hold` returns the storage to its owner;
 * an engine keeps a copy until its last read of `data`.
 */
struct HostImage
{
    ImageLayout layout;
    const std::byte* data = nullptr; //!< First byte of the first plane.
    std::size_t size_bytes = 0; //!< Readable bytes from `data`; at least canonical_size_bytes(layout).
    std::shared_ptr<const void> hold;
};

/** @brief Image storage by memory kind. Add each new kind as an alternative and leave HostImage unchanged. */
using ImageStorage = std::variant<HostImage>;

struct EngineImage
{
    StreamId stream;
    ImageStorage storage;
};

/** @brief Original measurement times of a group's anchor image, passed through unchanged. */
struct AnchorTimes
{
    int64_t local_common_clock_ns = 0;
    int64_t raw_device_clock_ns = 0;
};

/** @brief One image per stream of the resolved setup, in any order. */
struct EngineImageGroup
{
    uint64_t group_id = 0; //!< Unique within the run; a group that is not accepted is not offered again.
    int64_t fusion_time_ns = 0;
    AnchorTimes anchor;
    std::vector<EngineImage> images;
};

/** @brief One prepared IMU sample, expressed in the IMU frame of the resolved calibration. */
struct EngineImuSample
{
    uint64_t input_id = 0; //!< Unique within the run.
    int64_t fusion_time_ns = 0;
    std::array<double, 3> acceleration_mps2{ 0.0, 0.0, 0.0 };
    std::array<double, 3> angular_velocity_radps{ 0.0, 0.0, 0.0 };
};

/** @brief Only a Tracking pose is valid. */
enum class TrackingState
{
    Initializing, //!< The run's SlamWorld is not fixed yet.
    Tracking, //!< The pose is in the run's SlamWorld.
    Lost, //!< Tracking is lost but can recover into the same SlamWorld.
    Relocalizing, //!< Recovering into the same SlamWorld.
};

/**
 * @brief The estimate for one accepted image group.
 *
 * SlamWorld is fixed before the first Tracking pose and never redefined within a run; loop correction may move
 * poses within it. If a pose may not be in the run's SlamWorld, the engine reports a fatal error. No TrackingState
 * value means failure. The adapter converts the engine library's tracked frame to Body with a transform fixed
 * before processing.
 */
struct EnginePose
{
    uint64_t group_id = 0;
    AnchorTimes anchor;
    Transform T_slamworld_body;
    TrackingState tracking_state = TrackingState::Initializing;
};

/** @brief What an engine supports without approximation. These are membership sets and imply no preference. */
struct EngineCapabilities
{
    std::set<InputMode> input_modes;
    std::set<MemoryKind> memory_kinds;
    std::set<PixelFormat> pixel_formats;
    std::set<DistortionModel> distortion_models;
};

/** @brief Names and exact versions of the adapter and the engine library. */
struct EngineIdentity
{
    std::string adapter;
    std::string adapter_version;
    std::string engine_library;
    std::string engine_library_version;

    friend bool operator==(const EngineIdentity&, const EngineIdentity&) = default;
};

/** @brief Cumulative counts for one run. Counts never decrease and `fatal_error` never clears. */
struct EngineHealth
{
    uint64_t dropped_image_groups = 0;
    uint64_t dropped_imu_samples = 0;
    uint64_t imu_gaps = 0; //!< Runs of consecutive IMU drops.
    uint64_t invalid_input_rejections = 0;
    uint64_t not_running_rejections = 0;
    uint64_t output_stalls = 0; //!< Times processing stalled on a full pose queue.
    bool fatal_error = false;
    std::string fatal_reason;
};

} // namespace pose_provider
