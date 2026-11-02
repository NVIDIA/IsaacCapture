// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Engine adapter with no engine library behind it. It implements the full IPoseEngine contract and is the
// template for a new adapter. After a fatal error it releases accepted input unprocessed. Each pose is a
// placeholder: identity rotation and translation (image groups processed before it, IMU samples processed
// before it, 0).
#pragma once

#include <pose_engine_api/pose_engine.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <variant>

namespace pose_provider
{

/**
 * @brief Settings parsed from ResolvedPoseSetup::engine_settings: `key=value` entries separated by `;` or
 * newlines. An empty payload keeps every default.
 */
struct StubSettings
{
    std::size_t image_capacity = 4; //!< Accepted image groups not yet finished, including the one in flight.
    std::size_t imu_capacity = 256; //!< Accepted IMU samples not yet processed.
    std::size_t pose_capacity = 16; //!< Queued estimates.
    uint64_t initializing_groups = 1; //!< Leading groups reported as Initializing.
};

/** @brief Parses @p text into @p out; returns the reason on failure. */
std::string parse_stub_settings(const std::string& text, StubSettings& out);

class StubPoseEngine final : public IPoseEngine
{
public:
    StubPoseEngine() = default;
    ~StubPoseEngine() override;

    StubPoseEngine(const StubPoseEngine&) = delete;
    StubPoseEngine& operator=(const StubPoseEngine&) = delete;

    EngineCapabilities capabilities() const override;
    EngineIdentity identity() const override;
    std::string validate_configuration(const ResolvedPoseSetup& setup) const override;
    void initialize(const ResolvedPoseSetup& setup) override;
    void start() override;
    EnqueueResult on_image_group(EngineImageGroup group) override;
    EnqueueResult on_imu(const EngineImuSample& sample) override;
    void finish_input() override;
    bool input_drained() const override;
    bool try_get_next_pose(EnginePose& out) override;
    EngineHealth health() const override;
    void stop() override;

private:
    enum class State
    {
        Uninitialized,
        Initialized,
        Running,
        Draining,
        Stopped,
    };

    // Marks a run of IMU drops. Queued before the next accepted sample, or by finish_input() if no sample follows.
    struct ImuGap
    {
    };
    using Input = std::variant<EngineImageGroup, EngineImuSample, ImuGap>;

    bool admitting() const;
    bool matches_setup(const EngineImageGroup& group) const;
    void run_worker();

    // Written before start() and read-only afterwards.
    ResolvedPoseSetup setup_;
    StubSettings settings_;

    // Worker thread only.
    uint64_t groups_processed_ = 0;
    uint64_t imu_processed_ = 0;
    uint64_t pixel_checksum_ = 0;

    mutable std::mutex mutex_;
    std::condition_variable input_ready_;
    std::condition_variable pose_space_;
    State state_ = State::Uninitialized;
    bool stopping_ = false;
    std::deque<Input> inputs_;
    std::size_t images_pending_ = 0;
    std::size_t imus_queued_ = 0;
    bool imu_gap_open_ = false;
    bool worker_busy_ = false;
    bool input_discarded_ = false; // Accepted input left unprocessed by a fatal error or stop().
    std::deque<EnginePose> poses_;
    EngineHealth health_;

    std::thread worker_;
};

} // namespace pose_provider
