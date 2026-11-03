// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_records.hpp"
#include "resolved_setup.hpp"

#include <memory>
#include <string>

namespace pose_provider
{

/** @brief Admission result. EngineHealth counts every result except Accepted. */
enum class EnqueueResult
{
    Accepted, //!< Admitted for processing; not yet processed.
    DroppedBackpressure, //!< Queue full.
    RejectedInvalidInput, //!< Does not match the resolved setup or mode; the caller fails the run.
    RejectedNotRunning, //!< Called before start() or after finish_input() or stop(); expected only while
                        //!< aborting or cleaning up a partial startup.
};

/**
 * @brief A pose estimation engine adapter.
 *
 * Lifecycle: initialize() -> start() -> finish_input() -> stop(); stop() may also end any earlier state.
 * initialize() and start() throw std::logic_error out of this order, and finish_input() throws before start().
 * One producer thread submits input and one consumer thread polls output, health and input_drained(); the caller
 * serializes lifecycle calls. The engine processes image groups and IMU samples in the order it admits them,
 * through bounded queues.
 */
class IPoseEngine
{
public:
    /** @brief Ends the run as stop() does. */
    virtual ~IPoseEngine() = default;

    /** @brief Inputs this engine supports. Available in every state. */
    virtual EngineCapabilities capabilities() const = 0;

    /** @brief Available in every state. */
    virtual EngineIdentity identity() const = 0;

    /** @brief Empty when this engine can run @p setup, otherwise the reason it cannot. */
    virtual std::string validate_configuration(const ResolvedPoseSetup& setup) const = 0;

    /**
     * @brief Creates fresh estimation state and an empty map; never restores a saved map.
     * @throws std::invalid_argument when validate_configuration() rejects @p setup.
     */
    virtual void initialize(const ResolvedPoseSetup& setup) = 0;

    /** @brief Opens admission. */
    virtual void start() = 0;

    /**
     * @brief Offers one image group. Reports admission only, never blocks and never waits for readiness.
     *
     * When accepted, the engine holds the group's storage until its last read. Otherwise the engine keeps no
     * reference and reads nothing, and the holds are released when the call returns. Backpressure drops the whole
     * group. Processing that needs later IMU must not block the path that delivers that IMU.
     */
    virtual EnqueueResult on_image_group(EngineImageGroup group) = 0;

    /**
     * @brief Offers one IMU sample, copied before returning. Reports admission only and never blocks.
     *
     * Invalid in Stereo mode. Every drop is counted, and consecutive drops form one gap. Before the engine
     * processes the next accepted sample, or at finish_input() if none follows, it recovers from the gap or
     * reports a fatal error.
     */
    virtual EnqueueResult on_imu(const EngineImuSample& sample) = 0;

    /** @brief Ends admission; accepted input is still processed. Repeated calls have no further effect. */
    virtual void finish_input() = 0;

    /**
     * @brief True once all accepted input is processed, no storage read is pending and no new estimate can appear.
     *
     * Estimates may still be queued; keep polling them while waiting for this.
     */
    virtual bool input_drained() const = 0;

    /**
     * @brief Removes the next estimate without blocking; false when none is queued.
     *
     * Estimates follow accepted-group order, at most one per group. A full queue stalls processing; nothing is
     * dropped.
     */
    virtual bool try_get_next_pose(EnginePose& out) = 0;

    /** @brief Never blocks. */
    virtual EngineHealth health() const = 0;

    /**
     * @brief Ends the run from any state without needing a consumer, and wakes blocked work.
     *
     * Finishes or cancels pending reads before releasing storage. Repeated calls have no further effect.
     */
    virtual void stop() = 0;
};

/** @brief Creates an uninitialized engine adapter. Each adapter defines this; an executable links exactly one. */
std::unique_ptr<IPoseEngine> make_pose_engine();

} // namespace pose_provider
