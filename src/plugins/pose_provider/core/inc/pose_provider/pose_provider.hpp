// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pose_engine_api/pose_engine.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace pose_provider
{

/** @brief Largest accepted ProviderConfig timeout. */
inline constexpr std::chrono::milliseconds kMaxProviderTimeout{ 3'600'000 };

struct ProviderConfig
{
    //! From construction until the setup has arrived and the run is started; initialize() is not interrupted.
    std::chrono::milliseconds startup_timeout{ 30'000 };
    //! From entering Draining until the engine has drained.
    std::chrono::milliseconds drain_timeout{ 5'000 };
};

enum class ProviderState
{
    Preparing, //!< Waiting for the setup, then initializing the engine.
    Ready, //!< Initialized; waits for request_start().
    Running, //!< Admission open; estimates and health are polled.
    Draining, //!< Ending input, then waiting for the engine to drain.
    Finished, //!< The engine is stopped and the status is final.
};

/** @brief Where the first failure arose. */
enum class ProviderComponent
{
    Provider, //!< abort(), no start within the startup timeout, or an internal error.
    Setup, //!< No setup within the startup timeout, the setup supplier failed, or the engine rejected the setup.
    Engine, //!< An engine call threw, a fatal error, an out-of-contract estimate, or no drain within the drain timeout.
};

struct ProviderStatus
{
    ProviderState state = ProviderState::Preparing;
    bool failed = false; //!< Never clears once set.
    std::string failure_reason; //!< The first failure.
    std::optional<ProviderComponent> failed_component; //!< Present once failed.
    EngineHealth engine_health; //!< As last read; final once Finished.
    std::chrono::milliseconds startup_timeout{ 0 }; //!< Applied limits.
    std::chrono::milliseconds drain_timeout{ 0 };
};

/** @brief The run's setup once known, nullopt before. Polled once per update() while Preparing; must not block. */
using SetupSupplier = std::function<std::optional<ResolvedPoseSetup>()>;

/**
 * @brief Runs one engine adapter through one run.
 *
 * Not thread-safe: one control thread makes every call, so it is the engine's only producer, consumer and
 * lifecycle caller. Every path to Finished, including destruction, calls the engine's stop() exactly once.
 */
class PoseProvider
{
public:
    /**
     * @brief Without @p setup_supplier no setup arrives, and the startup timeout fails the run.
     * @throws std::invalid_argument for a null engine or a timeout outside (0, kMaxProviderTimeout].
     */
    PoseProvider(ProviderConfig config, std::unique_ptr<IPoseEngine> engine, SetupSupplier setup_supplier = {});

    /** @brief Aborts an unfinished run. */
    ~PoseProvider();

    PoseProvider(const PoseProvider&) = delete;
    PoseProvider& operator=(const PoseProvider&) = delete;

    /**
     * @brief Does the current state's work, continuing into each state it reaches. Never throws. It waits on no input
     * or engine queue; the run's setup and teardown, such as the engine's initialize() and stop(), run within it.
     */
    void update();

    /** @brief Lets the run leave Ready; may be called earlier. A pending request_finish() wins. */
    void request_start();

    /**
     * @brief Drains a running run, or ends one that has not started without failing it.
     * Takes effect at the next update(); repeated calls have no further effect.
     */
    void request_finish();

    /** @brief Fails and ends an unfinished run now, stopping the engine. No effect once Finished. */
    void abort();

    bool finished() const;
    ProviderStatus status() const;

private:
    using Clock = std::chrono::steady_clock;

    void step(); // One state's step.
    void prepare(); // Preparing
    void await_start(); // Ready
    void run(); // Running
    void drain(); // Draining: the finish-input step once, then the engine drain.
    void enter(ProviderState state);
    bool take_estimates(); // false after a failure.
    bool check_health(); // false after a failure.
    bool call_engine(const char* call, const std::function<void()>& work); // false after fail(Engine, ...).
    void fail(ProviderComponent component, std::string reason);
    void record_failure(ProviderComponent component, std::string reason);
    void stop_engine(); // Exactly once.
    void finish_run(); // The only way into Finished.

    ProviderConfig config_;
    SetupSupplier setup_supplier_;
    std::optional<ResolvedPoseSetup> setup_;
    std::unique_ptr<IPoseEngine> engine_; // After any collaborator whose storage the engine may hold.
    Clock::time_point startup_deadline_;
    Clock::time_point drain_deadline_;
    bool start_requested_ = false;
    bool finish_requested_ = false;
    bool input_finished_ = false;
    bool engine_stopped_ = false;
    ProviderStatus status_;
};

} // namespace pose_provider
