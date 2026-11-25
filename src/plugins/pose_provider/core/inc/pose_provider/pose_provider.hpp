// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pose_camera_input/camera_input.hpp>
#include <pose_engine_api/pose_engine.hpp>
#include <pusherio/plugin_session.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pose_provider
{

/** @brief Largest accepted ProviderConfig timeout. */
inline constexpr std::chrono::milliseconds kMaxProviderTimeout{ 3'600'000 };

/** @brief Longest accepted ProviderConfig::output_collection, in bytes: OpenXR's limit for a collection ID. */
inline constexpr std::size_t kMaxOutputCollectionBytes = 255;

/** @brief The run log's file name in ProviderConfig::episode_dir. */
inline constexpr const char* kRunLogFileName = "pose_metadata.mcap";

struct ProviderConfig
{
    //! From construction until the camera input is ready and the run is started; initialize() is not interrupted.
    std::chrono::milliseconds startup_timeout{ 30'000 };
    //! From entering Draining until the camera input has ended and the engine has drained.
    std::chrono::milliseconds drain_timeout{ 5'000 };
    //! The se3_tracker collection that receives the poses, new for each run. Empty: the run publishes nothing and opens
    //! no plugin session.
    std::string output_collection;
    //! Set by the plugin launcher. With an output collection, the plugin's device status is published under it.
    std::string plugin_root_id;
    //! An existing directory in which the run creates its run log, kRunLogFileName, which must not exist yet. Absent:
    //! the run writes no file.
    std::optional<std::filesystem::path> episode_dir;
};

enum class ProviderState
{
    Preparing, //!< Waiting for the camera input, then resolving the setup and initializing the engine.
    Ready, //!< Initialized; waits for request_start().
    Running, //!< Admission open; estimates and health are polled.
    Draining, //!< Ending input, then waiting for the engine to drain.
    Finished, //!< The engine is stopped, the camera input released and the status final.
};

/** @brief Where the first failure arose. */
enum class ProviderComponent
{
    Provider, //!< abort(), no start within the startup timeout, or an internal error.
    Setup, //!< The source description does not resolve to a setup, or the engine rejected the setup.
    Engine, //!< An engine call threw, a fatal error, a rejected image group, an estimate out of accepted-group order,
            //!< or no drain within the drain timeout.
    CameraInput, //!< No source description within the startup timeout, the input failed or ended before a finish, or
                 //!< it did not end within the drain timeout.
    InputPreparation, //!< A frame does not match the setup, a frame or loss report repeats or goes back in capture
                      //!< number, or the left eye's times are missing, negative on the device clock, or do not
                      //!< increase.
    Output, //!< The plugin session, the device status or the output collection did not open, or an estimate was not
            //!< published.
    RunLog, //!< The run log could not be created, written or closed.
};

struct ProviderStatus
{
    ProviderState state = ProviderState::Preparing;
    bool failed = false; //!< Never clears once set.
    std::string failure_reason; //!< The first failure.
    std::optional<ProviderComponent> failed_component; //!< Present once failed.
    EngineHealth engine_health; //!< As last read; final once Finished.
    uint64_t frames_lost = 0; //!< Captures missing after the first frame taken while Running, until the finish request.
    uint64_t frames_dropped = 0; //!< Frames dropped while Running because the engine held all prepared storage.
    std::chrono::milliseconds startup_timeout{ 0 }; //!< Applied limits.
    std::chrono::milliseconds drain_timeout{ 0 };
    std::size_t prepared_image_groups = 0; //!< Image groups whose prepared storage the engine can hold at once.
    std::optional<TrackingState> latest_tracking_state; //!< Of the latest published estimate.
    std::optional<int64_t> last_valid_measurement_time_ns; //!< Common-clock anchor time of the latest valid one.
    std::optional<int64_t> last_publication_time_ns; //!< Common-clock time the latest estimate was published.
};

class PoseInputAdapter;
class OutputSession;
class PoseOutput;
class RunLog;

/**
 * @brief Runs one engine adapter through one run on one camera input.
 *
 * Not thread-safe: one control thread makes every call, so it is the engine's only producer, consumer and
 * lifecycle caller, and the camera input's only user. Every path to Finished, including destruction, calls the
 * engine's stop() exactly once, then destroys the camera input, then closes the output, then the run log.
 */
class PoseProvider
{
public:
    /**
     * @brief Without @p camera_input no source description arrives, and the startup timeout fails the run. With an
     * output collection, the first update() opens the plugin session through @p session_factory, which tests pass, or
     * else an OpenXR session that does not wait for a headset system.
     * @throws std::invalid_argument for a null engine, a timeout outside (0, kMaxProviderTimeout] or an output
     * collection longer than kMaxOutputCollectionBytes.
     */
    PoseProvider(ProviderConfig config,
                 std::unique_ptr<IPoseEngine> engine,
                 std::unique_ptr<ICameraInput> camera_input = nullptr,
                 core::PluginSessionFactory session_factory = nullptr);

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

    /**
     * @brief The camera input's descriptor, readable when update() has camera events to take; -1 without an input or
     * once its last event is taken. Valid until the next update() or abort(): wait on it, never read or close it.
     */
    int wait_fd() const;

    /**
     * @brief How long to wait on wait_fd() before the next update(): until the provider's next timed check, and zero
     * once Finished. Read it, like wait_fd(), after each update().
     */
    std::chrono::milliseconds wait_timeout() const;

private:
    using Clock = std::chrono::steady_clock;

    // Explicit inputs only. Throws std::invalid_argument with the reason the source cannot be used.
    static ResolvedPoseSetup resolve_pose_setup(const SourceDescription& source,
                                                const EngineCapabilities& capabilities,
                                                const EngineIdentity& identity);

    void step(); // One state's step.
    void prepare(); // Preparing
    void await_start(); // Ready
    void run(); // Running
    void drain(); // Draining: detach() once, the camera input to its end, finish_input() once, then the engine drain.
    void enter(ProviderState state);
    bool take_camera_events(); // false after a failure.
    bool admit(CameraEvent& event); // Running: a frame's group offered, or a loss counted. false after a failure.
    bool take_estimates(); // false after a failure.
    bool check_health(); // false after a failure.
    template <typename Work>
    bool call_engine(const char* call, Work&& work); // false after fail(Engine, ...).
    void fail(ProviderComponent component, std::string reason);
    void record_failure(ProviderComponent component, std::string reason);
    void stop_engine(); // Exactly once.
    void finish_run(); // The only way into Finished.

    ProviderConfig config_;
    core::PluginSessionFactory session_factory_;
    std::unique_ptr<ICameraInput> camera_; // Destroyed by finish_run().
    std::optional<SourceDescription> source_; // As delivered, so its lenses are not the applied ones.
    std::optional<ResolvedPoseSetup> setup_;
    std::unique_ptr<PoseInputAdapter> input_; // Built after initialize().
    std::unique_ptr<IPoseEngine> engine_; // After any collaborator whose storage the engine may hold.
    std::unique_ptr<OutputSession> session_; // Closed last.
    std::unique_ptr<PoseOutput> output_; // Opened after initialize(); closed after the camera input's release.
    std::unique_ptr<RunLog> run_log_; // Opened at the first update, before the session; closed after the output.
    std::vector<uint64_t> accepted_groups_; // Accepted group IDs, oldest first, that may still get an estimate.
    Clock::time_point startup_deadline_;
    Clock::time_point drain_deadline_;
    bool start_requested_ = false;
    bool finish_requested_ = false;
    bool detached_ = false;
    bool camera_ended_ = false; // The camera input's last event was taken.
    uint64_t frames_after_cutoff_ = 0; // Frames taken in Draining, which are released unread.
    bool input_finished_ = false;
    bool engine_stopped_ = false;
    ProviderStatus status_;
};

} // namespace pose_provider
