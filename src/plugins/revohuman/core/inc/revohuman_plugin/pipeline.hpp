// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_clock.hpp"
#include "event_log.hpp"
#include "tactile.hpp"
#include "types.hpp"
#include "wrist_tracking.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! The 26 OpenXR joints of one hand in the stage, ready for a hand injector.
struct HandFrame
{
    std::array<Pose, kHandJointCount> joints{};
    std::array<float, kHandJointCount> radius{};
    std::uint32_t position_valid_mask = 0;
    std::uint32_t orientation_valid_mask = 0;
    bool tracked = false; //!< TRACKED bits as well as VALID bits.
    std::int64_t sample_mono_ns = 0;
};

//! Where one hand's pipeline delivers its output. The app maps each call onto OpenXR; tests
//! record the calls. Times are CLOCK_MONOTONIC ns. Raw times are the sampling time on the device
//! clock in ns where the device reports one (tactile); joint states pass their monotonic time.
class PushSink
{
public:
    virtual ~PushSink() = default;
    virtual void push_joint_state(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) = 0;
    virtual void push_tactile(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) = 0;
    //! Remove retained force data when its validity ends; metadata stays available.
    virtual void clear_tactile() = 0;
    virtual void push_shape(const std::vector<std::uint8_t>& buffer, std::int64_t now_ns) = 0;
    //! Remove the TactileMetadata: the glove now in use has no tactile sensors.
    virtual void clear_shape() = 0;
    virtual void hand_create() = 0;
    virtual void hand_destroy() = 0;
    virtual void hand_push(const HandFrame& frame) = 0;
};

//! WRIST pose of one hand in the OpenXR stage, as the host's wrist source reports it.
struct WristPose
{
    Pose pose_stage;
    bool valid = false; //!< pose_stage holds a pose; without one the hand is placed at the origin.
    bool tracked = false; //!< The runtime tracks the pose now, not only holds the last one.
};

//! Where the hand pipeline takes the WRIST pose from (plugin_utils::WristPoseSource in the plugin).
class WristProvider
{
public:
    virtual ~WristProvider() = default;
    virtual WristPose wrist(Hand hand, std::int64_t sample_mono_ns) = 0;
};

struct PipelineConfig
{
    //! No hand push for this long destroys the injector.
    std::int64_t stall_ns = 50'000'000;
    //! The TactileMetadata is pushed once a glove with tactile sensors is identified, then again at this period.
    std::int64_t shape_resend_ns = 1'000'000'000;
    //! Data older than this counts as absent in health().
    std::int64_t health_timeout_ns = 500'000'000;
    //! Maximum age of a tactile force sample; also used for tactile health.
    std::int64_t tactile_max_age_ns = 100'000'000;
    //! Period of the tactile_rate event in the event log; 0 turns it off.
    std::int64_t tactile_rate_period_ns = 1'000'000'000;
};

//! Why a tactile sample was not pushed. Without a matching shape no force direction exists, so
//! nothing is pushed rather than a guess.
enum class TactileSkip
{
    None,
    NoShape,
    NoTactile, //!< The glove's product has no tactile sensors (P1).
    Invalid,
    Uncalibrated,
    CountMismatch,
    NoSampleTime,
    StaleSample,
    NonMonotonicSample,
};

const char* tactile_skip_name(TactileSkip skip);

struct PipelineStats
{
    std::uint64_t glove_samples = 0;
    std::uint64_t glove_before_meta = 0;
    std::uint64_t joint_states_pushed = 0;
    std::uint64_t glove_without_kinematics = 0;
    //! Glove samples stamped after the time they were received, pushed at the receive time.
    std::uint64_t glove_future_clamped = 0;
    std::uint64_t hands_pushed = 0;
    std::uint64_t hands_tracked = 0;
    std::uint64_t injector_creates = 0;
    std::uint64_t injector_destroys = 0;
    std::uint64_t tactile_samples = 0;
    std::uint64_t tactile_pushed = 0;
    std::uint64_t tactile_skipped = 0;
    //! Tactile samples with the tick of the last pushed one, e.g. repeated on every glove sample.
    std::uint64_t tactile_repeats = 0;
    //! The tactile_repeats marked invalid: the firmware re-sends the last snapshot that way when one
    //! read of the tactile board fails.
    std::uint64_t tactile_invalid_repeats = 0;
    //! StaleSample skips, by side: sampled after the time it was received, or older than
    //! tactile_max_age_ns.
    std::uint64_t tactile_stale_future = 0;
    std::uint64_t tactile_stale_old = 0;
    std::uint64_t shapes_pushed = 0;
    std::uint64_t generation_changes = 0;
};

struct HandHealth
{
    bool hand_active = false;
    bool glove_live = false;
    bool glove_seen = false; //!< A glove sample has been accepted since the pipeline was created.
    //! The glove is live but its samples carry no joint poses, so no hand can be injected.
    bool kinematics_missing = false;
    bool wrist_tracked = false; //!< The last hand pushed had a tracked wrist.
    bool tactile_live = false;
    bool tactile_seen = false; //!< Tactile has been pushed since the pipeline was created.
    TactileSkip tactile_skip = TactileSkip::None; //!< Last reason, while tactile is not live.
};

//! Everything one glove needs between the SDK and OpenXR: joint states at the
//! encoder sampling time, the hand injected at the glove sample time with the wrist taken from
//! the WristProvider at that same time, and tactile forces in taxel-local axes at their own sampling time.
//! Single-threaded: call from one thread.
class HandPipeline
{
public:
    HandPipeline(Hand hand, PipelineConfig config, PushSink& sink, EventLog* log = nullptr);

    //! Identity and joint names; glove samples before it are dropped. A glove without tactile sensors
    //! (HandMeta::has_tactile) gets neither tactile nor TactileMetadata; its joints and hand go out as usual.
    void set_meta(const HandMeta& meta);
    //! TactileMetadata for gloves with tactile sensors; without it tactile is never pushed.
    void set_shape(std::shared_ptr<const LoadedShape> shape);

    void on_glove(const GloveSample& sample, std::int64_t now_ns);
    //! Without a provider the hand is placed at the stage origin, untracked.
    void set_wrist_provider(WristProvider* provider)
    {
        wrist_ = provider;
    }
    //! Detects stalls and keeps the shape fresh.
    void on_tick(std::int64_t now_ns);

    //! The glove's stream started over (reopened): drops the hand, history and clock anchors.
    void reset(const char* reason);

    HandHealth health(std::int64_t now_ns) const;
    const PipelineStats& stats() const
    {
        return stats_;
    }
    Hand hand() const
    {
        return hand_;
    }

private:
    bool has_tactile() const
    {
        return meta_ && meta_->has_tactile;
    }
    void clear_tactile();
    void clear_shape();
    void push_shape_if_due(std::int64_t now_ns);
    void accept_tactile(const GloveSample& glove, std::int64_t now_ns);
    void write_tactile_rate(std::int64_t now_ns);
    void inject(const GloveSample& sample, std::int64_t t_ns, std::int64_t now_ns);
    void apply(const InjectorLifecycle::Step& step, std::int64_t t_ns);
    void note_tactile_skip(TactileSkip skip, std::int64_t now_ns);

    Hand hand_;
    PipelineConfig config_;
    PushSink& sink_;
    EventLog* log_;

    std::optional<HandMeta> meta_;
    std::shared_ptr<const LoadedShape> shape_;
    //! Unset while no TactileMetadata is published, so the next push is the first one.
    std::optional<std::int64_t> last_shape_push_ns_;

    std::optional<std::uint32_t> generation_;
    TickUnwrapper tactile_ticks_;

    WristProvider* wrist_ = nullptr;
    InjectorLifecycle lifecycle_;

    std::optional<std::int64_t> last_glove_ns_;
    //! Unlike last_glove_ns_, kept by reset(): a reopened glove is not unseen.
    bool glove_seen_ = false;
    std::optional<std::int64_t> last_tactile_push_ns_;
    //! Device time of the last pushed tactile sample: each tactile tick is pushed once.
    std::optional<std::int64_t> last_tactile_raw_ns_;
    std::optional<std::int64_t> last_tactile_sample_ns_;
    bool wrist_tracked_ = false;
    TactileSkip tactile_skip_ = TactileSkip::None;
    //! The latest tactile sample was an invalid repeat, so an expiry now reports Invalid.
    bool tactile_invalid_held_ = false;
    bool kinematics_missing_ = false;

    bool tactile_published_ = false;
    bool shape_published_ = false;
    std::optional<std::int64_t> last_tactile_warning_ns_;
    PipelineStats stats_;

    //! Start of the current tactile_rate window, its counters, and
    //! the largest push latency (now minus sampling time) in it; -1 without a push.
    std::optional<std::int64_t> rate_window_ns_;
    PipelineStats rate_base_;
    std::int64_t rate_lag_max_ns_ = -1;
};

} // namespace revohuman
} // namespace plugins
