// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <schema/joint_state_generated.h>
#include <schema/tactile_array_generated.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace plugins
{
namespace revohuman
{

constexpr std::size_t kJointStateMaxBytes = 4096;
constexpr std::size_t kTactileMaxBytes = 16384;
// TactileDataArray: table, one 12-byte Point per taxel, slack.
static_assert(40 + 12 * kMaxTaxels + 32 <= kTactileMaxBytes, "a full glove must fit one tactile push");

struct PipelineConfig
{
    std::int64_t stall_ns = 50'000'000; //!< No hand push for this long removes the hand.
    //! From the first tick until the glove's first sample, at most this long, the hand exists with no valid joints,
    //! so readers do not show the headset's hand tracking in its place while the glove opens.
    std::int64_t startup_hold_ns = 5'000'000'000;
    std::int64_t shape_resend_ns = 1'000'000'000;
    std::int64_t health_timeout_ns = 500'000'000;
    std::int64_t tactile_max_age_ns = 100'000'000;
};

//! Why tactile is not live.
enum class TactileSkip
{
    None,
    NoShape, //!< The SDK has no taxel layout for the glove.
    NoTactile, //!< The product has no tactile sensors.
    Invalid,
    Uncalibrated,
    CountMismatch,
    Stale,
};

inline const char* tactile_skip_name(TactileSkip skip)
{
    constexpr const char* kNames[] = { "none",         "no_shape",       "no_tactile",  "invalid",
                                       "uncalibrated", "count_mismatch", "stale_sample" };
    static_assert(std::size(kNames) == static_cast<std::size_t>(TactileSkip::Stale) + 1);
    return kNames[static_cast<std::size_t>(skip)];
}

struct HandHealth
{
    bool hand_active = false;
    bool glove_live = false;
    bool glove_seen = false;
    bool wrist_tracked = false;
    bool tactile_live = false;
    bool tactile_seen = false;
    TactileSkip tactile_skip = TactileSkip::None;
};

/*!
 * @brief One glove between the SDK and OpenXR: joint states, the injected hand and taxel-local forces at their own
 * sampling time. Single-threaded.
 *
 * @tparam Sink Where the pipeline delivers; times are CLOCK_MONOTONIC ns, tactile raw times the device sampling
 * clock. It has push_joint_state(buffer, sample_ns), push_tactile(buffer, sample_ns, raw_ns), push_shape(buffer,
 * now_ns), clear_tactile() and clear_shape() (the tracker retains the last sample, so stale data must be removed
 * explicitly), hand_create(), hand_destroy(), and bool hand_push(joints, sample_ns), which places the WRIST-frame
 * joints at the host's wrist pose at the sample time and returns whether it was tracked.
 */
template <typename Sink>
class HandPipeline
{
public:
    HandPipeline(Hand hand, PipelineConfig config, Sink& sink) : hand_(hand), config_(config), sink_(sink)
    {
    }

    //! Glove samples before the first meta are dropped. Forces and layout of an earlier glove on this hand do not
    //! belong to this one.
    void set_meta(const HandMeta& meta)
    {
        meta_ = meta;
        clear_tactile();
        if (meta.has_tactile && meta.tactile_metadata)
            shape_push_ns_.reset(); // Pushed again before the first forces.
        else
            clear_shape();
        if (!meta.has_tactile)
            logger()->info("{}: product {} has no tactile sensors", hand_name(hand_), meta.product_code);
    }

    void on_glove(const GloveSample& sample, std::int64_t now_ns)
    {
        assert(meta_ && "the source delivers HandMeta before its first glove sample");
        follow_generation(sample.generation);
        stream_.last_glove_ns = now_ns;
        glove_seen_ = true;
        // A device clock mapped slightly ahead of CLOCK_MONOTONIC must not put samples in the future.
        const std::int64_t t_ns = std::min(sample.sample_mono_ns, now_ns);

        builder_.Clear();
        std::array<flatbuffers::Offset<core::JointState>, kGloveJointCount> joints;
        for (std::size_t i = 0; i < kGloveJointCount; ++i)
            joints[i] = core::CreateJointStateDirect(builder_, meta_->joint_names[i].c_str(),
                                                     static_cast<float>(sample.joints_rad[i]), 0, 0,
                                                     ((sample.joints_valid_mask >> i) & 1u) != 0);
        builder_.Finish(core::CreateJointStateOutput(
            builder_, builder_.CreateVector(joints.data(), joints.size()), builder_.CreateString(meta_->device_id)));
        sink_.push_joint_state({ builder_.GetBufferPointer(), builder_.GetSize() }, t_ns);

        if (sample.tactile && meta_->has_tactile)
            accept_tactile(*sample.tactile, now_ns);

        if (!stream_.last_hand_push_ns)
        {
            if (!std::exchange(held_push_ns_, std::nullopt)) // A held hand becomes the glove's.
                sink_.hand_create();
            logger()->info("{}: hand injected", hand_name(hand_));
        }
        stream_.last_hand_push_ns = now_ns;
        stream_.wrist_tracked = sink_.hand_push(sample.joint_poses, t_ns);
    }

    //! A tactile sample delivered apart from the glove samples (the SDK's split tactile transport), in sampling
    //! order. A stream's tactile comes either this way or with its glove samples, never both.
    void on_tactile(std::uint32_t generation, const TactileSample& tactile, std::int64_t now_ns)
    {
        assert(meta_ && "the source delivers HandMeta before its first tactile sample");
        follow_generation(generation);
        if (meta_->has_tactile)
            accept_tactile(tactile, now_ns);
    }

    //! Holds the hand until the first glove sample, removes a stalled hand, expires old forces and resends the
    //! TactileMetadata.
    void on_tick(std::int64_t now_ns)
    {
        hold_until_first_sample(now_ns);
        if (stream_.last_hand_push_ns && now_ns - *stream_.last_hand_push_ns > config_.stall_ns)
        {
            sink_.hand_destroy();
            stream_.last_hand_push_ns.reset();
            logger()->info("{}: hand removed (stall)", hand_name(hand_));
        }
        if (stream_.tactile_published && now_ns - *stream_.clock.sample_ns > config_.tactile_max_age_ns)
        {
            clear_tactile();
            if (stream_.tactile_skip == TactileSkip::None)
                note_skip(TactileSkip::Stale, now_ns);
        }
        push_shape_if_due(now_ns);
    }

    //! The glove was reopened: everything derived from its earlier samples goes.
    void reset()
    {
        if (stream_.last_hand_push_ns)
        {
            sink_.hand_destroy();
            logger()->info("{}: hand removed (glove reopened)", hand_name(hand_));
        }
        clear_tactile();
        stream_ = {};
    }

    HandHealth health(std::int64_t now_ns) const
    {
        HandHealth h;
        h.hand_active = stream_.last_hand_push_ns.has_value();
        h.glove_seen = glove_seen_;
        h.glove_live = stream_.last_glove_ns && now_ns - *stream_.last_glove_ns <= config_.health_timeout_ns;
        h.wrist_tracked = h.hand_active && stream_.wrist_tracked;
        // A skipped sample does not end tactile while the last pushed forces are fresh.
        h.tactile_live = stream_.tactile_published && now_ns - *stream_.clock.sample_ns <= config_.tactile_max_age_ns;
        h.tactile_seen = tactile_seen_;
        if (meta_ && !meta_->has_tactile)
            h.tactile_skip = TactileSkip::NoTactile;
        else if (meta_ && !meta_->tactile_metadata)
            h.tactile_skip = TactileSkip::NoShape;
        else if (!h.tactile_live)
            h.tactile_skip = stream_.tactile_skip;
        return h;
    }

private:
    //! The tactile device clock of one stream generation; a new generation can reuse its ticks.
    struct TactileClock
    {
        std::optional<std::uint32_t> last_tick;
        std::int64_t tick_ms = 0; //!< Ticks unwrapped past 2^32.
        std::optional<std::int64_t> sample_ns; //!< Of the last pushed forces.
        std::optional<std::int64_t> raw_ns;
    };
    struct Stream
    {
        std::optional<std::uint32_t> generation;
        TactileClock clock;
        std::optional<std::int64_t> last_glove_ns;
        std::optional<std::int64_t> last_hand_push_ns; //!< Set while the hand exists.
        bool tactile_published = false;
        bool wrist_tracked = false;
        TactileSkip tactile_skip = TactileSkip::None;
        std::optional<std::int64_t> last_skip_log_ns;
    };

    static constexpr std::int64_t kHeldPushPeriodNs = 5'000'000; //!< The glove's sample period.

    static std::shared_ptr<spdlog::logger> logger()
    {
        return isaaccapture::Logger::get("isaaccapture.plugins.revohuman.HandPipeline");
    }

    void hold_until_first_sample(std::int64_t now_ns)
    {
        if (glove_seen_)
            return;
        if (!first_tick_ns_)
            first_tick_ns_ = now_ns;
        if (now_ns - *first_tick_ns_ >= config_.startup_hold_ns)
        {
            if (std::exchange(held_push_ns_, std::nullopt))
            {
                sink_.hand_destroy();
                logger()->info("{}: hand released, no glove sample since start-up", hand_name(hand_));
            }
            return;
        }
        if (!held_push_ns_)
        {
            sink_.hand_create();
            logger()->info("{}: hand held with no valid joints until the glove streams", hand_name(hand_));
        }
        else if (now_ns - *held_push_ns_ < kHeldPushPeriodNs)
            return;
        sink_.hand_push(HandJointPoses{}, now_ns);
        held_push_ns_ = now_ns;
    }

    void follow_generation(std::uint32_t generation)
    {
        if (stream_.generation && *stream_.generation != generation)
        {
            clear_tactile();
            stream_.clock = {};
        }
        stream_.generation = generation;
    }

    void accept_tactile(const TactileSample& tactile, std::int64_t now_ns)
    {
        push_shape_if_due(now_ns); // Before the first forces that need it.
        // Unwrap every tick, pushed or not, so the device clock stays continuous.
        TactileClock& clock = stream_.clock;
        if (clock.last_tick)
            clock.tick_ms += static_cast<std::int32_t>(tactile.device_tick - *clock.last_tick);
        else
            clock.tick_ms = tactile.device_tick;
        clock.last_tick = tactile.device_tick;
        const std::int64_t raw_ns = clock.tick_ms * 1'000'000;
        const auto t_ns =
            tactile.sample_mono_ns ? std::optional(std::min(*tactile.sample_mono_ns, now_ns)) : std::nullopt;
        TactileSkip skip = TactileSkip::None;
        if (!meta_->tactile_metadata)
            skip = TactileSkip::NoShape;
        else if (!tactile.valid)
            skip = TactileSkip::Invalid;
        else if (!tactile.calibrated)
            skip = TactileSkip::Uncalibrated;
        else if (tactile.values.size() != meta_->taxel_count)
            skip = TactileSkip::CountMismatch;
        else if (!t_ns || now_ns - *t_ns > config_.tactile_max_age_ns ||
                 (clock.sample_ns && *t_ns <= *clock.sample_ns) || (clock.raw_ns && raw_ns <= *clock.raw_ns))
            skip = TactileSkip::Stale;
        note_skip(skip, now_ns);
        if (skip != TactileSkip::None)
            return;
        // Taxel-local normal force, in the SDK's newtons.
        builder_.Clear();
        core::Point* forces = nullptr;
        const auto vector = builder_.CreateUninitializedVectorOfStructs(tactile.values.size(), &forces);
        for (std::size_t i = 0; i < tactile.values.size(); ++i)
            forces[i] = core::Point(0.0f, 0.0f, tactile.values[i]);
        builder_.Finish(core::CreateTactileDataArray(builder_, vector));
        sink_.push_tactile({ builder_.GetBufferPointer(), builder_.GetSize() }, *t_ns, raw_ns);
        stream_.tactile_published = true;
        clock.sample_ns = t_ns;
        clock.raw_ns = raw_ns;
        tactile_seen_ = true;
    }

    void note_skip(TactileSkip skip, std::int64_t now_ns)
    {
        if (skip == stream_.tactile_skip)
            return;
        stream_.tactile_skip = skip;
        if (skip == TactileSkip::None)
            logger()->debug("{}: tactile pushed again", hand_name(hand_));
        else if (!stream_.last_skip_log_ns || now_ns - *stream_.last_skip_log_ns >= 1'000'000'000)
        {
            logger()->warn("{}: tactile not pushed ({})", hand_name(hand_), tactile_skip_name(skip));
            stream_.last_skip_log_ns = now_ns;
        }
    }

    void clear_tactile()
    {
        if (std::exchange(stream_.tactile_published, false))
            sink_.clear_tactile();
    }

    void clear_shape()
    {
        if (std::exchange(shape_push_ns_, std::nullopt))
            sink_.clear_shape();
    }

    void push_shape_if_due(std::int64_t now_ns)
    {
        if (!meta_ || !meta_->has_tactile || !meta_->tactile_metadata ||
            (shape_push_ns_ && now_ns - *shape_push_ns_ < config_.shape_resend_ns))
            return;
        sink_.push_shape(*meta_->tactile_metadata, now_ns);
        shape_push_ns_ = now_ns;
    }

    Hand hand_;
    PipelineConfig config_;
    Sink& sink_;
    std::optional<HandMeta> meta_;
    std::optional<std::int64_t> shape_push_ns_; //!< Set while TactileMetadata is published.
    std::optional<std::int64_t> first_tick_ns_;
    std::optional<std::int64_t> held_push_ns_; //!< Set while the hand is held before the first glove sample.
    bool glove_seen_ = false;
    bool tactile_seen_ = false;
    Stream stream_;
    flatbuffers::FlatBufferBuilder builder_{ kTactileMaxBytes };
};

} // namespace revohuman
} // namespace plugins
