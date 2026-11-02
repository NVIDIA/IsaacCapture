// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/pipeline.hpp"

#include "inc/revohuman_plugin/pose_math.hpp"
#include "inc/revohuman_plugin/serialize.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace plugins
{
namespace revohuman
{
namespace
{

std::shared_ptr<spdlog::logger> pipeline_logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.revohuman.HandPipeline");
}

} // namespace

const char* tactile_skip_name(TactileSkip skip)
{
    switch (skip)
    {
    case TactileSkip::None:
        return "none";
    case TactileSkip::NoShape:
        return "no_shape";
    case TactileSkip::NoTactile:
        return "no_tactile";
    case TactileSkip::Invalid:
        return "invalid";
    case TactileSkip::Uncalibrated:
        return "uncalibrated";
    case TactileSkip::CountMismatch:
        return "count_mismatch";
    case TactileSkip::NoSampleTime:
        return "no_sample_time";
    case TactileSkip::StaleSample:
        return "stale_sample";
    case TactileSkip::NonMonotonicSample:
        return "nonmonotonic_sample";
    }
    return "unknown";
}

HandPipeline::HandPipeline(Hand hand, PipelineConfig config, PushSink& sink, EventLog* log)
    : hand_(hand), config_(config), sink_(sink), log_(log), lifecycle_(config.stall_ns)
{
}

void HandPipeline::set_meta(const HandMeta& meta)
{
    if (meta.hand != hand_)
    {
        return;
    }
    const bool had_tactile = !meta_ || meta_->has_tactile;
    meta_ = meta;
    if (!meta.has_tactile)
    {
        // Another glove may have used this hand before: its forces and layout do not belong to this one.
        clear_tactile();
        clear_shape();
        if (had_tactile)
        {
            pipeline_logger()->info(
                "{}: product {} has no tactile sensors; tactile is not pushed", hand_name(hand_), meta.product_code);
        }
    }
    if (log_)
    {
        log_->write("meta", { { "hand", std::string(hand_name(hand_)) },
                              { "product_code", meta.product_code },
                              { "device_id", meta.device_id },
                              { "has_tactile", meta.has_tactile } });
    }
}

void HandPipeline::set_shape(std::shared_ptr<const LoadedShape> shape)
{
    clear_tactile();
    shape_ = std::move(shape);
    // The new shape replaces the published one at the next push.
    last_shape_push_ns_.reset();
}

void HandPipeline::on_glove(const GloveSample& sample, std::int64_t now_ns)
{
    ++stats_.glove_samples;
    if (!meta_)
    {
        if (stats_.glove_before_meta++ == 0)
        {
            pipeline_logger()->warn("{}: glove samples before the device metadata are dropped", hand_name(hand_));
        }
        return;
    }
    if (generation_ && *generation_ != sample.generation)
    {
        // A new stream generation can reuse device ticks and must not inherit tactile freshness.
        clear_tactile();
        ++stats_.generation_changes;
        tactile_ticks_.reset();
        last_tactile_raw_ns_.reset();
        last_tactile_sample_ns_.reset();
        last_tactile_push_ns_.reset();
        tactile_invalid_held_ = false;
        if (log_)
        {
            log_->write("generation", { { "hand", std::string(hand_name(hand_)) },
                                        { "from", static_cast<std::uint64_t>(*generation_) },
                                        { "to", static_cast<std::uint64_t>(sample.generation) } });
        }
    }
    generation_ = sample.generation;
    last_glove_ns_ = now_ns;
    glove_seen_ = true;

    // A sample cannot be stamped after the time it became available; a device clock mapped slightly
    // ahead of CLOCK_MONOTONIC would otherwise put joint states and the hand in the future.
    std::int64_t t_ns = now_ns;
    if (sample.has_sample_time)
    {
        t_ns = sample.sample_mono_ns;
        if (t_ns > now_ns)
        {
            ++stats_.glove_future_clamped;
            t_ns = now_ns;
        }
    }
    // The device reports when it sent a glove frame, not when it sampled the encoders, so the joint
    // state has no sample time on the device clock; the monotonic sample time stands in for it.
    sink_.push_joint_state(
        serialize_joint_state(meta_->device_id, meta_->joint_names, sample.joints_rad, sample.joints_valid_mask), t_ns,
        t_ns);
    ++stats_.joint_states_pushed;

    if (sample.tactile && has_tactile())
    {
        // The TactileMetadata goes out before the first forces that need it.
        push_shape_if_due(now_ns);
        accept_tactile(sample, now_ns);
    }

    kinematics_missing_ = !sample.kinematics.has_value();
    if (kinematics_missing_)
    {
        if (stats_.glove_without_kinematics++ == 0)
        {
            pipeline_logger()->warn("{}: glove samples carry no joint poses; the hand is not injected", hand_name(hand_));
        }
        return;
    }
    inject(sample, t_ns, now_ns);
}

void HandPipeline::accept_tactile(const GloveSample& glove, std::int64_t now_ns)
{
    ++stats_.tactile_samples;
    const TactileSample& tactile = *glove.tactile;
    const std::int64_t t_ns = tactile.sample_mono_ns;
    // Unwrap every sample, pushed or not, so the device clock stays continuous.
    const std::int64_t raw_ns = ms_ticks_to_ns(tactile_ticks_.unwrap(tactile.device_tick));
    const bool repeat = last_tactile_raw_ns_ && *last_tactile_raw_ns_ == raw_ns;
    tactile_invalid_held_ = false;
    TactileSkip skip = TactileSkip::None;
    if (!shape_)
    {
        skip = TactileSkip::NoShape;
    }
    else if (!tactile.valid)
    {
        // When one read of the tactile board fails, the firmware re-sends the last snapshot with the
        // valid bit cleared, and the next read normally recovers. Downstream still holds that sample,
        // so the re-send is a repeat until the sample's maximum age, and Invalid after it.
        if (repeat && tactile_published_ && now_ns >= *last_tactile_sample_ns_ &&
            now_ns - *last_tactile_sample_ns_ <= config_.tactile_max_age_ns)
        {
            ++stats_.tactile_repeats;
            ++stats_.tactile_invalid_repeats;
            tactile_invalid_held_ = true;
            return;
        }
        skip = TactileSkip::Invalid;
    }
    else if (!tactile.calibrated)
    {
        skip = TactileSkip::Uncalibrated;
    }
    else if (tactile.values.size() != shape_->taxel_count())
    {
        skip = TactileSkip::CountMismatch;
    }
    else if (!tactile.has_sample_time)
    {
        skip = TactileSkip::NoSampleTime;
    }
    else if (t_ns > now_ns || now_ns - t_ns > config_.tactile_max_age_ns)
    {
        skip = TactileSkip::StaleSample;
        ++(t_ns > now_ns ? stats_.tactile_stale_future : stats_.tactile_stale_old);
    }
    else if (repeat)
    {
        // Checked last, so a repeated tick cannot hide bad metadata or an expired sampling time. The
        // invalid re-send above is the one exception, and it is held only until the maximum age.
        ++stats_.tactile_repeats;
        return;
    }
    else if (last_tactile_sample_ns_ && (t_ns <= *last_tactile_sample_ns_ || raw_ns <= *last_tactile_raw_ns_))
    {
        skip = TactileSkip::NonMonotonicSample;
    }
    note_tactile_skip(skip, now_ns);
    if (skip != TactileSkip::None)
    {
        // The last pushed forces keep their own sample time and expire in on_tick(). Removing them
        // here would recreate the collection whenever skipped and valid samples alternate.
        ++stats_.tactile_skipped;
        return;
    }
    // Taxel-local pressure needs no encoder pose or skinning. Geometry consumers must
    // independently validate the hand pose before rotating these forces into another frame.
    sink_.push_tactile(serialize_tactile(taxel_forces(tactile.values)), t_ns, raw_ns);
    tactile_published_ = true;
    ++stats_.tactile_pushed;
    rate_lag_max_ns_ = std::max(rate_lag_max_ns_, now_ns - t_ns);
    last_tactile_push_ns_ = now_ns;
    last_tactile_raw_ns_ = raw_ns;
    last_tactile_sample_ns_ = t_ns;
}

void HandPipeline::clear_tactile()
{
    if (tactile_published_)
    {
        // SchemaTracker retains the last buffer: silence alone cannot invalidate force data.
        sink_.clear_tactile();
        tactile_published_ = false;
    }
    last_tactile_push_ns_.reset();
}

void HandPipeline::clear_shape()
{
    if (shape_published_)
    {
        // Like the forces, the retained metadata stays until it is removed.
        sink_.clear_shape();
        shape_published_ = false;
    }
    last_shape_push_ns_.reset();
}

void HandPipeline::push_shape_if_due(std::int64_t now_ns)
{
    if (!shape_ || !has_tactile() || (last_shape_push_ns_ && now_ns - *last_shape_push_ns_ < config_.shape_resend_ns))
    {
        return;
    }
    if (!last_shape_push_ns_ && log_)
    {
        log_->write("shape", { { "hand", std::string(hand_name(hand_)) },
                               { "sha256", shape_->sha256() },
                               { "taxels", static_cast<std::uint64_t>(shape_->taxel_count()) },
                               { "bytes", static_cast<std::uint64_t>(shape_->bytes().size()) } });
    }
    sink_.push_shape(shape_->bytes(), now_ns);
    shape_published_ = true;
    ++stats_.shapes_pushed;
    last_shape_push_ns_ = now_ns;
}

void HandPipeline::note_tactile_skip(TactileSkip skip, std::int64_t now_ns)
{
    if (skip == tactile_skip_)
    {
        return;
    }
    if (skip == TactileSkip::None)
    {
        pipeline_logger()->debug("{}: tactile pushed again", hand_name(hand_));
    }
    else if (!last_tactile_warning_ns_ || now_ns - *last_tactile_warning_ns_ >= 1'000'000'000)
    {
        pipeline_logger()->warn("{}: tactile not pushed ({})", hand_name(hand_), tactile_skip_name(skip));
        last_tactile_warning_ns_ = now_ns;
    }
    if (log_)
    {
        log_->write(
            "tactile", { { "hand", std::string(hand_name(hand_)) }, { "skip", std::string(tactile_skip_name(skip)) } });
    }
    tactile_skip_ = skip;
}

void HandPipeline::on_tick(std::int64_t now_ns)
{
    apply(lifecycle_.on_tick(now_ns), now_ns);
    if (tactile_published_ && (!last_tactile_sample_ns_ || now_ns < *last_tactile_sample_ns_ ||
                               now_ns - *last_tactile_sample_ns_ > config_.tactile_max_age_ns))
    {
        clear_tactile();
        // An invalid repeat that outlasts its sample is an invalid sample, not just an old one.
        note_tactile_skip(tactile_invalid_held_ ? TactileSkip::Invalid : TactileSkip::StaleSample, now_ns);
    }

    push_shape_if_due(now_ns);

    if (log_ && config_.tactile_rate_period_ns > 0)
    {
        if (!rate_window_ns_)
        {
            rate_window_ns_ = now_ns;
            rate_base_ = stats_;
        }
        else if (now_ns - *rate_window_ns_ >= config_.tactile_rate_period_ns)
        {
            write_tactile_rate(now_ns);
        }
    }
}

void HandPipeline::write_tactile_rate(std::int64_t now_ns)
{
    // Counts over the window since the last event, plus running totals so that a reader can join
    // any two events and tell a restarted plugin (totals going down) from a missed line.
    const PipelineStats& s = stats_;
    const PipelineStats& b = rate_base_;
    const std::uint64_t skipped = s.tactile_skipped - b.tactile_skipped;
    const std::uint64_t stale_future = s.tactile_stale_future - b.tactile_stale_future;
    const std::uint64_t stale_old = s.tactile_stale_old - b.tactile_stale_old;
    log_->write("tactile_rate", { { "hand", std::string(hand_name(hand_)) },
                                  { "t_ns", now_ns },
                                  { "window_s", static_cast<double>(now_ns - *rate_window_ns_) * 1e-9 },
                                  { "samples", s.tactile_samples - b.tactile_samples },
                                  { "pushed", s.tactile_pushed - b.tactile_pushed },
                                  { "repeats", s.tactile_repeats - b.tactile_repeats },
                                  { "invalid_repeats", s.tactile_invalid_repeats - b.tactile_invalid_repeats },
                                  { "skipped", skipped },
                                  { "skip_stale_future", stale_future },
                                  { "skip_stale_old", stale_old },
                                  { "skip_other", skipped - stale_future - stale_old },
                                  { "lag_max_ms", rate_lag_max_ns_ < 0 ? std::numeric_limits<double>::quiet_NaN() :
                                                                         static_cast<double>(rate_lag_max_ns_) * 1e-6 },
                                  { "samples_total", s.tactile_samples },
                                  { "pushed_total", s.tactile_pushed },
                                  { "skipped_total", s.tactile_skipped },
                                  { "invalid_repeats_total", s.tactile_invalid_repeats } });
    rate_window_ns_ = now_ns;
    rate_base_ = stats_;
    rate_lag_max_ns_ = -1;
}

void HandPipeline::inject(const GloveSample& sample, std::int64_t t_ns, std::int64_t now_ns)
{
    // The wrist comes from the host (WristPoseSource, like the other hand plugins). Without a pose
    // the hand stays at the stage origin and is pushed untracked, so its joints still show.
    const WristPose wrist = wrist_ ? wrist_->wrist(hand_, t_ns) : WristPose{};
    const bool tracked = wrist.valid && wrist.tracked;
    apply(lifecycle_.on_frame(now_ns), t_ns);
    const HandKinematics& kinematics = *sample.kinematics;
    HandFrame frame;
    frame.joints = joints_in_stage(wrist.valid ? wrist.pose_stage : Pose{}, kinematics);
    frame.radius = kinematics.radius;
    frame.position_valid_mask = kinematics.position_valid_mask & kMappedHandMask;
    frame.orientation_valid_mask = kinematics.orientation_valid_mask & kMappedHandMask;
    frame.tracked = tracked;
    frame.sample_mono_ns = t_ns;
    sink_.hand_push(frame);
    wrist_tracked_ = tracked;
    ++stats_.hands_pushed;
    if (tracked)
    {
        ++stats_.hands_tracked;
    }
}

void HandPipeline::apply(const InjectorLifecycle::Step& step, std::int64_t t_ns)
{
    if (step.destroy)
    {
        sink_.hand_destroy();
        ++stats_.injector_destroys;
        pipeline_logger()->info("{}: hand removed ({})", hand_name(hand_), step.reason);
        if (log_)
        {
            log_->write("injector", { { "hand", std::string(hand_name(hand_)) },
                                      { "action", std::string("destroy") },
                                      { "reason", step.reason },
                                      { "t_ns", t_ns } });
        }
    }
    if (step.create)
    {
        sink_.hand_create();
        ++stats_.injector_creates;
        pipeline_logger()->info("{}: hand injected ({})", hand_name(hand_), step.reason);
        if (log_)
        {
            log_->write("injector", { { "hand", std::string(hand_name(hand_)) },
                                      { "action", std::string("create") },
                                      { "reason", step.reason },
                                      { "t_ns", t_ns } });
        }
    }
}

void HandPipeline::reset(const char* reason)
{
    if (lifecycle_.active())
    {
        InjectorLifecycle::Step step;
        step.destroy = true;
        step.reason = reason;
        apply(step, 0);
    }
    clear_tactile();
    last_glove_ns_.reset();
    kinematics_missing_ = false;
    lifecycle_ = InjectorLifecycle(config_.stall_ns);
    tactile_ticks_.reset();
    last_tactile_raw_ns_.reset();
    last_tactile_sample_ns_.reset();
    last_tactile_push_ns_.reset();
    tactile_skip_ = TactileSkip::None;
    tactile_invalid_held_ = false;
    generation_.reset();
    wrist_tracked_ = false;
    if (log_)
    {
        log_->write("reset", { { "hand", std::string(hand_name(hand_)) }, { "reason", std::string(reason) } });
    }
}

HandHealth HandPipeline::health(std::int64_t now_ns) const
{
    HandHealth h;
    h.hand_active = lifecycle_.active();
    h.glove_seen = glove_seen_;
    h.glove_live = last_glove_ns_ && now_ns >= *last_glove_ns_ && now_ns - *last_glove_ns_ <= config_.health_timeout_ns;
    h.kinematics_missing = h.glove_live && kinematics_missing_;
    h.wrist_tracked = h.hand_active && wrist_tracked_;
    h.tactile_live = tactile_skip_ == TactileSkip::None && last_tactile_push_ns_ && last_tactile_sample_ns_ &&
                     now_ns >= *last_tactile_sample_ns_ &&
                     now_ns - *last_tactile_sample_ns_ <= config_.tactile_max_age_ns;
    h.tactile_seen = stats_.tactile_pushed > 0;
    if (meta_ && !meta_->has_tactile)
    {
        h.tactile_skip = TactileSkip::NoTactile;
    }
    else if (!h.tactile_live)
    {
        h.tactile_skip = shape_ ? tactile_skip_ : TactileSkip::NoShape;
    }
    return h;
}

} // namespace revohuman
} // namespace plugins
