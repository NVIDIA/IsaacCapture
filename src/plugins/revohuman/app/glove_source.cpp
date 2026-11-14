// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "glove_source.hpp"

#include <log_bridge/logger.hpp>
#include <revohuman_plugin/sdk.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace plugins
{
namespace revohuman
{
namespace
{

constexpr std::int64_t kDropLogNs = 1'000'000'000;
// The SDK reads the glove's tare record every 100 ms while a request runs and every second otherwise.
constexpr std::int64_t kTarePollNs = 100'000'000;

const std::shared_ptr<spdlog::logger>& logger()
{
    static const auto instance = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.GloveSource");
    return instance;
}

spdlog::level::level_enum to_spdlog(RhLogLevel level)
{
    if (level >= RH_LOG_ERROR)
        return spdlog::level::err;
    if (level >= RH_LOG_WARNING)
        return spdlog::level::warn;
    if (level >= RH_LOG_INFO)
        return spdlog::level::info;
    return spdlog::level::debug;
}

// The SDK's own lines, at their level; it calls this under its lock, so it must not block.
void forward_sdk_log(RhLogLevel level, const char* message, void*)
{
    static const auto sdk = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.sdk");
    sdk->log(to_spdlog(level), "{}", message != nullptr ? message : "");
}

const char* cause_name(RhSyncCause cause)
{
    constexpr const char* kNames[] = { "none",         "clock_not_ready", "clock_degraded", "host_clock_event",
                                       "device_reset", "link_unstable" };
    static_assert(std::size(kNames) == RH_CAUSE_LINK_UNSTABLE + 1);
    const auto i = static_cast<std::size_t>(cause);
    return i < std::size(kNames) ? kNames[i] : "unknown";
}

// False when the stream ended after the status was read (RH_ERR_STATE).
bool stream_query(RhStatus status, const char* what)
{
    if (status == RH_ERR_STATE)
        return false;
    if (status != RH_OK)
        throw std::runtime_error(std::string(what) + " failed: " + sdk_error());
    return true;
}

// Whether rh_autokit_recv or rh_autokit_recv_tactile popped an item. RH_ERR_TIMEOUT is an empty queue and RH_ERR_STATE
// a hand that is not streaming. RH_ERR_DISCONNECTED and RH_ERR_PROTOCOL (incompatible firmware) end the stream: the
// SDK reopens it or gives up on its own thread, and the next status says which.
bool popped(RhStatus status, const char* what)
{
    switch (status)
    {
    case RH_OK:
        return true;
    case RH_ERR_TIMEOUT:
    case RH_ERR_STATE:
    case RH_ERR_DISCONNECTED:
    case RH_ERR_PROTOCOL:
        return false;
    default:
        throw std::runtime_error(std::string(what) + " failed: " + sdk_error());
    }
}

TactileSample to_sample(const RhTactile& t)
{
    TactileSample sample{ .device_tick = t.device_tick,
                          .sample_mono_ns = t.has_sample_time ? std::optional(t.sample_mono_ns) : std::nullopt,
                          .valid = t.valid != 0 };
    std::array<float, RH_TACTILE_MAX_POINTS> newtons;
    const RhStatus status = rh_tactile_newtons(&t, newtons.data(), newtons.size());
    if (status == RH_OK)
    {
        sample.calibrated = true;
        sample.values.assign(newtons.begin(), newtons.begin() + t.point_count);
    }
    else if (status != RH_ERR_UNSUPPORTED) // Raw counts, not forces.
        throw std::runtime_error("rh_tactile_newtons failed: " + sdk_error());
    return sample;
}

// True for 1, 10, 100, ...: repeated problems are logged ever more rarely.
bool worth_logging(std::uint64_t count)
{
    while (count >= 10 && count % 10 == 0)
        count /= 10;
    return count == 1;
}

} // namespace

GloveSource::Sdk::Sdk()
{
    RhInitOptions init;
    rh_init_options_init(&init);
    if (rh_init(&init) != RH_OK)
        throw std::runtime_error("rh_init failed: " + sdk_error());
    rh_set_log_callback(&forward_sdk_log, nullptr);
    logger()->info("RevoHuman C SDK {}", rh_version());
}

GloveSource::Sdk::~Sdk()
{
    rh_shutdown();
    rh_set_log_callback(nullptr, nullptr); // The SDK no longer calls into a plugin that may be unloading.
}

GloveSource::GloveSource()
{
    RhAutoKitOptions options;
    rh_autokit_options_init(&options);
    options.queue_size = kQueueSize;
    // A glove streaming on its own is not reopened as a pair when the other arrives: that stops its samples for a whole
    // open, long enough for its hand to be removed.
    options.upgrade_to_pair = 0;
    // Every tactile sample, not only the latest one at each frame; firmware that cannot split keeps tactile on frames.
    options.kit.glove.split_tactile = 1;
    RhAutoKit* kit = nullptr;
    if (rh_autokit_open(&options, &kit) != RH_OK)
        throw std::runtime_error("rh_autokit_open failed: " + sdk_error());
    kit_.reset(kit);
}

std::optional<GloveSource::Opened> GloveSource::refresh(Hand hand, std::int64_t now_ns)
{
    RhAutoHandStatus status = RH_STRUCT_INIT(RhAutoHandStatus);
    if (rh_autokit_status(kit_.get(), to_rh(hand), &status) != RH_OK)
        throw std::runtime_error("rh_autokit_status failed: " + sdk_error());
    report(hand, status, now_ns);
    Slot& slot = slots_[index_of(hand)];
    slot.status = status;
    if (status.state != RH_AUTO_STREAMING || status.open_count == slot.adopted.open_count)
        return std::nullopt;
    // Identity and info belong to one stream; if it ended or reopened since the status, the next poll tries again.
    Opened opened;
    std::uint32_t identity_open = 0;
    std::uint32_t info_open = 0;
    if (!stream_query(
            rh_autokit_identity(kit_.get(), to_rh(hand), &opened.identity, &identity_open), "rh_autokit_identity") ||
        !stream_query(rh_autokit_glove_info(kit_.get(), to_rh(hand), &opened.info, &info_open), "rh_autokit_glove_info") ||
        identity_open != status.open_count || info_open != status.open_count)
        return std::nullopt;
    return opened;
}

bool GloveSource::streaming(Hand hand) const
{
    const Slot& slot = slots_[index_of(hand)];
    return slot.status.state == RH_AUTO_STREAMING && slot.status.open_count == slot.adopted.open_count;
}

// Logs what changed since the last status; the operator learns of every open, end, give-up and sync change.
void GloveSource::report(Hand hand, const RhAutoHandStatus& now, std::int64_t now_ns)
{
    Slot& slot = slots_[index_of(hand)];
    const RhAutoHandStatus& was = slot.status;
    const char* side = hand_name(hand);
    const std::string error = sdk_text(now.last_error_text);
    // Each retry passes through OPENING, so an error is logged when its text changes, not on every attempt.
    std::string& logged = slot.logged_error;
    if (now.state == RH_AUTO_STREAMING)
        logged.clear();
    else if (now.state == RH_AUTO_FAILED && was.state != RH_AUTO_FAILED)
        logger()->error("{}: giving up on the glove: {}", side, error);
    else if (was.state == RH_AUTO_STREAMING && now.state == RH_AUTO_AWAIT_PAIR)
        logger()->warn("{}: glove stream ended ({}); waiting for both gloves to reopen the pair", side, error);
    else if (was.state == RH_AUTO_STREAMING && now.state == RH_AUTO_ABSENT)
        logger()->warn("{}: glove stream ended ({}); looking for the glove again", side, error);
    else if (now.state == RH_AUTO_ABSENT && !error.empty() && error != logged)
    {
        if (now.absent)
            logger()->info("{}: no glove ({}); looking for it again", side, error);
        else
            logger()->warn("{}: cannot open the glove yet: {}", side, error);
    }
    if (now.state != RH_AUTO_STREAMING && now.state != RH_AUTO_OPENING)
        logged = error;
    if (now.state != RH_AUTO_STREAMING)
        return;
    if (now.pair_lost && !(was.state == RH_AUTO_STREAMING && was.pair_lost))
        logger()->warn("{}: glove open on its own after the pair was lost; the gloves stay unpaired", side);
    if (now.paired && was.state == RH_AUTO_STREAMING && was.paired && now.synchronized != was.synchronized)
    {
        if (now.synchronized)
            logger()->info("{}: gloves hardware-synced again", side);
        else
            logger()->warn("{}: gloves not hardware-synced; pairing in host time", side);
    }
    if (was.state == RH_AUTO_STREAMING ? now.capture_ok != was.capture_ok || now.capture_cause != was.capture_cause :
                                         !now.capture_ok)
    {
        if (now.capture_ok)
            logger()->info("{}: capture gate open", side);
        else
            logger()->warn(
                "{}: capture gate closed ({}); no samples until it opens", side, cause_name(now.capture_cause));
    }
    if (now.open_count != slot.adopted.open_count) // A new stream counts its drops from zero.
        slot.dropped_logged = slot.tactile_dropped_logged = 0;
    if ((now.dropped > slot.dropped_logged || now.tactile_dropped > slot.tactile_dropped_logged) &&
        (!slot.last_drop_log_ns || now_ns - *slot.last_drop_log_ns >= kDropLogNs))
    {
        logger()->warn("{}: the SDK dropped {} glove frames and {} tactile samples; the plugin polls too slowly", side,
                       now.dropped - slot.dropped_logged, now.tactile_dropped - slot.tactile_dropped_logged);
        slot.dropped_logged = now.dropped;
        slot.tactile_dropped_logged = now.tactile_dropped;
        slot.last_drop_log_ns = now_ns;
    }
}

std::optional<HandMeta> GloveSource::adopt(Hand hand, const Opened& opened)
{
    Slot& slot = slots_[index_of(hand)];
    const RhAutoHandStatus& status = slot.status;
    // The SDK starts each stream's tare status from scratch; a request on an older stream ended with it.
    slot.adopted = AdoptedStream{ .open_count = status.open_count };
    AdoptedStream& adopted = slot.adopted;
    if (slot.tare_open && *slot.tare_open != status.open_count)
    {
        logger()->warn(
            "{}: the glove reopened before the tactile tare result was known; check the glove LED", hand_name(hand));
        slot.tare_open.reset();
    }
    HandMeta meta;
    meta.product_code = sdk_text(opened.identity.variant_code);
    meta.has_tactile = (opened.identity.variant_sensors & RH_SENSOR_TACTILE) != 0;
    adopted.has_tactile = meta.has_tactile;
    // The firmware SN is not unique (both gloves of a pair may report the same one), so the hand keys the device and
    // the SN only traces the board.
    const std::string sn = sdk_text(opened.info.sn);
    meta.device_id = std::string("revohuman_") + hand_name(hand) + (sn.empty() ? "" : "_" + sn);
    try
    {
        HandModel model(meta.product_code);
        for (std::size_t i = 0; i < kGloveJointCount; ++i)
        {
            char name[RH_ID_MAX + 16];
            const std::int32_t length = rh_joint_name(to_rh(hand), static_cast<std::uint32_t>(i), name, sizeof(name));
            if (length < 0)
                throw std::runtime_error("rh_joint_name " + std::to_string(i) + " failed: " + sdk_error());
            if (static_cast<std::size_t>(length) >= sizeof(name))
                throw std::runtime_error("rh_joint_name " + std::to_string(i) + " is longer than " +
                                         std::to_string(sizeof(name) - 1) + " characters");
            meta.joint_names[i] = name;
        }
        adopted.model = std::move(model);
    }
    catch (const std::exception& e)
    {
        const std::string& why = adopted.model.emplace<std::string>(e.what());
        logger()->error("{}: cannot use the glove: {}", hand_name(hand), why);
        return std::nullopt;
    }
    const HandModel& model = std::get<HandModel>(adopted.model);
    logger()->info("{}: glove {} (product {}, port {}, firmware {}, {} taxels), hand model {}, {}", hand_name(hand),
                   meta.device_id, meta.product_code, sdk_text(opened.info.port), sdk_text(opened.info.fw_version),
                   opened.info.tactile_point_count, model.id(), status.paired ? "paired" : "on its own");
    if (meta.has_tactile)
    {
        // The SDK release the plugin pins has a taxel layout for the taxel count of every product with tactile
        // sensors; without one the hand is still pushed, and device status reports its tactile as failed.
        try
        {
            meta.tactile_metadata = model.tactile_metadata(hand, opened.info.tactile_point_count);
            meta.taxel_count = opened.info.tactile_point_count;
        }
        catch (const std::runtime_error& e)
        {
            logger()->warn("{}: the glove's tactile is not pushed: {}", hand_name(hand), e.what());
        }
    }
    adopted.product_code = meta.product_code;
    return meta;
}

std::optional<GloveSample> GloveSource::receive(Hand hand)
{
    AdoptedStream& adopted = slots_[index_of(hand)].adopted;
    RhGloveFrame frame = RH_STRUCT_INIT(RhGloveFrame);
    std::uint32_t open_count = 0;
    // A frame of a newer stream than the adopted one is popped and dropped; the next poll adopts that stream.
    while (popped(rh_autokit_recv(kit_.get(), to_rh(hand), 0, &frame, &open_count), "rh_autokit_recv") &&
           open_count == adopted.open_count)
    {
        const HandModel* model = std::get_if<HandModel>(&adopted.model);
        if (model == nullptr)
            continue;
        if (frame.struct_size < sizeof(RhGloveFrame))
            throw std::runtime_error("the RevoHuman C SDK library is older than the header it was built with");
        GloveSample sample;
        // Gated subscriptions deliver only frames with a sampling time.
        if (frame.hand != to_rh(hand) || !(frame.flags & RH_FRAME_FLAG_HAS_SAMPLE_TIME) ||
            frame.tactile.point_count > RH_TACTILE_MAX_POINTS ||
            rh_joints_to_radians(adopted.product_code.c_str(), frame.joints_deg, sample.joints_rad.data()) != RH_OK)
        {
            if (worth_logging(++adopted.rejected))
                logger()->warn("{}: unusable glove frame dropped ({} so far)", hand_name(hand), adopted.rejected);
            continue;
        }
        sample.generation = frame.generation;
        sample.sample_mono_ns = frame.sample_mono_ns;
        sample.joints_valid_mask = frame.joints_valid_mask;
        if ((frame.flags & RH_FRAME_FLAG_HAS_TACTILE) && tactile_on_frames(hand))
            sample.tactile = to_sample(frame.tactile);
        sample.joint_poses = model->joint_poses(hand, sample.joints_rad, sample.joints_valid_mask);
        return sample;
    }
    return std::nullopt;
}

std::optional<GloveSource::SplitTactile> GloveSource::receive_tactile(Hand hand)
{
    AdoptedStream& adopted = slots_[index_of(hand)].adopted;
    RhTactileSample sample = RH_STRUCT_INIT(RhTactileSample);
    std::uint32_t open_count = 0;
    // As receive(); RH_ERR_TIMEOUT for good when the firmware does not split tactile.
    while (popped(rh_autokit_recv_tactile(kit_.get(), to_rh(hand), 0, &sample, &open_count), "rh_autokit_recv_tactile") &&
           open_count == adopted.open_count)
    {
        if (!std::holds_alternative<HandModel>(adopted.model))
            continue;
        if (sample.struct_size < sizeof(RhTactileSample))
            throw std::runtime_error("the RevoHuman C SDK library is older than the header it was built with");
        if (sample.hand != to_rh(hand) || sample.tactile.point_count > RH_TACTILE_MAX_POINTS)
        {
            if (worth_logging(++adopted.tactile_rejected))
                logger()->warn(
                    "{}: unusable tactile sample dropped ({} so far)", hand_name(hand), adopted.tactile_rejected);
            continue;
        }
        if (std::exchange(adopted.tactile_from, TactileFrom::Split) != TactileFrom::Split)
            logger()->info("{}: tactile split from the glove frames, every sample", hand_name(hand));
        return SplitTactile{ sample.generation, to_sample(sample.tactile) };
    }
    return std::nullopt;
}

// A split stream's frames repeat its latest tactile sample: feeding both would push samples out of order, so a stream
// takes its tactile from one place. Frames are used only when a new stream sends no split sample for a while.
bool GloveSource::tactile_on_frames(Hand hand)
{
    AdoptedStream& adopted = slots_[index_of(hand)].adopted;
    if (adopted.tactile_from == TactileFrom::Undecided && ++adopted.tactile_frames >= kSplitWaitFrames)
    {
        adopted.tactile_from = TactileFrom::Frames;
        logger()->info("{}: tactile with the glove frames (the firmware does not split it)", hand_name(hand));
    }
    return adopted.tactile_from == TactileFrom::Frames;
}

void GloveSource::watch_tare(Hand hand, std::int64_t now_ns)
{
    Slot& slot = slots_[index_of(hand)];
    AdoptedStream& adopted = slot.adopted;
    if (!adopted.has_tactile || slot.status.open_count != adopted.open_count ||
        (adopted.last_tare_poll_ns && now_ns - *adopted.last_tare_poll_ns < kTarePollNs))
        return;
    adopted.last_tare_poll_ns = now_ns;
    RhTactileTareStatus now = RH_STRUCT_INIT(RhTactileTareStatus);
    std::uint32_t open_count = 0;
    if (!stream_query(rh_autokit_tactile_tare_status(kit_.get(), to_rh(hand), &now, &open_count),
                      "rh_autokit_tactile_tare_status") ||
        open_count != adopted.open_count)
        return;
    RhTactileTareStatus& was = adopted.tare;
    if (now.version == was.version)
        return;
    const char* side = hand_name(hand);
    if (now.support == RH_TARE_SUPPORT_UNSUPPORTED && was.support != RH_TARE_SUPPORT_UNSUPPORTED)
        logger()->info(
            "{}: the glove cannot tare its tactile sensors on request ({}); hold its button for about 2 s to "
            "tare instead",
            side, sdk_text(now.support_message));
    // This plugin's own request; request_wall_ns is set when its result is.
    if (now.request_wall_ns != 0 && now.request_wall_ns != was.request_wall_ns)
    {
        if (now.request_result == RH_TARE_RESULT_SENT)
            logger()->info("{}: tactile tare: {}", side, sdk_text(now.request_message));
        else
            logger()->warn("{}: tactile tare request: {}", side, sdk_text(now.request_message));
        if (slot.tare_open == open_count)
            slot.tare_open.reset();
    }
    // A tare from the glove's button or another host while this stream runs; otherwise the last tare's zero point
    // after a reconnect or a controller restart. The message says which.
    const bool external = now.origin == RH_TARE_ORIGIN_BUTTON || now.origin == RH_TARE_ORIGIN_OTHER_HOST;
    if (external && was.has_record && (now.sequence != was.sequence || now.outcome != was.outcome))
        logger()->log(now.outcome == RH_TARE_OUTCOME_PENDING || now.outcome == RH_TARE_OUTCOME_SENT ?
                          spdlog::level::info :
                          spdlog::level::warn,
                      "{}: tactile tare: {}", side, sdk_text(now.message));
    else if ((now.stale && !was.stale && now.outcome == RH_TARE_OUTCOME_SENT) || (now.mcu_reset && !was.mcu_reset))
        logger()->info("{}: tactile tare: {}", side, sdk_text(now.message));
    was = now;
}

void GloveSource::request_tare()
{
    for (const Hand hand : kHands)
    {
        Slot& slot = slots_[index_of(hand)];
        const char* side = hand_name(hand);
        if (!streaming(hand) || !slot.adopted.has_tactile)
        {
            logger()->info("{}: tactile tare not requested: the glove {}", side,
                           streaming(hand) ? "has no tactile sensors" : "is not streaming");
            continue;
        }
        std::uint32_t open_count = 0;
        const RhStatus status = rh_autokit_tactile_tare_request(kit_.get(), to_rh(hand), &open_count);
        if (status == RH_OK)
        {
            slot.tare_open = open_count;
            slot.adopted.last_tare_poll_ns.reset();
            logger()->info(
                "{}: tactile tare requested; keep the hand still and the tactile surface untouched until the "
                "command is sent (about 3 s)",
                side);
            continue;
        }
        const RhErrorCode code = rh_last_error_code();
        const std::string error = sdk_error();
        const RhTactileTareStatus& tare = slot.adopted.tare;
        const char* why = nullptr;
        if (status == RH_ERR_UNSUPPORTED)
            why = code == RH_E_TACTILE_UNSUPPORTED ?
                      "the glove has no tactile sensors" :
                      "the glove cannot tare on request; hold its button for about 2 s instead";
        else if (status == RH_ERR_STATE)
            why = tare.in_progress ? "the previous request is still running" :
                  tare.support == RH_TARE_SUPPORT_UNKNOWN ?
                                     "the glove's tare support is not known yet; try again in a few seconds" :
                                     "the glove stream is reconnecting or busy; try again in a few seconds";
        if (why)
            logger()->warn("{}: tactile tare not requested ({}): {}", side, error, why);
        else
            logger()->warn("{}: tactile tare request failed: {}", side, error);
    }
}

std::optional<GloveNote> GloveSource::note(Hand hand) const
{
    const Slot& slot = slots_[index_of(hand)];
    const RhAutoHandStatus& s = slot.status;
    const std::string error = sdk_text(s.last_error_text);
    const std::string because = error.empty() ? "" : ": " + error;
    switch (s.state)
    {
    case RH_AUTO_FAILED:
        return GloveNote{ GloveNote::Kind::Ended, "cannot open the glove" + because };
    case RH_AUTO_AWAIT_PAIR:
        return GloveNote{ GloveNote::Kind::Waiting, "waiting for both gloves to reopen the pair" + because };
    case RH_AUTO_STREAMING:
        break;
    default:
        return GloveNote{ GloveNote::Kind::Waiting, "waiting for the glove" + because };
    }
    if (const auto* why = std::get_if<std::string>(&slot.adopted.model))
        return GloveNote{ GloveNote::Kind::Ended, "cannot use the glove: " + *why };
    if (!s.capture_ok)
        return GloveNote{ GloveNote::Kind::NotReady,
                          std::string("capture gate closed (") + cause_name(s.capture_cause) + ")" };
    if (s.pair_lost)
        return GloveNote{ GloveNote::Kind::Unsynced, "unpaired after the pair was lost" };
    if (s.paired && !s.synchronized)
        return GloveNote{ GloveNote::Kind::Unsynced, "not hardware-synced; pairing in host time" };
    if (!s.paired && slots_[1 - index_of(hand)].status.state == RH_AUTO_STREAMING) // The other glove.
        return GloveNote{ GloveNote::Kind::Unsynced, "not hardware-synced; the gloves opened separately" };
    return std::nullopt;
}

bool GloveSource::finished() const
{
    for (const Slot& slot : slots_)
        if (slot.status.state != RH_AUTO_FAILED && !std::holds_alternative<std::string>(slot.adopted.model))
            return false;
    return true;
}

} // namespace revohuman
} // namespace plugins
