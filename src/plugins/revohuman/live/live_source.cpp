// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/live_source.hpp"

#include "inc/revohuman_plugin/rh_frame.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace plugins
{
namespace revohuman
{
namespace
{

//! The SDK's capture gate is polled at most this often.
constexpr std::int64_t kGateCheckPeriodNs = 100'000'000;
//! SDK queue overflows are reported at most this often.
constexpr std::int64_t kDropReportPeriodNs = 1'000'000'000;
//! A hand without an open glove is looked for this often.
constexpr std::int64_t kOpenRetryPeriodNs = 2'000'000'000;

std::shared_ptr<spdlog::logger> live_logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.revohuman.LiveSource");
}

const char* level_name(RhLogLevel level)
{
    if (level >= RH_LOG_ERROR)
    {
        return "error";
    }
    if (level >= RH_LOG_WARNING)
    {
        return "warning";
    }
    return level >= RH_LOG_INFO ? "info" : "debug";
}

// The SDK may log from any of its threads; spdlog loggers are thread-safe. The SDK's log lines are
// kept at debug level: everything the plugin acts on is reported by the plugin itself, with the
// SDK error code.
void forward_sdk_log(RhLogLevel level, const char* message, void*)
{
    static const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.sdk");
    logger->debug("SDK {}: {}", level_name(level), message != nullptr ? message : "");
}

const char* status_name(RhStatus status)
{
    switch (status)
    {
    case RH_OK:
        return "RH_OK";
    case RH_ERR_INVALID_ARG:
        return "RH_ERR_INVALID_ARG";
    case RH_ERR_NOT_INITIALIZED:
        return "RH_ERR_NOT_INITIALIZED";
    case RH_ERR_NOT_FOUND:
        return "RH_ERR_NOT_FOUND";
    case RH_ERR_TIMEOUT:
        return "RH_ERR_TIMEOUT";
    case RH_ERR_PROTOCOL:
        return "RH_ERR_PROTOCOL";
    case RH_ERR_DISCONNECTED:
        return "RH_ERR_DISCONNECTED";
    case RH_ERR_BUFFER_TOO_SMALL:
        return "RH_ERR_BUFFER_TOO_SMALL";
    case RH_ERR_UNSUPPORTED:
        return "RH_ERR_UNSUPPORTED";
    case RH_ERR_CANCELLED:
        return "RH_ERR_CANCELLED";
    case RH_ERR_INVALID_DATA:
        return "RH_ERR_INVALID_DATA";
    case RH_ERR_BUSY:
        return "RH_ERR_BUSY";
    case RH_ERR_STATE:
        return "RH_ERR_STATE";
    case RH_ERR_CONFLICT:
        return "RH_ERR_CONFLICT";
    case RH_ERR_IO:
        return "RH_ERR_IO";
    case RH_ERR_INTERNAL:
        return "RH_ERR_INTERNAL";
    default:
        return "unknown status";
    }
}

//! What the user can do about the errors a glove open or stream meets; nullptr where the class name says it.
const char* error_hint(RhErrorCode code)
{
    switch (code)
    {
    case RH_E_DISCONNECTED:
        return "the glove's USB link dropped";
    case RH_E_PORT_BUSY:
        return "another process holds the glove's serial port";
    case RH_E_FIRMWARE_INCOMPATIBLE:
    case RH_E_TIME_SYNC_INCOMPATIBLE:
    case RH_E_TIME_SYNC_UNSUPPORTED:
        return "the glove firmware is not supported by this SDK; update the firmware";
    case RH_E_UNREGISTERED_GLOVE:
        return "a glove is plugged in, but its MCU serial number does not say which hand it is";
    case RH_E_STREAM_RATE_CONFLICT:
        return "the glove already streams at another rate";
    case RH_E_VARIANT_MISMATCH:
    case RH_E_PRODUCT_UNIDENTIFIED:
    case RH_E_PRODUCT_MISMATCH:
        return "the SDK cannot identify the glove product";
    default:
        return nullptr;
    }
}

//! An absent glove, as opposed to a plugged-in glove the SDK cannot assign to a hand (E1400), which
//! the SDK reports with the same status.
bool glove_absent(RhStatus status, RhErrorCode code)
{
    return status == RH_ERR_NOT_FOUND && code != RH_E_UNREGISTERED_GLOVE;
}

//! A failed SDK call with the status and error code it left, so a failed open can be retried.
struct SdkError : std::runtime_error
{
    SdkError(RhStatus status, RhErrorCode code, const std::string& what)
        : std::runtime_error(what), status(status), code(code)
    {
    }
    RhStatus status;
    RhErrorCode code;
};

void check(RhStatus status, const char* what)
{
    if (status != RH_OK)
    {
        const RhErrorCode code = rh_last_error_code();
        throw SdkError(status, code,
                       std::string("LiveSource: ") + what + " failed (" + status_name(status) + "): " + rh_error_text());
    }
}

template <std::size_t N>
std::string field(const char (&chars)[N])
{
    return std::string(chars, strnlen(chars, N));
}

RhHand to_rh(Hand hand)
{
    return hand == Hand::Left ? RH_HAND_LEFT : RH_HAND_RIGHT;
}

std::size_t slot_index(Hand hand)
{
    return static_cast<std::size_t>(hand);
}

const char* cause_name(RhSyncCause cause)
{
    switch (cause)
    {
    case RH_CAUSE_NONE:
        return "none";
    case RH_CAUSE_CLOCK_NOT_READY:
        return "clock_not_ready";
    case RH_CAUSE_CLOCK_DEGRADED:
        return "clock_degraded";
    case RH_CAUSE_HOST_CLOCK_EVENT:
        return "host_clock_event";
    case RH_CAUSE_DEVICE_RESET:
        return "device_reset";
    case RH_CAUSE_LINK_UNSTABLE:
        return "link_unstable";
    default:
        return "unknown";
    }
}

//! The glove's firmware SN, which every RevoHuman glove has. The SDK's unit id names the USB port the
//! glove is plugged into and the USB serial is not unique to a glove, so neither identifies it; both are
//! logged when it opens. The hand name stands in only if the SDK reports an empty SN, which open_stream()
//! warns about.
std::string device_id_of(const RhGloveInfo& info, Hand hand)
{
    std::string id = field(info.sn);
    if (id.empty())
    {
        id = std::string("revohuman_") + hand_name(hand);
    }
    return id;
}

//! Whether opening a glove may succeed later without a configuration change: no glove plugged in yet,
//! a port held by another process, or a link that dropped while the kit identified the glove. A glove
//! the SDK cannot assign to a hand (E1400) is retried too, since plugging in the other glove or
//! replugging this one can settle it, but it is reported as a problem, not as an absent glove.
//! A wrong product or variant, incompatible firmware and bad options stay wrong.
bool open_retryable(RhStatus status, RhErrorCode code)
{
    switch (status)
    {
    case RH_ERR_NOT_FOUND:
    case RH_ERR_TIMEOUT:
    case RH_ERR_DISCONNECTED:
    case RH_ERR_BUSY:
    case RH_ERR_STATE:
    case RH_ERR_IO:
        return true;
    case RH_ERR_PROTOCOL:
        return code != RH_E_FIRMWARE_INCOMPATIBLE && code != RH_E_TIME_SYNC_INCOMPATIBLE &&
               code != RH_E_TIME_SYNC_UNSUPPORTED && code != RH_E_UNSUPPORTED_FEATURE;
    case RH_ERR_CONFLICT:
        // The SDK asks for a retry when a port re-enumerated during identification.
        return code != RH_E_PRODUCT_MISMATCH && code != RH_E_VARIANT_MISMATCH && code != RH_E_STREAM_RATE_CONFLICT;
    default:
        return false;
    }
}

//! True for 1, 10, 100, ...: repeated problems are logged with decreasing frequency.
bool worth_logging(std::uint64_t count)
{
    while (count >= 10 && count % 10 == 0)
    {
        count /= 10;
    }
    return count == 1;
}

} // namespace

std::string rh_error_text()
{
    const RhErrorCode code = rh_last_error_code();
    // "E1200 StreamOptionsRejected: <text>": the class name is the actual exception class, which may be a
    // subclass without a code of its own; the text after it goes to the debug log only.
    std::string head;
    char buf[512];
    if (rh_last_error(buf, sizeof(buf)) > 0)
    {
        const std::string full(buf);
        live_logger()->debug("SDK error: {}", full);
        head = full.substr(0, full.find(':'));
    }
    if (head.empty())
    {
        const char* class_name = rh_error_class_name(code);
        head = "E" + std::to_string(code);
        if (class_name != nullptr)
        {
            head += std::string(" ") + class_name;
        }
    }
    const char* hint = error_hint(code);
    return hint != nullptr ? head + " (" + hint + ")" : head;
}

namespace
{
// rh_shutdown is process-wide: only the last live source may call it.
std::mutex g_runtime_mutex;
int g_runtime_users = 0;
} // namespace

LiveSource::SdkRuntime::SdkRuntime()
{
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_runtime_users == 0)
    {
        RhInitOptions init;
        rh_init_options_init(&init);
        check(rh_init(&init), "rh_init");
        rh_set_log_callback(&forward_sdk_log, nullptr);
    }
    ++g_runtime_users;
}

LiveSource::SdkRuntime::~SdkRuntime()
{
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (--g_runtime_users == 0)
    {
        // Every handle is closed by now; rh_shutdown closes (and warns about) any the plugin leaked. Once the
        // callback is removed the SDK no longer calls into the plugin, which may be about to unload.
        rh_shutdown();
        rh_set_log_callback(nullptr, nullptr);
    }
}

LiveSource::LiveSource(LiveSourceConfig config, KinematicsFactory kinematics)
    : config_(std::move(config)), kinematics_(std::move(kinematics))
{
    if (config_.queue_size == 0)
    {
        throw std::invalid_argument("LiveSource: queue_size must be positive");
    }
    runtime_.emplace();
    if (kinematics_)
    {
        live_logger()->info("RevoHuman C SDK {}", rh_version());
    }
    else
    {
        live_logger()->info("RevoHuman C SDK {}, no hand kinematics (the hands are not injected)", rh_version());
    }

    // Nothing streams yet, so the first attempts may block here.
    if (both_plugged_in())
    {
        for (HandSlot& slot : slots_)
        {
            ++slot.attempts;
        }
        adopt_pair(open_pair());
    }
    else
    {
        for (const Hand hand : { Hand::Left, Hand::Right })
        {
            HandSlot& slot = slots_[slot_index(hand)];
            ++slot.attempts;
            adopt(hand, slot, open_hand(hand));
        }
    }
    const HandSlot& left = slots_[slot_index(Hand::Left)];
    const HandSlot& right = slots_[slot_index(Hand::Right)];
    if (left.state == HandSlot::State::Failed && right.state == HandSlot::State::Failed)
    {
        throw std::runtime_error("LiveSource: cannot open either glove: left: " + left.error + "; right: " + right.error);
    }
}

LiveSource::OpenOutcome LiveSource::open_hand(Hand hand) const
{
    OpenOutcome out;
    RhKitOptions options;
    rh_kit_options_init(&options);
    options.hands = hand == Hand::Left ? RH_HANDS_LEFT : RH_HANDS_RIGHT;
    // The SDK identifies the product; the hand model follows from it.
    options.product = "auto";
    options.rate_hz = config_.rate_hz;
    // No SDK tracker: the wrist pose comes from the host, like the other hand plugins.
    options.tracker_config_json = nullptr;
    RhKit* kit = nullptr;
    out.status = rh_kit_open(&options, &kit);
    if (out.status != RH_OK)
    {
        // The SDK's last error belongs to this thread, so it is read here.
        out.code = rh_last_error_code();
        out.error = std::string(status_name(out.status)) + ": " + rh_error_text();
        out.result = open_retryable(out.status, out.code) ? OpenResult::Retry : OpenResult::Fatal;
        return out;
    }
    HandStream stream;
    stream.kit = kit;
    try
    {
        open_stream(hand, stream);
    }
    catch (const std::runtime_error& e)
    {
        if (stream.sub != nullptr)
        {
            rh_sub_close(stream.sub);
        }
        rh_kit_close(kit);
        // A glove unplugged between rh_kit_open and rh_glove_subscribe is looked for again.
        const auto* sdk = dynamic_cast<const SdkError*>(&e);
        if (sdk != nullptr)
        {
            out.status = sdk->status;
            out.code = sdk->code;
        }
        out.result = sdk != nullptr && open_retryable(sdk->status, sdk->code) ? OpenResult::Retry : OpenResult::Fatal;
        out.error = e.what();
        return out;
    }
    out.result = OpenResult::Opened;
    out.stream = std::move(stream);
    return out;
}

LiveSource::PairOutcome LiveSource::open_pair() const
{
    PairOutcome out;
    RhKitOptions options;
    rh_kit_options_init(&options);
    options.hands = RH_HANDS_BOTH;
    options.product = "auto";
    // Both gloves join one sync session and are sampled on the same USB frame; the open fails when
    // they cannot be synced.
    options.sync = RH_SYNC_ON;
    options.rate_hz = config_.rate_hz;
    options.tracker_config_json = nullptr;
    RhKit* kit = nullptr;
    if (rh_kit_open(&options, &kit) == RH_OK)
    {
        std::array<HandStream, 2> streams;
        try
        {
            for (const Hand hand : { Hand::Left, Hand::Right })
            {
                HandStream& stream = streams[slot_index(hand)];
                stream.kit = kit;
                stream.paired = true;
                open_stream(hand, stream);
            }
            for (const Hand hand : { Hand::Left, Hand::Right })
            {
                OpenOutcome& hand_out = out.hands[slot_index(hand)];
                hand_out.result = OpenResult::Opened;
                hand_out.stream = std::move(streams[slot_index(hand)]);
            }
            return out;
        }
        catch (const std::runtime_error& e)
        {
            for (const HandStream& stream : streams)
            {
                if (stream.sub != nullptr)
                {
                    rh_sub_close(stream.sub);
                }
            }
            rh_kit_close(kit);
            out.error = e.what();
        }
    }
    else
    {
        out.error = rh_error_text();
    }
    // A glove that cannot be opened with the other may still stream on its own.
    for (const Hand hand : { Hand::Left, Hand::Right })
    {
        out.hands[slot_index(hand)] = open_hand(hand);
    }
    return out;
}

bool LiveSource::both_plugged_in() const
{
    RhScanOptions options;
    rh_scan_options_init(&options);
    // Only reads sysfs, so no port is opened and a glove another process holds still counts.
    options.probe = 0;
    RhScanList* list = nullptr;
    if (rh_scan(&options, &list) != RH_OK)
    {
        live_logger()->debug("cannot scan for gloves: {}", rh_error_text());
        return false;
    }
    std::array<bool, 2> found{ false, false };
    const std::size_t count = rh_scan_count(list);
    for (std::size_t i = 0; i < count; ++i)
    {
        RhDiscoveredGlove glove = RH_STRUCT_INIT(RhDiscoveredGlove);
        // Without a hand (USB serial collision) or a port, rh_kit_open cannot find the glove either.
        if (rh_scan_get(list, i, &glove) != RH_OK || glove.port[0] == '\0')
        {
            continue;
        }
        if (glove.hand == RH_HAND_LEFT)
        {
            found[slot_index(Hand::Left)] = true;
        }
        else if (glove.hand == RH_HAND_RIGHT)
        {
            found[slot_index(Hand::Right)] = true;
        }
    }
    rh_scan_free(list);
    return found[0] && found[1];
}

void LiveSource::open_stream(Hand hand, HandStream& stream) const
{
    auto logger = live_logger();
    HandMeta& meta = stream.meta;
    meta.hand = hand;
    // Joint mapping and hand model both follow the identified product; there is no default to fall back to.
    RhProductIdentity identity = RH_STRUCT_INIT(RhProductIdentity);
    check(rh_kit_identity(stream.kit, &identity), "rh_kit_identity");
    meta.product_code = field(identity.variant_code);
    if (meta.product_code.empty())
    {
        throw std::runtime_error("LiveSource: the SDK opened the glove without identifying its product");
    }
    // A product without tactile sensors (P1) still streams joints and the hand.
    meta.has_tactile = (identity.variant_sensors & RH_SENSOR_TACTILE) != 0;
    if (kinematics_)
    {
        try
        {
            stream.kinematics = kinematics_(meta.product_code);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("LiveSource: no hand kinematics for product '" + meta.product_code +
                                     "': " + e.what());
        }
    }

    stream.glove = rh_kit_glove(stream.kit, to_rh(hand));
    if (stream.glove == nullptr)
    {
        throw std::runtime_error(std::string("LiveSource: the kit opened no ") + hand_name(hand) + " glove");
    }
    RhGloveInfo info = RH_STRUCT_INIT(RhGloveInfo);
    check(rh_glove_info(stream.glove, &info), "rh_glove_info");
    meta.device_id = device_id_of(info, hand);
    if (field(info.sn).empty())
    {
        logger->warn("{}: the SDK reports no SN for the glove; its device id is the hand name", hand_name(hand));
    }
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        char name[RH_ID_MAX + 16];
        const std::int32_t length = rh_joint_name(to_rh(hand), static_cast<std::uint32_t>(i), name, sizeof(name));
        if (length < 0)
        {
            throw std::runtime_error("LiveSource: rh_joint_name " + std::to_string(i) + " failed: " + rh_error_text());
        }
        if (static_cast<std::size_t>(length) >= sizeof(name))
        {
            throw std::runtime_error("LiveSource: rh_joint_name " + std::to_string(i) + " is longer than " +
                                     std::to_string(sizeof(name) - 1) + " characters");
        }
        meta.joint_names[i] = name;
    }
    RhSubscribeOptions subscribe;
    rh_subscribe_options_init(&subscribe);
    subscribe.queue_size = config_.queue_size;
    subscribe.ungated = 0;
    // No callback: frames are pulled on poll(), so the SDK starts no thread for them.
    check(rh_glove_subscribe(stream.glove, &subscribe, nullptr, nullptr, &stream.sub), "rh_glove_subscribe");

    logger->info(
        "{}: glove {} (product '{}', port {}, unit {}, USB serial {}, firmware {}, tactile {}), hand model {}",
        hand_name(hand), meta.device_id, meta.product_code, field(info.port), field(info.unit_id),
        field(info.usb_serial), field(info.fw_version),
        meta.has_tactile ? std::to_string(info.tactile_point_count) + " taxels" : std::string("none on this product"),
        stream.kinematics ? stream.kinematics->model_id() : std::string("none"));
}

void LiveSource::adopt(Hand hand, HandSlot& slot, OpenOutcome outcome)
{
    auto logger = live_logger();
    const bool changed = outcome.status != slot.open_status || outcome.error != slot.error;
    slot.open_status = outcome.status;
    switch (outcome.result)
    {
    case OpenResult::Opened:
        slot.state = HandSlot::State::Open;
        slot.stream = std::move(outcome.stream);
        slot.error.clear();
        slot.meta_due = true;
        ++slot.stats.opens;
        if (slot.attempts > 1)
        {
            logger->info("{}: glove opened after {} attempts", hand_name(hand), slot.attempts);
        }
        break;
    case OpenResult::Retry:
        slot.state = HandSlot::State::Closed;
        slot.error = std::move(outcome.error);
        if (glove_absent(outcome.status, outcome.code))
        {
            // An absent glove is normal, for example when only one hand is used.
            if (changed)
            {
                logger->info("{}: no glove ({}); looking for it every {} s", hand_name(hand), slot.error,
                             kOpenRetryPeriodNs / 1'000'000'000);
            }
        }
        else if (changed || worth_logging(slot.attempts))
        {
            logger->warn("{}: cannot open the glove yet ({} attempts): {}", hand_name(hand), slot.attempts, slot.error);
        }
        break;
    case OpenResult::Fatal:
        slot.state = HandSlot::State::Failed;
        slot.error = std::move(outcome.error);
        logger->error("{}: giving up on the glove: {}", hand_name(hand), slot.error);
        break;
    }
}

void LiveSource::adopt_pair(PairOutcome outcome)
{
    auto logger = live_logger();
    OpenOutcome& left = outcome.hands[slot_index(Hand::Left)];
    OpenOutcome& right = outcome.hands[slot_index(Hand::Right)];
    if (outcome.error.empty())
    {
        pair_kit_ = left.stream->kit;
        pair_error_.clear();
        pair_failures_ = 0;
        logger->info("both gloves open in one kit, sampled on the same USB frame");
    }
    else
    {
        const bool changed = outcome.error != pair_error_;
        pair_error_ = std::move(outcome.error);
        ++pair_failures_;
        if (left.result == OpenResult::Opened && right.result == OpenResult::Opened)
        {
            unpairable_ = true;
            logger->warn(
                "cannot open the gloves in one kit ({}); each streams on its own, not sampled on the same "
                "USB frame",
                pair_error_);
        }
        else if (changed || worth_logging(pair_failures_))
        {
            logger->info("cannot open the gloves in one kit ({} attempts): {}", pair_failures_, pair_error_);
        }
    }
    adopt(Hand::Left, slots_[slot_index(Hand::Left)], std::move(left));
    adopt(Hand::Right, slots_[slot_index(Hand::Right)], std::move(right));
}

void LiveSource::take_opened()
{
    std::array<std::optional<OpenOutcome>, 2> opened;
    std::optional<PairOutcome> pair;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        opened.swap(opened_);
        pair.swap(pair_opened_);
    }
    if (pair)
    {
        adopt_pair(std::move(*pair));
    }
    for (const Hand hand : { Hand::Left, Hand::Right })
    {
        if (opened[slot_index(hand)])
        {
            adopt(hand, slots_[slot_index(hand)], std::move(*opened[slot_index(hand)]));
        }
    }
}

void LiveSource::look_for_gloves(std::int64_t now_ns)
{
    std::array<bool, 2> due{ false, false };
    for (std::size_t i = 0; i < slots_.size(); ++i)
    {
        HandSlot& slot = slots_[i];
        if (slot.state != HandSlot::State::Closed)
        {
            continue;
        }
        // The constructor made the first attempt; the first poll only starts the retry clock.
        due[i] = slot.next_open_ns && now_ns >= *slot.next_open_ns;
        if (!slot.next_open_ns || due[i])
        {
            slot.next_open_ns = now_ns + kOpenRetryPeriodNs;
        }
    }
    if (!due[0] && !due[1])
    {
        return;
    }
    if (slots_[0].state == HandSlot::State::Closed && slots_[1].state == HandSlot::State::Closed && both_plugged_in())
    {
        for (HandSlot& slot : slots_)
        {
            slot.next_open_ns = now_ns + kOpenRetryPeriodNs;
        }
        request_pair();
        return;
    }
    for (const Hand hand : { Hand::Left, Hand::Right })
    {
        if (due[slot_index(hand)])
        {
            request_open(hand, slots_[slot_index(hand)]);
        }
    }
}

void LiveSource::consider_pairing(std::int64_t now_ns)
{
    using State = HandSlot::State;
    const HandSlot& left = slots_[slot_index(Hand::Left)];
    const HandSlot& right = slots_[slot_index(Hand::Right)];
    // A glove plugged in while the other streams opens on its own first, so that the other glove
    // does not restart for two products that cannot share a kit.
    if (left.state != State::Open || right.state != State::Open || left.stream->paired || right.stream->paired ||
        left.stream->meta.product_code != right.stream->meta.product_code || unpairable_)
    {
        return;
    }
    live_logger()->info(
        "both gloves stream on their own; reopening them in one kit to sample them on the same USB "
        "frame");
    for (HandSlot& slot : slots_)
    {
        close_stream(slot, "reopening it in one kit with the other glove", now_ns);
    }
    request_pair();
}

void LiveSource::request_pair()
{
    for (HandSlot& slot : slots_)
    {
        ++slot.attempts;
    }
    if (!config_.background_open)
    {
        adopt_pair(open_pair());
        return;
    }
    for (HandSlot& slot : slots_)
    {
        slot.state = HandSlot::State::Opening;
    }
    start_worker();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pair_wanted_ = true;
    }
    wake_.notify_one();
}

void LiveSource::request_open(Hand hand, HandSlot& slot)
{
    ++slot.attempts;
    if (!config_.background_open)
    {
        adopt(hand, slot, open_hand(hand));
        return;
    }
    slot.state = HandSlot::State::Opening;
    start_worker();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        open_wanted_[slot_index(hand)] = true;
    }
    wake_.notify_one();
}

void LiveSource::release(HandStream& stream)
{
    // Subscriptions close before their kit.
    if (stream.sub != nullptr)
    {
        rh_sub_close(stream.sub);
        stream.sub = nullptr;
    }
    stream.glove = nullptr;
    RhKit* kit = std::exchange(stream.kit, nullptr);
    // The other glove may still stream from a two-hand kit.
    if (kit != nullptr && !stream.paired)
    {
        close_kit(kit);
    }
}

void LiveSource::release_pair()
{
    for (const HandSlot& slot : slots_)
    {
        if (slot.stream && slot.stream->paired)
        {
            return;
        }
    }
    if (RhKit* kit = std::exchange(pair_kit_, nullptr))
    {
        close_kit(kit);
    }
}

void LiveSource::close_kit(RhKit* kit)
{
    if (!config_.background_open || stop_)
    {
        rh_kit_close(kit);
        return;
    }
    // Closing stops the device stream, which can wait on a glove that is going away.
    start_worker();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        to_close_.push_back(kit);
    }
    wake_.notify_one();
}

void LiveSource::start_worker()
{
    if (!worker_.joinable())
    {
        worker_ = std::thread(&LiveSource::worker_loop, this);
    }
}

void LiveSource::worker_loop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;)
    {
        wake_.wait(
            lock, [this] { return stop_ || !to_close_.empty() || pair_wanted_ || open_wanted_[0] || open_wanted_[1]; });
        if (stop_)
        {
            return;
        }
        // Closes first: a glove looked for again needs its old kit to let go of the port.
        if (!to_close_.empty())
        {
            RhKit* kit = to_close_.back();
            to_close_.pop_back();
            lock.unlock();
            rh_kit_close(kit);
            lock.lock();
            continue;
        }
        if (pair_wanted_)
        {
            pair_wanted_ = false;
            lock.unlock();
            PairOutcome outcome = open_pair();
            lock.lock();
            pair_opened_ = std::move(outcome);
            continue;
        }
        const Hand hand = open_wanted_[slot_index(Hand::Left)] ? Hand::Left : Hand::Right;
        open_wanted_[slot_index(hand)] = false;
        lock.unlock();
        OpenOutcome outcome = open_hand(hand);
        lock.lock();
        opened_[slot_index(hand)] = std::move(outcome);
    }
}

LiveSource::~LiveSource()
{
    if (worker_.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        // An open in progress finishes first; requests not started yet are dropped.
        worker_.join();
    }
    stop_ = true;
    // Every subscription closes before its kit, so a two-hand kit closes after both of its gloves.
    std::vector<RhKit*> pair_kits;
    const auto let_go = [&pair_kits, this](HandStream& stream)
    {
        if (stream.paired && stream.kit != nullptr &&
            std::find(pair_kits.begin(), pair_kits.end(), stream.kit) == pair_kits.end())
        {
            pair_kits.push_back(stream.kit);
        }
        release(stream);
    };
    for (HandSlot& slot : slots_)
    {
        if (slot.stream)
        {
            let_go(*slot.stream);
        }
    }
    for (auto& outcome : opened_)
    {
        if (outcome && outcome->stream)
        {
            let_go(*outcome->stream);
        }
    }
    if (pair_opened_)
    {
        for (OpenOutcome& outcome : pair_opened_->hands)
        {
            if (outcome.stream)
            {
                let_go(*outcome.stream);
            }
        }
    }
    pair_kit_ = nullptr;
    for (RhKit* kit : pair_kits)
    {
        rh_kit_close(kit);
    }
    for (RhKit* kit : to_close_)
    {
        rh_kit_close(kit);
    }
}

SourceBatch LiveSource::poll(std::int64_t now_ns)
{
    SourceBatch batch;
    if (!finished_)
    {
        take_opened();
        look_for_gloves(now_ns);
        // Before service(), so a glove reopened in one kit right after it opened on its own
        // delivers only the HandMeta of the kit it streams from.
        consider_pairing(now_ns);
        for (const Hand hand : { Hand::Left, Hand::Right })
        {
            service(hand, slots_[slot_index(hand)], now_ns, batch);
        }
        if (slots_[0].state == HandSlot::State::Failed && slots_[1].state == HandSlot::State::Failed)
        {
            finished_ = true;
            live_logger()->error("both gloves given up; no more samples");
        }
    }
    released_ += batch.records.size();
    batch.finished = finished_;
    return batch;
}

void LiveSource::service(Hand hand, HandSlot& slot, std::int64_t now_ns, SourceBatch& batch)
{
    if (slot.state != HandSlot::State::Open)
    {
        return;
    }
    if (slot.meta_due)
    {
        batch.records.push_back(slot.stream->meta);
        batch.hand_restarted[slot_index(hand)] = slot.streamed;
        slot.streamed = true;
        slot.meta_due = false;
    }
    if (std::optional<std::string> ended = pull(hand, slot, batch.records))
    {
        end_stream(hand, slot, *ended, now_ns);
        return;
    }
    check_gate(hand, *slot.stream, now_ns);
    check_dropped(hand, slot, now_ns);
}

std::optional<std::string> LiveSource::pull(Hand hand, HandSlot& slot, std::vector<SourceRecord>& out)
{
    HandStream& stream = *slot.stream;
    const std::string& product_code = stream.meta.product_code;
    // At most one queue's worth per poll, so a fast producer cannot hold up the tick loop.
    for (std::uint32_t n = 0; n < config_.queue_size; ++n)
    {
        RhGloveFrame frame = RH_STRUCT_INIT(RhGloveFrame);
        const RhStatus status = rh_sub_recv_glove(stream.sub, 0, &frame);
        if (status == RH_ERR_TIMEOUT)
        {
            return std::nullopt;
        }
        if (status != RH_OK)
        {
            return rh_error_text();
        }
        ++slot.stats.frames;

        std::array<double, kGloveJointCount> joints_rad{};
        if (rh_joints_to_radians(product_code.c_str(), frame.joints_deg, joints_rad.data()) != RH_OK)
        {
            reject(hand, slot, "rh_joints_to_radians failed: " + rh_error_text());
            continue;
        }
        GloveSample sample;
        try
        {
            sample = from_rh_frame(frame, joints_rad);
        }
        catch (const std::invalid_argument& e)
        {
            reject(hand, slot, e.what());
            continue;
        }
        if (sample.hand != hand)
        {
            reject(hand, slot, std::string("frame of the ") + hand_name(sample.hand) + " glove");
            continue;
        }
        if (!sample.has_sample_time)
        {
            // Gated subscriptions only carry frames with a sampling time; without one the sample
            // cannot be aligned with anything.
            reject(hand, slot, "gated frame without a sampling time");
            continue;
        }
        if (stream.kinematics)
        {
            sample.kinematics = stream.kinematics->solve(hand, sample.joints_rad, sample.joints_valid_mask);
        }
        if (!sample.kinematics)
        {
            ++slot.stats.without_kinematics;
        }
        out.push_back(std::move(sample));
    }
    return std::nullopt;
}

void LiveSource::reject(Hand hand, HandSlot& slot, const std::string& reason)
{
    if (worth_logging(++slot.stats.rejected))
    {
        live_logger()->warn("{}: glove frame dropped ({} so far): {}", hand_name(hand), slot.stats.rejected, reason);
    }
}

void LiveSource::check_gate(Hand hand, HandStream& stream, std::int64_t now_ns)
{
    if (stream.last_gate_check_ns != 0 && now_ns - stream.last_gate_check_ns < kGateCheckPeriodNs)
    {
        return;
    }
    stream.last_gate_check_ns = now_ns;
    const bool ok = rh_glove_capture_ok(stream.glove);
    const RhSyncCause cause = ok ? RH_CAUSE_NONE : rh_glove_capture_cause(stream.glove);
    if (ok == stream.gate_ok && cause == stream.gate_cause)
    {
        return;
    }
    if (ok)
    {
        live_logger()->info("{}: capture gate open", hand_name(hand));
    }
    else
    {
        live_logger()->warn("{}: capture gate closed ({}); the glove delivers no samples until it opens",
                            hand_name(hand), cause_name(cause));
    }
    stream.gate_ok = ok;
    stream.gate_cause = cause;
}

void LiveSource::check_dropped(Hand hand, HandSlot& slot, std::int64_t now_ns)
{
    HandStream& stream = *slot.stream;
    const std::uint64_t dropped = rh_sub_dropped(stream.sub);
    slot.stats.sdk_dropped = slot.dropped_before + dropped;
    if (dropped == stream.reported_dropped ||
        (stream.last_drop_report_ns != 0 && now_ns - stream.last_drop_report_ns < kDropReportPeriodNs))
    {
        return;
    }
    live_logger()->warn("{}: the SDK dropped {} glove frames (queue of {}); the plugin polls too slowly",
                        hand_name(hand), dropped - stream.reported_dropped, config_.queue_size);
    stream.reported_dropped = dropped;
    stream.last_drop_report_ns = now_ns;
}

void LiveSource::end_stream(Hand hand, HandSlot& slot, const std::string& reason, std::int64_t now_ns)
{
    auto logger = live_logger();
    const bool paired = slot.stream->paired;
    ++slot.stats.ends;
    // A glove plugged in again may be one that can share a kit with the other.
    unpairable_ = false;
    close_stream(slot, "the glove stream ended: " + reason, now_ns);
    if (!paired)
    {
        logger->warn("{}: glove stream ended ({}); looking for the glove again", hand_name(hand), reason);
        return;
    }
    logger->warn("{}: glove stream ended ({}); looking for both gloves again", hand_name(hand), reason);
    // The SDK cannot take one glove out of a two-hand kit, so the other glove closes with it.
    const Hand mate = hand == Hand::Left ? Hand::Right : Hand::Left;
    HandSlot& other = slots_[slot_index(mate)];
    if (other.state == HandSlot::State::Open && other.stream->paired)
    {
        ++other.stats.ends;
        close_stream(other, std::string("the paired ") + hand_name(hand) + " glove's stream ended: " + reason, now_ns);
        logger->warn("{}: closing the glove with the paired {} glove; looking for both gloves again", hand_name(mate),
                     hand_name(hand));
    }
    release_pair();
}

void LiveSource::close_stream(HandSlot& slot, std::string error, std::int64_t now_ns)
{
    slot.stats.sdk_dropped = slot.dropped_before + rh_sub_dropped(slot.stream->sub);
    slot.dropped_before = slot.stats.sdk_dropped;
    release(*slot.stream);
    slot.stream.reset();
    slot.state = HandSlot::State::Closed;
    slot.open_status = RH_OK;
    slot.error = std::move(error);
    slot.meta_due = false;
    slot.attempts = 0;
    slot.next_open_ns = now_ns + kOpenRetryPeriodNs;
}

std::optional<Source::GloveNote> LiveSource::glove_note(Hand hand) const
{
    const HandSlot& slot = slots_[slot_index(hand)];
    switch (slot.state)
    {
    case HandSlot::State::Failed:
        return GloveNote{ .kind = GloveNote::Kind::Ended, .text = "cannot open the glove: " + slot.error };
    case HandSlot::State::Closed:
    case HandSlot::State::Opening:
        return GloveNote{ .kind = GloveNote::Kind::Waiting, .text = "waiting for the glove: " + slot.error };
    case HandSlot::State::Open:
        break;
    }
    if (!slot.stream->gate_ok)
    {
        return GloveNote{ .kind = GloveNote::Kind::NotReady,
                          .text = std::string("capture gate closed (") + cause_name(slot.stream->gate_cause) + ")" };
    }
    return std::nullopt;
}

bool LiveSource::opened(Hand hand) const
{
    return slots_[slot_index(hand)].state == HandSlot::State::Open;
}

bool LiveSource::paired(Hand hand) const
{
    const HandSlot& slot = slots_[slot_index(hand)];
    return slot.state == HandSlot::State::Open && slot.stream->paired;
}

std::string LiveSource::product_code(Hand hand) const
{
    const HandSlot& slot = slots_[slot_index(hand)];
    return slot.stream ? slot.stream->meta.product_code : std::string();
}

std::string LiveSource::model_id(Hand hand) const
{
    const HandSlot& slot = slots_[slot_index(hand)];
    return slot.stream && slot.stream->kinematics ? slot.stream->kinematics->model_id() : std::string();
}

LiveSource::HandStats LiveSource::stats(Hand hand) const
{
    return slots_[slot_index(hand)].stats;
}

} // namespace revohuman
} // namespace plugins
