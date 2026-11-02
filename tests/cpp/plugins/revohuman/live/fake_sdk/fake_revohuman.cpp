// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fake_revohuman.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

struct RhKit
{
    std::array<bool, 2> open{ false, false };
};

struct RhGlove
{
    RhHand hand = RH_HAND_UNKNOWN;
};

struct RhScanList
{
    std::vector<RhDiscoveredGlove> gloves;
};

struct RhSubscription
{
    RhHand hand = RH_HAND_UNKNOWN;
    std::uint32_t capacity = 0;
    std::deque<RhGloveFrame> queue;
    std::uint64_t dropped = 0;
    bool kit_closed = false;
};

namespace
{

struct Failure
{
    RhStatus status = RH_OK;
    RhErrorCode code = RH_E_OK;
    std::string message;
};

struct GloveState
{
    RhGlove handle;
    std::string sn;
    std::string unit_id;
    std::map<std::uint32_t, std::optional<std::string>> joint_names;
    bool present = true;
    bool own_variant = false;
    std::optional<std::string> variant;
    std::optional<Failure> open_failure;
    int open_failures_left = 0;
    std::optional<Failure> subscribe_failure;
    int subscribe_failures_left = 0;
    bool gate_ok = true;
    RhSyncCause cause = RH_CAUSE_NONE;
    std::optional<Failure> end;
    RhKit* kit = nullptr;
    RhSubscription* sub = nullptr;
};

struct State
{
    fake_rh::Calls calls;
    std::optional<std::string> variant = "P1";
    std::optional<Failure> open_failure;
    int open_failures_left = 0;
    std::optional<Failure> pair_open_failure;
    int pair_open_failures_left = 0;
    std::optional<Failure> scan_failure;
    int scan_failures_left = 0;
    std::array<GloveState, 2> gloves;
    std::vector<RhSubscription*> subs;
    std::vector<RhKit*> kits;

    State()
    {
        gloves[0].handle.hand = RH_HAND_LEFT;
        gloves[0].unit_id = "fake-unit-left";
        gloves[1].handle.hand = RH_HAND_RIGHT;
        gloves[1].unit_id = "fake-unit-right";
    }
};

State& state()
{
    static State s;
    return s;
}

// Every entry point holds it; recursive because the controls call the helpers below.
std::recursive_mutex& state_mutex()
{
    static std::recursive_mutex m;
    return m;
}

using Lock = std::lock_guard<std::recursive_mutex>;

// Like the real SDK, the last error belongs to the calling thread.
thread_local RhErrorCode last_code = RH_E_OK;
thread_local std::string last_error;

std::array<std::atomic<int>, 2> open_delay_ms{};
std::atomic<int> opens_started_count{ 0 };

std::size_t slot(RhHand hand)
{
    if (hand != RH_HAND_LEFT && hand != RH_HAND_RIGHT)
    {
        throw std::logic_error("fake_rh: hand must be left or right");
    }
    return hand == RH_HAND_LEFT ? 0 : 1;
}

RhStatus fail(const Failure& f)
{
    last_code = f.code;
    last_error = "E" + std::to_string(f.code) + " " + f.message;
    return f.status;
}

bool take_failure(std::optional<Failure>& failure, int& left, Failure& out)
{
    if (!failure || left <= 0)
    {
        return false;
    }
    --left;
    out = *failure;
    return true;
}

const char* side_of(std::size_t i)
{
    return i == 0 ? "left" : "right";
}

const char* port_of(std::size_t i)
{
    return i == 0 ? "/dev/ttyACM1" : "/dev/ttyACM0";
}

const std::optional<std::string>& variant_of(const State& s, std::size_t i)
{
    return s.gloves[i].own_variant ? s.gloves[i].variant : s.variant;
}

//! Sensors of a product as the SDK's product table lists them: only PTV1 has tactile and a wrist camera.
RhSensorMask sensors_of(const std::string& variant)
{
    const RhSensorMask base = RH_SENSOR_ENCODER | RH_SENSOR_TRACKER;
    return variant == "PTV1" ? base | RH_SENSOR_TACTILE | RH_SENSOR_WRIST_CAMERA : base;
}

std::string copied(const char* s)
{
    return s != nullptr ? std::string(s) : std::string("<null>");
}

template <std::size_t N>
void put(char (&dst)[N], const std::string& src)
{
    std::strncpy(dst, src.c_str(), N - 1);
    dst[N - 1] = '\0';
}

int32_t put_string(const std::string& s, char* buf, size_t buf_len)
{
    if (buf_len > 0)
    {
        const std::size_t n = std::min(s.size(), buf_len - 1);
        std::memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return static_cast<int32_t>(s.size());
}

} // namespace

namespace fake_rh
{

void reset()
{
    Lock lock(state_mutex());
    for (RhSubscription* sub : state().subs)
    {
        delete sub;
    }
    for (RhKit* kit : state().kits)
    {
        delete kit;
    }
    state() = State{};
    for (auto& delay : open_delay_ms)
    {
        delay = 0;
    }
    opens_started_count = 0;
    last_code = RH_E_OK;
    last_error.clear();
}

Calls& calls()
{
    return state().calls;
}

Calls snapshot()
{
    Lock lock(state_mutex());
    return state().calls;
}

int open_kits()
{
    Lock lock(state_mutex());
    return static_cast<int>(state().kits.size());
}

int open_pair_kits()
{
    Lock lock(state_mutex());
    return static_cast<int>(std::count_if(
        state().kits.begin(), state().kits.end(), [](const RhKit* kit) { return kit->open[0] && kit->open[1]; }));
}

void set_variant(std::optional<std::string> code)
{
    Lock lock(state_mutex());
    state().variant = std::move(code);
}

void set_variant(RhHand hand, std::optional<std::string> code)
{
    Lock lock(state_mutex());
    GloveState& g = state().gloves[slot(hand)];
    g.own_variant = true;
    g.variant = std::move(code);
}

void set_glove_ids(RhHand hand, const std::string& sn, const std::string& unit_id)
{
    Lock lock(state_mutex());
    state().gloves[slot(hand)].sn = sn;
    state().gloves[slot(hand)].unit_id = unit_id;
}

void set_joint_name(RhHand hand, std::uint32_t index, std::optional<std::string> name)
{
    Lock lock(state_mutex());
    state().gloves[slot(hand)].joint_names[index] = std::move(name);
}

void set_present(RhHand hand, bool present)
{
    Lock lock(state_mutex());
    state().gloves[slot(hand)].present = present;
}

void fail_next_scan(RhStatus status, RhErrorCode code, const std::string& message, int times)
{
    Lock lock(state_mutex());
    state().scan_failure = Failure{ status, code, message };
    state().scan_failures_left = times;
}

void fail_next_open(RhStatus status, RhErrorCode code, const std::string& message, int times)
{
    Lock lock(state_mutex());
    state().open_failure = Failure{ status, code, message };
    state().open_failures_left = times;
}

void fail_next_pair_open(RhStatus status, RhErrorCode code, const std::string& message, int times)
{
    Lock lock(state_mutex());
    state().pair_open_failure = Failure{ status, code, message };
    state().pair_open_failures_left = times;
}

void fail_next_open(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message, int times)
{
    Lock lock(state_mutex());
    GloveState& g = state().gloves[slot(hand)];
    g.open_failure = Failure{ status, code, message };
    g.open_failures_left = times;
}

void set_open_delay_ms(RhHand hand, int ms)
{
    open_delay_ms[slot(hand)] = ms;
}

int opens_started()
{
    return opens_started_count;
}

void fail_next_subscribe(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message, int times)
{
    Lock lock(state_mutex());
    GloveState& g = state().gloves[slot(hand)];
    g.subscribe_failure = Failure{ status, code, message };
    g.subscribe_failures_left = times;
}

void push(RhHand hand, const RhGloveFrame& frame)
{
    Lock lock(state_mutex());
    RhSubscription* sub = state().gloves[slot(hand)].sub;
    if (sub == nullptr)
    {
        throw std::logic_error("fake_rh::push: the glove has no subscription");
    }
    if (sub->queue.size() >= sub->capacity)
    {
        sub->queue.pop_front();
        ++sub->dropped;
    }
    sub->queue.push_back(frame);
}

void end(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message)
{
    Lock lock(state_mutex());
    state().gloves[slot(hand)].end = Failure{ status, code, message };
}

void set_gate(RhHand hand, bool ok, RhSyncCause cause)
{
    Lock lock(state_mutex());
    state().gloves[slot(hand)].gate_ok = ok;
    state().gloves[slot(hand)].cause = cause;
}

RhGloveFrame frame(RhHand hand, std::int64_t sample_mono_ns)
{
    RhGloveFrame f;
    std::memset(&f, 0, sizeof(f));
    f.struct_size = sizeof(RhGloveFrame);
    f.hand = hand;
    f.generation = 1;
    f.flags = RH_FRAME_FLAG_GATE_OPEN | RH_FRAME_FLAG_HAS_SAMPLE_TIME;
    f.device_tick = static_cast<std::uint32_t>(sample_mono_ns / 1'000'000);
    f.joints_valid_mask = (1u << RH_JOINT_COUNT) - 1u;
    f.sample_wall_ns = sample_mono_ns;
    f.sample_mono_ns = sample_mono_ns;
    for (std::size_t i = 0; i < RH_JOINT_COUNT; ++i)
    {
        f.joints_deg[i] = 10.0 * static_cast<double>(i);
    }
    return f;
}

} // namespace fake_rh

extern "C"
{

    void rh_init_options_init(RhInitOptions* opts)
    {
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(*opts);
        opts->log_level = RH_LOG_WARNING;
    }

    RhStatus rh_init(const RhInitOptions*)
    {
        Lock lock(state_mutex());
        ++state().calls.init;
        return RH_OK;
    }

    const char* rh_version(void)
    {
        return "0.0.0-fake";
    }

    void rh_set_log_callback(RhLogCallback cb, void*)
    {
        Lock lock(state_mutex());
        state().calls.log_callback_set = cb != nullptr;
    }

    void rh_shutdown(void)
    {
        Lock lock(state_mutex());
        State& s = state();
        ++s.calls.shutdown;
        s.calls.kits_open_at_shutdown = static_cast<int>(s.kits.size());
        s.calls.log_callback_set_at_shutdown = s.calls.log_callback_set;
    }

    const char* rh_error_class_name(RhErrorCode code)
    {
        switch (code)
        {
        case RH_E_REVOHUMAN_ERROR:
            return "RevoHumanError";
        case RH_E_DISCONNECTED:
            return "TransportDisconnected";
        case RH_E_PORT_BUSY:
            return "PortBusyError";
        case RH_E_FIRMWARE_INCOMPATIBLE:
            return "FirmwareIncompatible";
        case RH_E_UNREGISTERED_GLOVE:
            return "UnregisteredGloveError";
        case RH_E_PRODUCT_UNIDENTIFIED:
            return "ProductUnidentified";
        default:
            return nullptr;
        }
    }

    RhErrorCode rh_last_error_code(void)
    {
        return last_code;
    }

    int32_t rh_last_error(char* buf, size_t buf_len)
    {
        return put_string(last_error, buf, buf_len);
    }

    void rh_kit_options_init(RhKitOptions* opts)
    {
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(*opts);
        opts->hands = RH_HANDS_BOTH;
        opts->product = "auto";
        opts->strict = 1;
        opts->sync = RH_SYNC_ON;
        opts->rate_hz = 200.0;
        opts->glove.struct_size = sizeof(opts->glove);
    }

    RhStatus rh_kit_open(const RhKitOptions* opts, RhKit** out)
    {
        const std::array<bool, 2> wanted{ opts->hands != RH_HANDS_RIGHT, opts->hands != RH_HANDS_LEFT };
        ++opens_started_count;
        const int delay = std::max(wanted[0] ? open_delay_ms[0].load() : 0, wanted[1] ? open_delay_ms[1].load() : 0);
        if (delay > 0)
        {
            // Outside the lock, so other threads go on while this open is "connecting".
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
        Lock lock(state_mutex());
        State& s = state();
        ++s.calls.kit_open;
        s.calls.hands = opts->hands;
        s.calls.open_hands.push_back(opts->hands);
        s.calls.product = copied(opts->product);
        s.calls.tracker_config_null = opts->tracker_config_json == nullptr;
        s.calls.rate_hz = opts->rate_hz;
        s.calls.sync = opts->sync;
        Failure failure;
        if (take_failure(s.open_failure, s.open_failures_left, failure))
        {
            return fail(failure);
        }
        for (std::size_t i = 0; i < 2; ++i)
        {
            GloveState& g = s.gloves[i];
            if (!wanted[i])
            {
                continue;
            }
            if (!g.present)
            {
                return fail({ RH_ERR_NOT_FOUND, RH_E_REVOHUMAN_ERROR,
                              std::string("RevoHumanError: no ") + side_of(i) + " glove plugged in" });
            }
            if (take_failure(g.open_failure, g.open_failures_left, failure))
            {
                return fail(failure);
            }
            if (g.kit != nullptr)
            {
                return fail({ RH_ERR_BUSY, RH_E_PORT_BUSY,
                              std::string("PortBusyError: the ") + side_of(i) + " glove is in another kit" });
            }
        }
        // Like the SDK, which reads both gloves' device info before it pairs them.
        if (wanted[0] && wanted[1] && variant_of(s, 0) && variant_of(s, 1) && *variant_of(s, 0) != *variant_of(s, 1))
        {
            return fail({ RH_ERR_CONFLICT, RH_E_VARIANT_MISMATCH,
                          "VariantMismatch: the left glove is " + *variant_of(s, 0) + ", the right glove is " +
                              *variant_of(s, 1) });
        }
        if (wanted[0] && wanted[1] && take_failure(s.pair_open_failure, s.pair_open_failures_left, failure))
        {
            return fail(failure);
        }
        auto* kit = new RhKit;
        kit->open = wanted;
        for (std::size_t i = 0; i < 2; ++i)
        {
            if (wanted[i])
            {
                s.gloves[i].kit = kit;
            }
        }
        s.kits.push_back(kit);
        *out = kit;
        return RH_OK;
    }

    void rh_kit_close(RhKit* kit)
    {
        Lock lock(state_mutex());
        State& s = state();
        if (kit == nullptr || std::find(s.kits.begin(), s.kits.end(), kit) == s.kits.end())
        {
            return;
        }
        ++s.calls.kit_close;
        for (RhSubscription* sub : s.subs)
        {
            if (s.gloves[slot(sub->hand)].kit == kit)
            {
                sub->kit_closed = true;
            }
        }
        for (GloveState& g : s.gloves)
        {
            if (g.kit == kit)
            {
                g.kit = nullptr;
            }
        }
        std::erase(s.kits, kit);
        delete kit;
    }

    RhStatus rh_kit_identity(const RhKit* kit, RhProductIdentity* out)
    {
        Lock lock(state_mutex());
        const State& s = state();
        const std::optional<std::string>* variant = &s.variant;
        for (std::size_t i = 0; i < 2; ++i)
        {
            if (kit->open[i] && s.gloves[i].own_variant)
            {
                variant = &s.gloves[i].variant;
            }
        }
        if (!*variant)
        {
            return fail({ RH_ERR_STATE, RH_E_PRODUCT_UNIDENTIFIED, "ProductUnidentified: no evidence" });
        }
        put(out->variant_code, **variant);
        out->variant_sensors = sensors_of(**variant);
        return RH_OK;
    }

    RhGlove* rh_kit_glove(RhKit* kit, RhHand hand)
    {
        Lock lock(state_mutex());
        if (kit == nullptr || (hand != RH_HAND_LEFT && hand != RH_HAND_RIGHT) || !kit->open[slot(hand)])
        {
            return nullptr;
        }
        return &state().gloves[slot(hand)].handle;
    }

    RhStatus rh_glove_info(const RhGlove* glove, RhGloveInfo* out)
    {
        Lock lock(state_mutex());
        if (out->struct_size < sizeof(RhGloveInfo))
        {
            return fail({ RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: struct_size" });
        }
        const GloveState& g = state().gloves[slot(glove->hand)];
        out->hand = glove->hand;
        put(out->port, port_of(slot(glove->hand)));
        put(out->unit_id, g.unit_id);
        put(out->sn, g.sn);
        put(out->fw_version, "fake-1.0");
        out->encoder_count = RH_JOINT_COUNT;
        const auto& variant = variant_of(state(), slot(glove->hand));
        out->tactile_point_count = variant && (sensors_of(*variant) & RH_SENSOR_TACTILE) ? 211 : 0;
        return RH_OK;
    }

    int32_t rh_joint_name(RhHand hand, uint32_t index, char* buf, size_t buf_len)
    {
        if (index >= RH_JOINT_COUNT)
        {
            return fail({ RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: joint index" });
        }
        if (hand == RH_HAND_LEFT || hand == RH_HAND_RIGHT)
        {
            Lock lock(state_mutex());
            const auto& names = state().gloves[slot(hand)].joint_names;
            if (const auto it = names.find(index); it != names.end())
            {
                if (!it->second)
                {
                    return fail({ RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: no joint name" });
                }
                return put_string(*it->second, buf, buf_len);
            }
        }
        const char* side = hand == RH_HAND_LEFT ? "left_" : (hand == RH_HAND_RIGHT ? "right_" : "");
        return put_string(std::string(side) + "fake_joint_" + std::to_string(index), buf, buf_len);
    }

    RhStatus rh_joints_to_radians(const char* product_code, const double deg[RH_JOINT_COUNT], double rad[RH_JOINT_COUNT])
    {
        Lock lock(state_mutex());
        state().calls.joints_product = copied(product_code);
        const double offset = product_code != nullptr && std::strcmp(product_code, "PTV1") == 0 ? 1.0 : 0.0;
        for (std::size_t i = 0; i < RH_JOINT_COUNT; ++i)
        {
            rad[i] = deg[i] / 100.0 + offset;
        }
        return RH_OK;
    }

    void rh_scan_options_init(RhScanOptions* opts)
    {
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(*opts);
    }

    RhStatus rh_scan(const RhScanOptions*, RhScanList** out)
    {
        Lock lock(state_mutex());
        State& s = state();
        ++s.calls.scan;
        *out = nullptr;
        Failure failure;
        if (take_failure(s.scan_failure, s.scan_failures_left, failure))
        {
            return fail(failure);
        }
        auto* list = new RhScanList;
        for (std::size_t i = 0; i < 2; ++i)
        {
            const GloveState& g = s.gloves[i];
            if (!g.present)
            {
                continue;
            }
            RhDiscoveredGlove found;
            std::memset(&found, 0, sizeof(found));
            found.struct_size = sizeof(found);
            found.hand = g.handle.hand;
            put(found.port, port_of(i));
            put(found.unit_id, g.unit_id);
            found.registered = 1;
            list->gloves.push_back(found);
        }
        *out = list;
        return RH_OK;
    }

    size_t rh_scan_count(const RhScanList* list)
    {
        return list != nullptr ? list->gloves.size() : 0;
    }

    RhStatus rh_scan_get(const RhScanList* list, size_t index, RhDiscoveredGlove* out)
    {
        if (list == nullptr || index >= list->gloves.size())
        {
            return fail({ RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: scan index" });
        }
        *out = list->gloves[index];
        return RH_OK;
    }

    void rh_scan_free(RhScanList* list)
    {
        delete list;
    }

    void rh_subscribe_options_init(RhSubscribeOptions* opts)
    {
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(*opts);
        opts->queue_size = 64;
    }

    RhStatus rh_glove_subscribe(
        RhGlove* glove, const RhSubscribeOptions* opts, RhGloveFrameCallback cb, void*, RhSubscription** out)
    {
        Lock lock(state_mutex());
        State& s = state();
        ++s.calls.subscribe;
        s.calls.queue_size = opts->queue_size;
        s.calls.ungated = opts->ungated;
        s.calls.callback_given = cb != nullptr;
        GloveState& g = s.gloves[slot(glove->hand)];
        Failure failure;
        if (take_failure(g.subscribe_failure, g.subscribe_failures_left, failure))
        {
            return fail(failure);
        }
        auto* sub = new RhSubscription;
        sub->hand = glove->hand;
        sub->capacity = opts->queue_size;
        s.subs.push_back(sub);
        g.sub = sub;
        g.end.reset();
        *out = sub;
        return RH_OK;
    }

    RhStatus rh_sub_recv_glove(RhSubscription* sub, int32_t, RhGloveFrame* out)
    {
        Lock lock(state_mutex());
        if (!sub->queue.empty())
        {
            *out = sub->queue.front();
            sub->queue.pop_front();
            return RH_OK;
        }
        if (sub->kit_closed)
        {
            return fail({ RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: the kit is closed" });
        }
        const GloveState& g = state().gloves[slot(sub->hand)];
        if (g.end)
        {
            return fail(*g.end);
        }
        return fail({ RH_ERR_TIMEOUT, RH_E_REVOHUMAN_ERROR, "RevoHumanError: timeout" });
    }

    uint64_t rh_sub_dropped(const RhSubscription* sub)
    {
        Lock lock(state_mutex());
        return sub->dropped;
    }

    void rh_sub_close(RhSubscription* sub)
    {
        Lock lock(state_mutex());
        State& s = state();
        ++s.calls.sub_close;
        if (sub->kit_closed)
        {
            ++s.calls.sub_close_after_kit_close;
        }
        GloveState& g = s.gloves[slot(sub->hand)];
        if (g.sub == sub)
        {
            g.sub = nullptr;
        }
        std::erase(s.subs, sub);
        delete sub;
    }

    bool rh_glove_capture_ok(const RhGlove* glove)
    {
        Lock lock(state_mutex());
        return state().gloves[slot(glove->hand)].gate_ok;
    }

    RhSyncCause rh_glove_capture_cause(const RhGlove* glove)
    {
        Lock lock(state_mutex());
        const GloveState& g = state().gloves[slot(glove->hand)];
        return g.gate_ok ? RH_CAUSE_NONE : g.cause;
    }

} // extern "C"
