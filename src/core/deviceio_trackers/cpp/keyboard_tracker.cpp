// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/keyboard_tracker.hpp"

#include <chrono>
#include <unordered_map>
#include <utility>

namespace core
{

namespace
{

// Same clock as core::os_monotonic_now_ns() (CLOCK_MONOTONIC on Linux via libstdc++, QPC on
// Windows via MSVC); read through <chrono> because deviceio_trackers stays off oxr_utils. Replace
// with os_monotonic_now_ns() once it moves to a target both libraries can use.
int64_t monotonic_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Chromium's physical key table (deps/third_party/chromium/include/chromium/dom_code_data.inc, BSD-3-Clause): one row
// per key with its USB HID, evdev, XKB, Windows and macOS codes and its W3C KeyboardEvent.code.
// Rows without a code string, or that the Linux kernel does not map (evdev 0), are skipped.
struct DomCodeRow
{
    const char* code;
    uint16_t evdev;
};

#define DOM_CODE(usb, evdev, xkb, win, mac, code, id)                                                                  \
    DomCodeRow                                                                                                         \
    {                                                                                                                  \
        code, evdev                                                                                                    \
    }
#define DOM_CODE_DECLARATION constexpr DomCodeRow kDomCodeRows[] =
#include <chromium/dom_code_data.inc>
#undef DOM_CODE
#undef DOM_CODE_DECLARATION

const std::unordered_map<std::string_view, uint16_t>& w3c_to_evdev()
{
    static const std::unordered_map<std::string_view, uint16_t> table = []
    {
        std::unordered_map<std::string_view, uint16_t> out;
        for (const auto& key : keyboard_key_codes())
            out.emplace(key.w3c_code, key.evdev_code);
        return out;
    }();
    return table;
}

const std::unordered_map<uint16_t, std::string_view>& evdev_to_w3c()
{
    // The first row wins when several codes share an evdev code.
    static const std::unordered_map<uint16_t, std::string_view> table = []
    {
        std::unordered_map<uint16_t, std::string_view> out;
        for (const auto& key : keyboard_key_codes())
            out.emplace(key.evdev_code, key.w3c_code);
        return out;
    }();
    return table;
}

} // namespace

const std::vector<KeyCodeName>& keyboard_key_codes()
{
    static const std::vector<KeyCodeName> key_codes = []
    {
        std::vector<KeyCodeName> out;
        for (const auto& row : kDomCodeRows)
        {
            if (row.code != nullptr && row.evdev != 0)
                out.push_back({ row.code, row.evdev });
        }
        return out;
    }();
    return key_codes;
}

std::optional<std::string_view> w3c_code_from_evdev(uint16_t evdev_code)
{
    const auto& table = evdev_to_w3c();
    const auto it = table.find(evdev_code);
    if (it == table.end())
        return std::nullopt;
    return it->second;
}

std::optional<uint16_t> evdev_code_from_w3c(std::string_view w3c_code)
{
    const auto& table = w3c_to_evdev();
    const auto it = table.find(w3c_code);
    if (it == table.end())
    {
        return std::nullopt;
    }
    return it->second;
}

// ============================================================================
// KeyboardInputState
// ============================================================================

uint64_t KeyboardInputState::add_provider()
{
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t id = next_provider_id_++;
    held_by_provider_.emplace(id, std::set<uint16_t>{});
    return id;
}

void KeyboardInputState::remove_provider(uint64_t provider_id, int64_t timestamp_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = held_by_provider_.find(provider_id);
    if (it == held_by_provider_.end())
    {
        return;
    }
    std::set<uint16_t> keys = std::move(it->second);
    held_by_provider_.erase(it);
    release_keys_locked(keys, timestamp_ns);
}

void KeyboardInputState::key_down(uint64_t provider_id, uint16_t code, int64_t timestamp_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = held_by_provider_.find(provider_id);
    if (it == held_by_provider_.end() || it->second.count(code) != 0)
    {
        return;
    }
    const bool newly_held = !held_locked(code);
    it->second.insert(code);
    if (newly_held)
    {
        push_event_locked({ timestamp_ns, code, true });
    }
}

void KeyboardInputState::key_up(uint64_t provider_id, uint16_t code, int64_t timestamp_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = held_by_provider_.find(provider_id);
    if (it == held_by_provider_.end() || it->second.erase(code) == 0)
    {
        return;
    }
    if (!held_locked(code))
    {
        push_event_locked({ timestamp_ns, code, false });
    }
}

void KeyboardInputState::tap(uint64_t provider_id, uint16_t code, int64_t timestamp_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (held_by_provider_.count(provider_id) == 0 || held_locked(code))
    {
        return;
    }
    push_event_locked({ timestamp_ns, code, true });
    push_event_locked({ timestamp_ns, code, false });
}

void KeyboardInputState::release_all(uint64_t provider_id, int64_t timestamp_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = held_by_provider_.find(provider_id);
    if (it == held_by_provider_.end())
    {
        return;
    }
    std::set<uint16_t> keys = std::exchange(it->second, {});
    release_keys_locked(keys, timestamp_ns);
}

bool KeyboardInputState::held_locked(uint16_t code) const
{
    for (const auto& [id, keys] : held_by_provider_)
    {
        if (keys.count(code) != 0)
        {
            return true;
        }
    }
    return false;
}

std::set<uint16_t> KeyboardInputState::held_keys_locked() const
{
    std::set<uint16_t> held;
    for (const auto& [id, keys] : held_by_provider_)
    {
        held.insert(keys.begin(), keys.end());
    }
    return held;
}

void KeyboardInputState::release_keys_locked(std::set<uint16_t>& keys, int64_t timestamp_ns)
{
    // `keys` were just taken from their provider, so a key still held is held by another one.
    for (uint16_t code : keys)
    {
        if (!held_locked(code))
        {
            push_event_locked({ timestamp_ns, code, false });
        }
    }
}

KeyboardInputState::Snapshot KeyboardInputState::drain()
{
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snapshot;
    std::set<uint16_t> held = held_keys_locked();
    snapshot.held_keys.assign(held.begin(), held.end());
    if (resync_)
    {
        // The log was dropped: report the minimal change since the previous drain instead.
        for (uint16_t code : drained_held_)
        {
            if (held.count(code) == 0)
            {
                snapshot.events.push_back({ resync_timestamp_ns_, code, false });
            }
        }
        for (uint16_t code : held)
        {
            if (drained_held_.count(code) == 0)
            {
                snapshot.events.push_back({ resync_timestamp_ns_, code, true });
            }
        }
        resync_ = false;
    }
    else
    {
        snapshot.events = std::exchange(pending_, {});
    }
    drained_held_ = std::move(held);
    snapshot.provider_count = held_by_provider_.size();
    return snapshot;
}

void KeyboardInputState::start_session()
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Start from the keys held now, reported as presses; later transitions queue normally, so a
    // tap before the first drain still arrives.
    drained_held_.clear();
    pending_.clear();
    resync_ = false;
    const int64_t now_ns = monotonic_now_ns();
    for (uint16_t code : held_keys_locked())
    {
        pending_.push_back({ now_ns, code, true });
    }
}

void KeyboardInputState::push_event_locked(const Event& event)
{
    // Held state is authoritative, so dropping the log can never leave a key stuck; drain()
    // rebuilds the transitions from it.
    if (resync_)
    {
        resync_timestamp_ns_ = event.timestamp_ns;
        return;
    }
    if (pending_.size() >= MAX_PENDING_EVENTS)
    {
        resync_locked(event.timestamp_ns);
        return;
    }
    pending_.push_back(event);
}

void KeyboardInputState::resync_locked(int64_t timestamp_ns)
{
    pending_.clear();
    resync_ = true;
    resync_timestamp_ns_ = timestamp_ns;
}

// ============================================================================
// KeyboardProvider
// ============================================================================

KeyboardProvider::KeyboardProvider(Key, std::shared_ptr<KeyboardInputState> state)
    : state_(std::move(state)), id_(state_->add_provider())
{
}

KeyboardProvider::~KeyboardProvider()
{
    close();
}

// After close() the state no longer knows this provider's id, so every call below is a no-op.

void KeyboardProvider::key_down(uint16_t evdev_code, std::optional<int64_t> timestamp_ns)
{
    if (evdev_code < kKeyboardKeyCodeCount)
    {
        state_->key_down(id_, evdev_code, timestamp_ns.value_or(monotonic_now_ns()));
    }
}

void KeyboardProvider::key_up(uint16_t evdev_code, std::optional<int64_t> timestamp_ns)
{
    if (evdev_code < kKeyboardKeyCodeCount)
    {
        state_->key_up(id_, evdev_code, timestamp_ns.value_or(monotonic_now_ns()));
    }
}

void KeyboardProvider::key_down(std::string_view w3c_code, std::optional<int64_t> timestamp_ns)
{
    if (const auto code = evdev_code_from_w3c(w3c_code))
    {
        key_down(*code, timestamp_ns);
    }
}

void KeyboardProvider::key_up(std::string_view w3c_code, std::optional<int64_t> timestamp_ns)
{
    if (const auto code = evdev_code_from_w3c(w3c_code))
    {
        key_up(*code, timestamp_ns);
    }
}

void KeyboardProvider::tap(uint16_t evdev_code, std::optional<int64_t> timestamp_ns)
{
    if (evdev_code < kKeyboardKeyCodeCount)
    {
        state_->tap(id_, evdev_code, timestamp_ns.value_or(monotonic_now_ns()));
    }
}

void KeyboardProvider::tap(std::string_view w3c_code, std::optional<int64_t> timestamp_ns)
{
    if (const auto code = evdev_code_from_w3c(w3c_code))
    {
        tap(*code, timestamp_ns);
    }
}

void KeyboardProvider::release_all(std::optional<int64_t> timestamp_ns)
{
    state_->release_all(id_, timestamp_ns.value_or(monotonic_now_ns()));
}

void KeyboardProvider::close()
{
    state_->remove_provider(id_, monotonic_now_ns());
}

// ============================================================================
// KeyboardTracker
// ============================================================================

KeyboardTracker::KeyboardTracker() : state_(std::make_shared<KeyboardInputState>())
{
}

std::shared_ptr<KeyboardProvider> KeyboardTracker::create_provider()
{
    return std::make_shared<KeyboardProvider>(KeyboardProvider::Key{}, state_);
}

const Serialized<KeyboardOutput>& KeyboardTracker::get_data(const ITrackerSession& session) const
{
    return static_cast<const IKeyboardTrackerImpl&>(session.get_tracker_impl(*this)).get_data();
}

} // namespace core
