// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/keyboard_tracker_base.hpp>
#include <schema/keyboard_generated.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

namespace core
{

//! A physical key: its W3C `KeyboardEvent.code` and its Linux evdev code.
struct KeyCodeName
{
    std::string_view w3c_code;
    uint16_t evdev_code;
};

//! Number of Linux evdev key codes (`KEY_CNT`, i.e. `KEY_MAX` 0x2ff + 1). Providers reject codes
//! at or above it, so every accepted code indexes the keyboard key bitmaps.
inline constexpr uint16_t kKeyboardKeyCodeCount = 0x300;

//! Every key with both a W3C `KeyboardEvent.code` and an evdev code, from Chromium's key table.
const std::vector<KeyCodeName>& keyboard_key_codes();

//! Evdev key code for a W3C `KeyboardEvent.code` ("KeyW", "ArrowUp", "Numpad8", ...), or
//! nullopt for a code with no evdev equivalent.
std::optional<uint16_t> evdev_code_from_w3c(std::string_view w3c_code);

//! W3C `KeyboardEvent.code` for an evdev key code, or nullopt for a code with no W3C name.
std::optional<std::string_view> w3c_code_from_evdev(uint16_t evdev_code);

/*!
 * @brief Thread-safe key state shared by a KeyboardTracker, its providers and its impl.
 *
 * Internal: reached only through a KeyboardTracker's providers and its tracker impls.
 * Providers report transitions from whatever thread their surface delivers events on;
 * the tracker impl drains once per frame on the session thread. Held keys are kept per
 * provider so one surface losing focus releases only its own keys. Events describe the merged
 * keyboard: a press is logged when the first provider takes a key and a release when the last
 * one lets go. Replaying a drain's events over the previous drain's held keys always yields
 * its own held keys.
 */
class KeyboardInputState
{
public:
    struct Event
    {
        int64_t timestamp_ns;
        uint16_t code;
        bool pressed;
    };

    struct Snapshot
    {
        std::vector<uint16_t> held_keys; //!< Union of every provider's held keys, sorted.
        std::vector<Event> events; //!< Merged-keyboard transitions since the previous drain.
        std::size_t provider_count; //!< Providers open at drain time.
    };

    //! Bound on undrained events. Past it the log is dropped, and the next drain reports the
    //! minimal change since the previous drain instead (releases, then presses): it still rebuilds
    //! the held keys, but sub-frame taps and the order in between are lost.
    static constexpr std::size_t MAX_PENDING_EVENTS = 4096;

    //! Registers a provider; returns its id, which keys that provider's held keys in later calls.
    uint64_t add_provider();
    //! Releases the provider's held keys, then forgets it.
    void remove_provider(uint64_t provider_id, int64_t timestamp_ns);

    //! No-ops for an unknown provider, a press of a key it already holds (autorepeat) and a
    //! release of a key it does not hold. A change hidden by another provider holding the same
    //! key logs no event.
    void key_down(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    void key_up(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    //! Press and release in one step, for surfaces that report presses only. A no-op while any
    //! provider holds the key, so a tap never releases a real hold.
    void tap(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    void release_all(uint64_t provider_id, int64_t timestamp_ns);

    Snapshot drain();

    //! Called when a session starts sampling: drops transitions logged before it, and the next
    //! drain reports every key held now as pressed, so the first sample rebuilds from no keys.
    void start_session();

private:
    bool held_locked(uint16_t code) const;
    std::set<uint16_t> held_keys_locked() const;
    void release_keys_locked(std::set<uint16_t>& keys, int64_t timestamp_ns);
    void push_event_locked(const Event& event);
    void resync_locked(int64_t timestamp_ns);

    mutable std::mutex mutex_;
    std::map<uint64_t, std::set<uint16_t>> held_by_provider_;
    std::vector<Event> pending_;
    std::set<uint16_t> drained_held_; //!< Pressed keys at the previous drain.
    bool resync_ = false; //!< The log was dropped: the next drain reports a minimal delta.
    int64_t resync_timestamp_ns_ = 0; //!< Timestamp for that delta's events.
    uint64_t next_provider_id_ = 1;
};

/*!
 * @brief One input surface (a focused window, a browser tab, ...) feeding a KeyboardTracker.
 *
 * The surface contract is documented once, on the Python ``KeyEventSource`` protocol that hosts
 * implement; in short: report press/release only while the surface has focus and its own UI is
 * not taking keyboard input, call release_all() on blur, close, disconnect or when the host UI
 * takes the keyboard, and make one surface's calls one at a time, in the order it observed them.
 * Autorepeat may be forwarded: a press of a key this provider holds is ignored. A surface without
 * release events reports tap(). Closing (or destroying) the provider releases its keys; calls
 * after close() are ignored. Created by KeyboardTracker::create_provider().
 */
class KeyboardProvider
{
public:
    //! Only KeyboardTracker can construct a provider.
    class Key
    {
        friend class KeyboardTracker;
        Key() = default;
    };

    KeyboardProvider(Key, std::shared_ptr<KeyboardInputState> state);
    ~KeyboardProvider();

    KeyboardProvider(const KeyboardProvider&) = delete;
    KeyboardProvider& operator=(const KeyboardProvider&) = delete;
    KeyboardProvider(KeyboardProvider&&) = delete;
    KeyboardProvider& operator=(KeyboardProvider&&) = delete;

    //! Timestamps default to the monotonic clock at call time. Codes at or above
    //! kKeyboardKeyCodeCount are ignored.
    void key_down(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    void key_up(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! W3C `KeyboardEvent.code` variants. Codes with no evdev equivalent ("Unidentified", keys
    //! a browser names but the kernel does not map) are ignored.
    void key_down(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    void key_up(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! For surfaces that report presses only (no releases): records a press and its release
    //! together, so it reaches the per-frame pressed set without ever being held. A no-op while any
    //! provider holds the key, so a tap never releases a real hold.
    void tap(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    void tap(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! Release everything this provider holds: on blur, close, disconnect, and whenever the host's
    //! own UI takes the keyboard (e.g. a text field gains focus).
    void release_all(std::optional<int64_t> timestamp_ns = std::nullopt);
    void close();

private:
    std::shared_ptr<KeyboardInputState> state_;
    uint64_t id_;
};

/*!
 * @brief In-process keyboard device: merges key events from every attached provider.
 *
 * Needs no OpenXR extension and no extra process. Surfaces attach through
 * create_provider(); the session publishes the merged state once per update and records
 * it to MCAP. In replay the recorded state is returned and provider input is ignored.
 */
class KeyboardTracker : public ITracker
{
public:
    KeyboardTracker();

    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    std::shared_ptr<KeyboardProvider> create_provider();

    //! Empty when no provider is attached (or, in replay, when the recording has no sample).
    const Serialized<KeyboardOutput>& get_data(const ITrackerSession& session) const;

private:
    // The tracker impls own sampling: only they may drain the shared state.
    friend class LiveDeviceIOFactory;
    friend class ReplayDeviceIOFactory;

    const std::shared_ptr<KeyboardInputState>& input_state() const
    {
        return state_;
    }

    static constexpr const char* TRACKER_NAME = "KeyboardTracker";

    std::shared_ptr<KeyboardInputState> state_;
};

} // namespace core
