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
#include <string>
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
 * one lets go. Replaying a drain's events over the previous drain's pressed keys always yields
 * its own pressed keys.
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
        std::vector<uint16_t> pressed_keys; //!< Union of every provider's held keys, sorted.
        std::vector<Event> events; //!< Merged-keyboard transitions since the previous drain.
        std::size_t provider_count; //!< Providers open at drain time.
    };

    //! Bound on undrained events. Past it the log is dropped, and the next drain reports the
    //! minimal change since the previous drain instead (releases, then presses): it still rebuilds
    //! the pressed keys, but sub-frame taps and the order in between are lost.
    static constexpr std::size_t MAX_PENDING_EVENTS = 4096;

    uint64_t add_provider();
    //! Releases the provider's held keys, then forgets it.
    void remove_provider(uint64_t provider_id, int64_t timestamp_ns);

    //! Returns false when the provider's own state does not change (autorepeat, or releasing an
    //! unheld key). A change hidden by another provider holding the same key returns true but
    //! logs no event.
    bool key_down(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    bool key_up(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    //! Press and release in one step, for surfaces that report presses only. A no-op returning
    //! false while any provider holds the key, so a tap never releases a real hold.
    bool tap(uint64_t provider_id, uint16_t code, int64_t timestamp_ns);
    void release_all(uint64_t provider_id, int64_t timestamp_ns);

    Snapshot drain();

    //! Called when a session starts sampling: drops transitions logged before it, and the next
    //! drain reports every key held now as pressed, so the first sample rebuilds from no keys.
    void start_session();

private:
    void push_event_locked(const Event& event);
    void resync_locked(int64_t timestamp_ns);
    void press_locked(uint16_t code, int64_t timestamp_ns);
    void release_locked(uint16_t code, int64_t timestamp_ns);

    mutable std::mutex mutex_;
    std::map<uint64_t, std::set<uint16_t>> held_by_provider_;
    std::map<uint16_t, std::size_t> holders_; //!< Providers holding each key; absent = released.
    std::vector<Event> pending_;
    std::set<uint16_t> drained_held_; //!< Pressed keys at the previous drain.
    bool resync_ = false; //!< The log was dropped: the next drain reports a minimal delta.
    int64_t resync_timestamp_ns_ = 0; //!< Timestamp for that delta's events.
    uint64_t next_provider_id_ = 1;
};

/*!
 * @brief One input surface (a focused window, a browser tab, ...) feeding a KeyboardTracker.
 *
 * Contract for the surface: report press/release only while it has focus and its own UI is not
 * taking keyboard input, never report autorepeat as new presses, and call release_all() on blur,
 * close, disconnect, or when the host UI takes the keyboard, so no key can stay stuck. A surface
 * without release events reports tap(). Calls for one surface must be made one at a time, in the
 * order the surface observed them: a press observed before focus loss completes before
 * release_all(), never after it. Every method is thread-safe, but that does not order calls: a
 * press delivered after release_all() stays held. Closing (or destroying) the provider releases
 * its keys. Created by KeyboardTracker::create_provider().
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

    KeyboardProvider(Key, std::shared_ptr<KeyboardInputState> state, std::string name);
    ~KeyboardProvider();

    KeyboardProvider(const KeyboardProvider&) = delete;
    KeyboardProvider& operator=(const KeyboardProvider&) = delete;
    KeyboardProvider(KeyboardProvider&&) = delete;
    KeyboardProvider& operator=(KeyboardProvider&&) = delete;

    //! Timestamps default to the monotonic clock at call time.
    //! Return false when this provider's state does not change (autorepeat, or releasing an unheld
    //! key), for a code at or above kKeyboardKeyCodeCount, or when the provider is closed. Pressing
    //! a key another provider already holds returns true but logs no event.
    bool key_down(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    bool key_up(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! W3C `KeyboardEvent.code` variants. Unknown codes are ignored and return false.
    bool key_down(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    bool key_up(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! For surfaces that report presses only (no releases): records a press and its release
    //! together, so it reaches the per-frame pressed set without ever being held. A no-op returning
    //! false while any provider holds the key, so a tap never releases a real hold.
    bool tap(uint16_t evdev_code, std::optional<int64_t> timestamp_ns = std::nullopt);
    bool tap(std::string_view w3c_code, std::optional<int64_t> timestamp_ns = std::nullopt);

    //! Release everything this provider holds: on blur, close, disconnect, and whenever the host's
    //! own UI takes the keyboard (e.g. a text field gains focus).
    void release_all(std::optional<int64_t> timestamp_ns = std::nullopt);
    void close();

    bool is_closed() const;
    const std::string& name() const
    {
        return name_;
    }

private:
    std::shared_ptr<KeyboardInputState> state_;
    std::string name_;
    uint64_t id_;
    mutable std::mutex mutex_; // guards closed_ against a concurrent close()
    bool closed_ = false;
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

    std::shared_ptr<KeyboardProvider> create_provider(std::string name);

    //! Empty when no provider is attached (or, in replay, when the recording has no sample).
    const Serialized<KeyboardOutput>& get_data(const ITrackerSession& session) const;

private:
    // The tracker impls own sampling: only they (and the unit tests) may drain the shared state.
    friend class LiveDeviceIOFactory;
    friend class ReplayDeviceIOFactory;
    friend struct KeyboardTrackerTestAccess;

    const std::shared_ptr<KeyboardInputState>& input_state() const
    {
        return state_;
    }

    static constexpr const char* TRACKER_NAME = "KeyboardTracker";

    std::shared_ptr<KeyboardInputState> state_;
};

} // namespace core
