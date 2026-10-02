// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the in-process keyboard: provider transitions, per-provider focus release,
// the multi-provider merge, and the per-frame drain the live impl publishes.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace core
{
//! Friend of KeyboardTracker, defined only here: the tests drain its state directly.
struct KeyboardTrackerTestAccess
{
    static const std::shared_ptr<KeyboardInputState>& state(const KeyboardTracker& tracker)
    {
        return tracker.input_state();
    }
};
} // namespace core

namespace
{

const std::shared_ptr<core::KeyboardInputState>& state_of(const core::KeyboardTracker& tracker)
{
    return core::KeyboardTrackerTestAccess::state(tracker);
}

constexpr uint16_t KEY_W = 17;
constexpr uint16_t KEY_A = 30;
constexpr uint16_t KEY_K = 37;

std::vector<uint16_t> event_codes(const core::KeyboardInputState::Snapshot& snapshot, bool pressed)
{
    std::vector<uint16_t> codes;
    for (const auto& event : snapshot.events)
    {
        if (event.pressed == pressed)
        {
            codes.push_back(event.code);
        }
    }
    return codes;
}

} // namespace

TEST_CASE("KeyboardInputState: no provider drains empty", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    const auto snapshot = state_of(tracker)->drain();

    CHECK(snapshot.provider_count == 0);
    CHECK(snapshot.pressed_keys.empty());
    CHECK(snapshot.events.empty());
}

TEST_CASE("KeyboardProvider: held keys and ordered events, autorepeat ignored", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");

    CHECK(provider->key_down(KEY_W, 100));
    CHECK_FALSE(provider->key_down(KEY_W, 110)); // autorepeat
    CHECK(provider->key_down(std::string_view("KeyA"), 120));
    CHECK_FALSE(provider->key_down(std::string_view("NotAKey")));

    auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.provider_count == 1);
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W, KEY_A });
    REQUIRE(snapshot.events.size() == 2);
    CHECK(snapshot.events[0].timestamp_ns == 100);
    CHECK(snapshot.events[1].code == KEY_A);

    // A drain empties the event log but keeps held state.
    snapshot = state_of(tracker)->drain();
    CHECK(snapshot.events.empty());
    CHECK(snapshot.pressed_keys.size() == 2);
}

TEST_CASE("KeyboardProvider: a sub-frame tap keeps its press event", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");

    provider->key_down(KEY_K);
    provider->key_up(KEY_K);

    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.pressed_keys.empty());
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_K });
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: focus loss releases only that provider's keys", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto window = tracker.create_provider("window");
    auto browser = tracker.create_provider("browser");

    window->key_down(KEY_W);
    browser->key_down(KEY_W);
    browser->key_down(KEY_A);
    state_of(tracker)->drain();

    browser->release_all();
    const auto snapshot = state_of(tracker)->drain();

    // W is still held through the window provider, so only A is released.
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_A });
    CHECK(snapshot.provider_count == 2);
}

TEST_CASE("KeyboardInputState: events follow the merged keyboard across providers", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto a = tracker.create_provider("a");
    auto b = tracker.create_provider("b");
    auto state = state_of(tracker);

    CHECK(a->key_down(KEY_W, 1));
    CHECK(b->key_down(KEY_W, 2)); // b's own state changes; the merged key was already down
    CHECK_FALSE(b->key_down(KEY_W, 3)); // autorepeat
    auto snapshot = state->drain();
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_W });
    REQUIRE(snapshot.events.size() == 1);
    CHECK(snapshot.events[0].timestamp_ns == 1);

    CHECK(a->key_up(KEY_W, 4)); // b still holds it
    snapshot = state->drain();
    CHECK(snapshot.events.empty());
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });

    CHECK(b->key_up(KEY_W, 5));
    snapshot = state->drain();
    REQUIRE(snapshot.events.size() == 1);
    CHECK_FALSE(snapshot.events[0].pressed);
    CHECK(snapshot.events[0].timestamp_ns == 5);
    CHECK(snapshot.pressed_keys.empty());
}

TEST_CASE("KeyboardInputState: closing one provider keeps a key another holds", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto a = tracker.create_provider("a");
    auto b = tracker.create_provider("b");
    a->key_down(KEY_W);
    b->key_down(KEY_W);
    state_of(tracker)->drain();

    a->close();
    auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.events.empty());
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });

    b->release_all();
    snapshot = state_of(tracker)->drain();
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_W });
    CHECK(snapshot.pressed_keys.empty());
}

TEST_CASE("KeyboardInputState: a tap is hidden while another provider holds the key", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto window = tracker.create_provider("window");
    auto hotkeys = tracker.create_provider("hotkeys");
    window->key_down(KEY_K);
    state_of(tracker)->drain();

    CHECK_FALSE(hotkeys->tap(KEY_K));
    auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.events.empty());
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_K });

    window->key_up(KEY_K);
    state_of(tracker)->drain();
    CHECK(hotkeys->tap(KEY_K));
    snapshot = state_of(tracker)->drain();
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_K });
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: closing releases keys and detaches", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    provider->key_down(KEY_W);
    state_of(tracker)->drain();

    provider->close();
    CHECK(provider->is_closed());
    CHECK_FALSE(provider->key_down(KEY_A));

    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.provider_count == 0);
    CHECK(snapshot.pressed_keys.empty());
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_W });
}

TEST_CASE("KeyboardProvider: destruction releases keys", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    {
        auto provider = tracker.create_provider("window");
        provider->key_down(KEY_W);
    }
    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.provider_count == 0);
    CHECK(snapshot.pressed_keys.empty());
}

TEST_CASE("KeyboardInputState: an overflowing log still rebuilds the pressed keys", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    provider->key_down(KEY_K, 1);
    state_of(tracker)->drain(); // previous sample: K held

    provider->key_down(KEY_W, 2); // an early press the overflow must not lose
    for (std::size_t i = 0; i < core::KeyboardInputState::MAX_PENDING_EVENTS; ++i)
    {
        provider->key_down(KEY_A, 10);
        provider->key_up(KEY_A, 10);
    }
    provider->key_up(KEY_K, 20);

    // The minimal change since the previous sample: K released, W pressed; the A taps are lost.
    auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });
    REQUIRE(snapshot.events.size() == 2);
    CHECK(snapshot.events[0].code == KEY_K);
    CHECK_FALSE(snapshot.events[0].pressed);
    CHECK(snapshot.events[1].code == KEY_W);
    CHECK(snapshot.events[1].pressed);
    CHECK(snapshot.events[1].timestamp_ns == 20);

    // Logging resumes with the next sample.
    provider->key_up(KEY_W, 30);
    snapshot = state_of(tracker)->drain();
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_W });
    CHECK(snapshot.pressed_keys.empty());
}

TEST_CASE("KeyboardInputState: a session starts from no keys and without earlier taps", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    provider->key_down(KEY_W, 1);
    provider->tap(KEY_K, 2); // typed before the session: not session input

    state_of(tracker)->start_session();
    const auto snapshot = state_of(tracker)->drain();

    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_W });
    CHECK(event_codes(snapshot, false).empty());
}

TEST_CASE("KeyboardInputState: a tap before the first drain of a session is reported", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    provider->key_down(KEY_W, 1);

    state_of(tracker)->start_session();
    provider->tap(KEY_K, 2);
    const auto snapshot = state_of(tracker)->drain();

    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_W, KEY_K });
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: concurrent providers and drains", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    // Total events stay under MAX_PENDING_EVENTS so none can be dropped between drains.
    constexpr int kThreads = 4;
    constexpr int kIterations = 400;
    static_assert(static_cast<std::size_t>(kThreads * kIterations * 2) < core::KeyboardInputState::MAX_PENDING_EVENTS);

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back(
            [&tracker, t]()
            {
                auto provider = tracker.create_provider("thread" + std::to_string(t));
                const auto code = static_cast<uint16_t>(KEY_W + t);
                for (int i = 0; i < kIterations; ++i)
                {
                    provider->key_down(code);
                    provider->key_up(code);
                }
            });
    }
    std::size_t releases = 0;
    for (int i = 0; i < 100; ++i)
    {
        releases += event_codes(state_of(tracker)->drain(), false).size();
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    releases += event_codes(state_of(tracker)->drain(), false).size();

    CHECK(releases == static_cast<std::size_t>(kThreads * kIterations));
    CHECK(state_of(tracker)->drain().pressed_keys.empty());
}

TEST_CASE("evdev_code_from_w3c maps standard keys", "[unit][keyboard]")
{
    CHECK(core::evdev_code_from_w3c("KeyW") == KEY_W);
    CHECK(core::evdev_code_from_w3c("ArrowUp") == uint16_t{ 103 });
    CHECK(core::evdev_code_from_w3c("Numpad8") == uint16_t{ 72 });
    CHECK_FALSE(core::evdev_code_from_w3c("NotAKey").has_value());
}

TEST_CASE("w3c_code_from_evdev names keys from Chromium's key table", "[unit][keyboard]")
{
    CHECK(core::w3c_code_from_evdev(KEY_W) == std::string_view("KeyW"));
    CHECK(core::w3c_code_from_evdev(127) == std::string_view("ContextMenu"));
    CHECK(core::w3c_code_from_evdev(86) == std::string_view("IntlBackslash"));
    CHECK_FALSE(core::w3c_code_from_evdev(0).has_value());
}

TEST_CASE("keyboard_key_codes: every key round-trips", "[unit][keyboard]")
{
    const auto& keys = core::keyboard_key_codes();
    REQUIRE(keys.size() > 100);
    for (const auto& key : keys)
    {
        CHECK(core::evdev_code_from_w3c(key.w3c_code) == key.evdev_code);
        const auto name = core::w3c_code_from_evdev(key.evdev_code);
        REQUIRE(name.has_value());
        CHECK(core::evdev_code_from_w3c(*name) == key.evdev_code);
        CHECK(key.evdev_code < core::kKeyboardKeyCodeCount);
    }
}

TEST_CASE("KeyboardProvider: codes above KEY_MAX are rejected", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    constexpr uint16_t kFn = 464; // above 255, inside the evdev range
    constexpr uint16_t kLast = core::kKeyboardKeyCodeCount - 1;

    CHECK(provider->key_down(kFn));
    CHECK(provider->key_down(std::string_view("Fn")) == false); // already held
    CHECK(provider->key_down(kLast));
    CHECK_FALSE(provider->key_down(core::kKeyboardKeyCodeCount));
    CHECK_FALSE(provider->key_up(core::kKeyboardKeyCodeCount));
    CHECK_FALSE(provider->tap(core::kKeyboardKeyCodeCount));

    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ kFn, kLast });
}

TEST_CASE("KeyboardProvider: tap reports a press and release without holding", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("hotkeys");

    CHECK(provider->tap(KEY_K, 50));
    CHECK(provider->tap(std::string_view("KeyW")));
    CHECK_FALSE(provider->tap(std::string_view("NotAKey")));

    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.pressed_keys.empty());
    CHECK(event_codes(snapshot, true) == std::vector<uint16_t>{ KEY_K, KEY_W });
    CHECK(event_codes(snapshot, false) == std::vector<uint16_t>{ KEY_K, KEY_W });
    CHECK(snapshot.events[0].timestamp_ns == snapshot.events[1].timestamp_ns);
}

TEST_CASE("KeyboardProvider: tap never releases a key the provider holds", "[unit][keyboard]")
{
    core::KeyboardTracker tracker;
    auto provider = tracker.create_provider("window");
    provider->key_down(KEY_W);
    state_of(tracker)->drain();

    CHECK_FALSE(provider->tap(KEY_W));

    const auto snapshot = state_of(tracker)->drain();
    CHECK(snapshot.events.empty());
    CHECK(snapshot.pressed_keys == std::vector<uint16_t>{ KEY_W });
}
