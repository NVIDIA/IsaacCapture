// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The in-process keyboard driven through a live DeviceIOSession: provider transitions, per-provider
// focus release, the multi-provider merge, the per-frame output and its MCAP round trip. The
// keyboard impl never touches the OpenXR session handles, so placeholder handles stand in for a
// runtime here.

#include "mcap_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/keyboard_generated.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

constexpr uint16_t KEY_W = 17;
constexpr uint16_t KEY_A = 30;
constexpr uint16_t KEY_K = 37;

// Non-null values the keyboard impl never dereferences.
core::OpenXRSessionHandles placeholder_handles()
{
    core::OpenXRSessionHandles handles;
    handles.instance = reinterpret_cast<XrInstance>(1);
    handles.session = reinterpret_cast<XrSession>(1);
    handles.space = reinterpret_cast<XrSpace>(1);
    return handles;
}

struct Event
{
    int64_t timestamp_ns;
    uint16_t code;
    bool pressed;
};

//! One published KeyboardOutput, copied out; `present` is false when the output was absent.
struct Frame
{
    bool present = false;
    std::vector<uint16_t> held;
    std::vector<Event> events;

    std::vector<uint16_t> codes(bool pressed) const
    {
        std::vector<uint16_t> out;
        for (const auto& event : events)
        {
            if (event.pressed == pressed)
                out.push_back(event.code);
        }
        return out;
    }
};

Frame read(const core::KeyboardTracker& keyboard, const core::ITrackerSession& session)
{
    Frame frame;
    const auto& data = keyboard.get_data(session);
    if (!data)
        return frame;
    frame.present = true;
    if (const auto* held = data->held_keys())
        frame.held.assign(held->begin(), held->end());
    if (const auto* events = data->events())
    {
        for (const auto* event : *events)
            frame.events.push_back({ event->timestamp_ns(), event->code(), event->action() == core::KeyAction_Press });
    }
    return frame;
}

//! A keyboard in a running live session; step() publishes one frame and returns it.
struct LiveKeyboard
{
    std::shared_ptr<core::KeyboardTracker> keyboard = std::make_shared<core::KeyboardTracker>();
    std::unique_ptr<core::DeviceIOSession> session;

    void start()
    {
        session = core::DeviceIOSession::run({ keyboard }, placeholder_handles());
        REQUIRE(session != nullptr);
    }

    Frame step()
    {
        session->update();
        return read(*keyboard, *session);
    }
};

LiveKeyboard started()
{
    LiveKeyboard live;
    live.start();
    return live;
}

} // namespace

TEST_CASE("KeyboardTracker: no provider publishes nothing", "[unit][keyboard]")
{
    auto live = started();
    CHECK_FALSE(live.step().present);
}

TEST_CASE("KeyboardProvider: held keys and ordered events, autorepeat ignored", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();

    provider->key_down(KEY_W, 100);
    provider->key_down(KEY_W, 110); // autorepeat
    provider->key_down(std::string_view("KeyA"), 120);
    provider->key_down(std::string_view("NotAKey"));

    auto frame = live.step();
    REQUIRE(frame.present);
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W, KEY_A });
    REQUIRE(frame.events.size() == 2);
    CHECK(frame.events[0].timestamp_ns == 100);
    CHECK(frame.events[1].code == KEY_A);

    // The next frame has no new events but keeps the held keys.
    frame = live.step();
    CHECK(frame.events.empty());
    CHECK(frame.held.size() == 2);
}

TEST_CASE("KeyboardProvider: a sub-frame tap keeps its press event", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();

    provider->key_down(KEY_K);
    provider->key_up(KEY_K);

    const auto frame = live.step();
    CHECK(frame.held.empty());
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_K });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: focus loss releases only that provider's keys", "[unit][keyboard]")
{
    auto live = started();
    auto window = live.keyboard->create_provider();
    auto browser = live.keyboard->create_provider();

    window->key_down(KEY_W);
    browser->key_down(KEY_W);
    browser->key_down(KEY_A);
    live.step();

    browser->release_all();
    const auto frame = live.step();

    // W is still held through the window provider, so only A is released.
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_A });
}

TEST_CASE("KeyboardTracker: events follow the merged keyboard across providers", "[unit][keyboard]")
{
    auto live = started();
    auto a = live.keyboard->create_provider();
    auto b = live.keyboard->create_provider();

    a->key_down(KEY_W, 1);
    b->key_down(KEY_W, 2); // b's own state changes; the merged key was already down
    b->key_down(KEY_W, 3); // autorepeat
    auto frame = live.step();
    REQUIRE(frame.events.size() == 1);
    CHECK(frame.events[0].pressed);
    CHECK(frame.events[0].timestamp_ns == 1);

    a->key_up(KEY_W, 4); // b still holds it
    frame = live.step();
    CHECK(frame.events.empty());
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });

    b->key_up(KEY_W, 5);
    frame = live.step();
    REQUIRE(frame.events.size() == 1);
    CHECK_FALSE(frame.events[0].pressed);
    CHECK(frame.events[0].timestamp_ns == 5);
    CHECK(frame.held.empty());
}

TEST_CASE("KeyboardTracker: closing one provider keeps a key another holds", "[unit][keyboard]")
{
    auto live = started();
    auto a = live.keyboard->create_provider();
    auto b = live.keyboard->create_provider();
    a->key_down(KEY_W);
    b->key_down(KEY_W);
    live.step();

    a->close();
    auto frame = live.step();
    CHECK(frame.events.empty());
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });

    b->release_all();
    frame = live.step();
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.held.empty());
}

TEST_CASE("KeyboardTracker: a tap is hidden while another provider holds the key", "[unit][keyboard]")
{
    auto live = started();
    auto window = live.keyboard->create_provider();
    auto hotkeys = live.keyboard->create_provider();
    window->key_down(KEY_K);
    live.step();

    hotkeys->tap(KEY_K);
    auto frame = live.step();
    CHECK(frame.events.empty());
    CHECK(frame.held == std::vector<uint16_t>{ KEY_K });

    window->key_up(KEY_K);
    live.step();
    hotkeys->tap(KEY_K);
    frame = live.step();
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_K });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: closing releases keys and detaches", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();
    provider->key_down(KEY_W);
    live.step();

    provider->close();
    provider->key_down(KEY_A); // ignored after close
    provider->close(); // closing again does nothing

    // The frame the last provider closes in still publishes, so its releases arrive.
    auto frame = live.step();
    REQUIRE(frame.present);
    CHECK(frame.held.empty());
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(true).empty());

    CHECK_FALSE(live.step().present);
}

TEST_CASE("KeyboardProvider: destruction releases keys", "[unit][keyboard]")
{
    auto live = started();
    {
        auto provider = live.keyboard->create_provider();
        provider->key_down(KEY_W);
    }
    const auto frame = live.step();
    CHECK(frame.held.empty());
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_W });
}

TEST_CASE("KeyboardTracker: an overflowing log still rebuilds the held keys", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();
    provider->key_down(KEY_K, 1);
    live.step(); // previous sample: K held

    provider->key_down(KEY_W, 2); // an early press the overflow must not lose
    for (std::size_t i = 0; i < core::KeyboardInputState::MAX_PENDING_EVENTS; ++i)
    {
        provider->key_down(KEY_A, 10);
        provider->key_up(KEY_A, 10);
    }
    provider->key_up(KEY_K, 20);

    // The minimal change since the previous sample: K released, W pressed; the A taps are lost.
    auto frame = live.step();
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });
    REQUIRE(frame.events.size() == 2);
    CHECK(frame.events[0].code == KEY_K);
    CHECK_FALSE(frame.events[0].pressed);
    CHECK(frame.events[1].code == KEY_W);
    CHECK(frame.events[1].pressed);
    CHECK(frame.events[1].timestamp_ns == 20);

    // Logging resumes with the next sample.
    provider->key_up(KEY_W, 30);
    frame = live.step();
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.held.empty());
}

TEST_CASE("KeyboardTracker: a session starts from no keys and without earlier taps", "[unit][keyboard]")
{
    LiveKeyboard live;
    auto provider = live.keyboard->create_provider();
    provider->key_down(KEY_W, 1);
    provider->tap(KEY_K, 2); // typed before the session: not session input

    live.start();
    const auto frame = live.step();

    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(false).empty());
}

TEST_CASE("KeyboardTracker: a tap before the first frame of a session is reported", "[unit][keyboard]")
{
    LiveKeyboard live;
    auto provider = live.keyboard->create_provider();
    provider->key_down(KEY_W, 1);

    live.start();
    provider->tap(KEY_K, 2);
    const auto frame = live.step();

    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_W, KEY_K });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_K });
}

TEST_CASE("KeyboardProvider: concurrent providers and frames", "[unit][keyboard]")
{
    auto live = started();
    // Total events stay under MAX_PENDING_EVENTS so none can be dropped between frames.
    constexpr int kThreads = 4;
    constexpr int kIterations = 400;
    static_assert(static_cast<std::size_t>(kThreads * kIterations * 2) < core::KeyboardInputState::MAX_PENDING_EVENTS);

    // Created up front, so the keyboard stays present on every frame below.
    std::vector<std::shared_ptr<core::KeyboardProvider>> providers;
    for (int t = 0; t < kThreads; ++t)
        providers.push_back(live.keyboard->create_provider());

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back(
            [&providers, t]()
            {
                const auto code = static_cast<uint16_t>(KEY_W + t);
                for (int i = 0; i < kIterations; ++i)
                {
                    providers[t]->key_down(code);
                    providers[t]->key_up(code);
                }
            });
    }
    std::size_t releases = 0;
    for (int i = 0; i < 100; ++i)
    {
        releases += live.step().codes(false).size();
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    const auto last = live.step();
    releases += last.codes(false).size();

    CHECK(releases == static_cast<std::size_t>(kThreads * kIterations));
    CHECK(last.held.empty());
}

TEST_CASE("KeyboardProvider: codes outside the evdev range are ignored", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();
    constexpr uint16_t kFn = 464; // above 255, inside the evdev range
    constexpr uint16_t kLast = core::kKeyboardKeyCodeCount - 1;

    provider->key_down(kFn);
    provider->key_down(std::string_view("Fn")); // same key, already held
    provider->key_down(kLast);
    provider->key_down(core::kKeyboardKeyCodeCount);
    provider->tap(core::kKeyboardKeyCodeCount);

    const auto frame = live.step();
    CHECK(frame.held == std::vector<uint16_t>{ kFn, kLast });
    CHECK(frame.codes(true) == std::vector<uint16_t>{ kFn, kLast });
}

TEST_CASE("KeyboardProvider: tap reports a press and release without holding", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();

    provider->tap(KEY_K, 50);
    provider->tap(std::string_view("KeyW"));
    provider->tap(std::string_view("NotAKey"));

    const auto frame = live.step();
    CHECK(frame.held.empty());
    CHECK(frame.codes(true) == std::vector<uint16_t>{ KEY_K, KEY_W });
    CHECK(frame.codes(false) == std::vector<uint16_t>{ KEY_K, KEY_W });
    CHECK(frame.events[0].timestamp_ns == frame.events[1].timestamp_ns);
}

TEST_CASE("KeyboardProvider: tap never releases a key the provider holds", "[unit][keyboard]")
{
    auto live = started();
    auto provider = live.keyboard->create_provider();
    provider->key_down(KEY_W);
    live.step();

    provider->tap(KEY_W);

    const auto frame = live.step();
    CHECK(frame.events.empty());
    CHECK(frame.held == std::vector<uint16_t>{ KEY_W });
}

TEST_CASE("DeviceIOSession: keyboard records and replays", "[unit][keyboard]")
{
    const auto path = mcap_test::temp_mcap_path("test_keyboard_session");
    mcap_test::TempFileCleanup cleanup(path);

    {
        auto keyboard = std::make_shared<core::KeyboardTracker>();
        auto provider = keyboard->create_provider();
        auto session = core::DeviceIOSession::run(
            { keyboard }, placeholder_handles(), core::McapRecordingConfig{ path, { { keyboard.get(), "kb" } } });
        provider->key_down(KEY_W, 10);
        session->update();
        provider->key_up(KEY_W, 20);
        session->update();
    }

    core::KeyboardTracker replayed;
    auto replay = core::ReplaySession::run(core::McapReplayConfig{ path, { { &replayed, "kb" } } });

    replay->update();
    const auto first = read(replayed, *replay);
    REQUIRE(first.present);
    CHECK(first.held == std::vector<uint16_t>{ KEY_W });

    replay->update();
    const auto second = read(replayed, *replay);
    REQUIRE(second.present);
    CHECK(second.held.empty());
    REQUIRE(second.events.size() == 1);
    CHECK_FALSE(second.events[0].pressed);
}
