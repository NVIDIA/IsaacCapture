// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The in-process keyboard runs in a live DeviceIOSession and records to MCAP. Its impl never
// touches the OpenXR session handles, so placeholder handles stand in for a runtime here.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/keyboard_generated.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#ifdef _WIN32
#    include <process.h>
#    define GET_PID() _getpid()
#else
#    include <unistd.h>
#    define GET_PID() ::getpid()
#endif

namespace
{

constexpr uint16_t KEY_W = 17;
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

std::string temp_mcap_path()
{
    static std::atomic<int> count{ 0 };
    const auto name = "test_keyboard_session_" + std::to_string(GET_PID()) + "_" + std::to_string(count++) + ".mcap";
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

TEST_CASE("DeviceIOSession: publishes keyboard provider input", "[unit][keyboard]")
{
    auto keyboard = std::make_shared<core::KeyboardTracker>();
    auto session = core::DeviceIOSession::run({ keyboard }, placeholder_handles());
    REQUIRE(session != nullptr);

    session->update();
    CHECK_FALSE(keyboard->get_data(*session)); // no provider attached yet

    auto provider = keyboard->create_provider("test");
    provider->key_down(KEY_W, 10);
    provider->tap(KEY_K, 20);
    session->update();

    const auto& data = keyboard->get_data(*session);
    REQUIRE(data);
    REQUIRE(data->pressed_keys()->size() == 1);
    CHECK(data->pressed_keys()->Get(0) == KEY_W);
    REQUIRE(data->events()->size() == 3);
    CHECK(data->events()->Get(1)->code() == KEY_K);
    CHECK(data->events()->Get(1)->action() == core::KeyAction_Press);
    CHECK(data->events()->Get(2)->action() == core::KeyAction_Release);
}

TEST_CASE("DeviceIOSession: keyboard records and replays", "[unit][keyboard]")
{
    const auto path = temp_mcap_path();
    struct Cleanup
    {
        std::string path;
        ~Cleanup()
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{ path };

    {
        auto keyboard = std::make_shared<core::KeyboardTracker>();
        auto provider = keyboard->create_provider("test");
        auto session = core::DeviceIOSession::run({ keyboard }, placeholder_handles(),
                                                  core::McapRecordingConfig{ path, { { keyboard.get(), "kb" } } });
        provider->key_down(KEY_W, 10);
        session->update();
        provider->key_up(KEY_W, 20);
        session->update();
    }

    core::KeyboardTracker replayed;
    auto replay = core::ReplaySession::run(core::McapReplayConfig{ path, { { &replayed, "kb" } } });

    replay->update();
    const auto& first = replayed.get_data(*replay);
    REQUIRE(first);
    REQUIRE(first->pressed_keys()->size() == 1);
    CHECK(first->pressed_keys()->Get(0) == KEY_W);

    replay->update();
    const auto& second = replayed.get_data(*replay);
    REQUIRE(second);
    CHECK((second->pressed_keys() == nullptr || second->pressed_keys()->size() == 0));
    REQUIRE(second->events()->size() == 1);
    CHECK(second->events()->Get(0)->action() == core::KeyAction_Release);
}
