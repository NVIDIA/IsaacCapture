// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The in-process gamepad reads a joystick-API device in a live DeviceIOSession. A FIFO stands in
// for /dev/input/jsN: the test writes js_event records into it and closes it to unplug the pad.
// The impl never touches the OpenXR session handles, so placeholder handles stand in for a runtime.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/gamepad_tracker.hpp>
#include <linux/joystick.h>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/gamepad_generated.h>
#include <sys/stat.h>

#include <atomic>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>

namespace
{

constexpr uint8_t BUTTON_X = 2;
constexpr uint8_t AXIS_LEFT_Y = 1;

// Non-null values the gamepad impl never dereferences.
core::OpenXRSessionHandles placeholder_handles()
{
    core::OpenXRSessionHandles handles;
    handles.instance = reinterpret_cast<XrInstance>(1);
    handles.session = reinterpret_cast<XrSession>(1);
    handles.space = reinterpret_cast<XrSpace>(1);
    return handles;
}

std::string temp_path(const std::string& stem, const std::string& ext)
{
    static std::atomic<int> count{ 0 };
    const auto name = stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(count++) + ext;
    return (std::filesystem::temp_directory_path() / name).string();
}

// A fake joystick device: a FIFO the test holds open for writing.
class FakeJoystick
{
public:
    FakeJoystick() : path_(temp_path("test_gamepad_js", ""))
    {
        REQUIRE(::mkfifo(path_.c_str(), 0600) == 0);
        // O_RDWR keeps a writer attached without blocking, so the tracker's reader never sees EOF
        // until the test unplugs the device.
        fd_ = ::open(path_.c_str(), O_RDWR | O_NONBLOCK);
        REQUIRE(fd_ >= 0);
    }

    ~FakeJoystick()
    {
        unplug();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string& path() const
    {
        return path_;
    }

    void send(uint8_t type, uint8_t number, int16_t value)
    {
        js_event event{};
        event.type = type;
        event.number = number;
        event.value = value;
        REQUIRE(::write(fd_, &event, sizeof(event)) == static_cast<ssize_t>(sizeof(event)));
    }

    void unplug()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    std::string path_;
    int fd_ = -1;
};

} // namespace

TEST_CASE("GamepadTracker: reads buttons and axes", "[unit][gamepad]")
{
    FakeJoystick joystick;
    auto gamepad = std::make_shared<core::GamepadTracker>(joystick.path());
    auto session = core::DeviceIOSession::run({ gamepad }, placeholder_handles());

    session->update();
    const auto& idle = gamepad->get_data(*session);
    REQUIRE(idle);
    CHECK(idle->is_valid());
    CHECK((idle->pressed_buttons() == nullptr || idle->pressed_buttons()->size() == 0));

    joystick.send(JS_EVENT_BUTTON, BUTTON_X, 1);
    joystick.send(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767);
    session->update();

    const auto& data = gamepad->get_data(*session);
    REQUIRE(data);
    REQUIRE(data->pressed_buttons()->size() == 1);
    CHECK(data->pressed_buttons()->Get(0) == BUTTON_X);
    REQUIRE(data->axes()->size() > AXIS_LEFT_Y);
    CHECK(data->axes()->Get(AXIS_LEFT_Y) == -1.0F);

    joystick.send(JS_EVENT_BUTTON, BUTTON_X, 0);
    session->update();
    const auto& released = gamepad->get_data(*session);
    REQUIRE(released);
    CHECK((released->pressed_buttons() == nullptr || released->pressed_buttons()->size() == 0));
}

TEST_CASE("GamepadTracker: an unplugged gamepad reports nothing", "[unit][gamepad]")
{
    FakeJoystick joystick;
    auto gamepad = std::make_shared<core::GamepadTracker>(joystick.path());
    auto session = core::DeviceIOSession::run({ gamepad }, placeholder_handles());

    joystick.send(JS_EVENT_BUTTON, BUTTON_X, 1);
    session->update();
    REQUIRE(gamepad->get_data(*session));

    joystick.unplug();
    session->update();
    CHECK_FALSE(gamepad->get_data(*session));
}

TEST_CASE("GamepadTracker: a missing device reports nothing", "[unit][gamepad]")
{
    auto gamepad = std::make_shared<core::GamepadTracker>("/nonexistent/js0");
    auto session = core::DeviceIOSession::run({ gamepad }, placeholder_handles());

    session->update();
    CHECK_FALSE(gamepad->get_data(*session));
}

TEST_CASE("GamepadTracker: records and replays", "[unit][gamepad]")
{
    const auto path = temp_path("test_gamepad", ".mcap");
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
        FakeJoystick joystick;
        auto gamepad = std::make_shared<core::GamepadTracker>(joystick.path());
        auto session = core::DeviceIOSession::run(
            { gamepad }, placeholder_handles(), core::McapRecordingConfig{ path, { { gamepad.get(), "pad" } } });
        joystick.send(JS_EVENT_BUTTON, BUTTON_X, 1);
        session->update();
        joystick.send(JS_EVENT_BUTTON, BUTTON_X, 0);
        session->update();
    }

    core::GamepadTracker replayed;
    auto replay = core::ReplaySession::run(core::McapReplayConfig{ path, { { &replayed, "pad" } } });

    replay->update();
    const auto& first = replayed.get_data(*replay);
    REQUIRE(first);
    REQUIRE(first->pressed_buttons()->size() == 1);
    CHECK(first->pressed_buttons()->Get(0) == BUTTON_X);

    replay->update();
    const auto& second = replayed.get_data(*replay);
    REQUIRE(second);
    CHECK((second->pressed_buttons() == nullptr || second->pressed_buttons()->size() == 0));
}
