// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The in-process SpaceMouse reads a 3Dconnexion HID device in a live DeviceIOSession. A FIFO stands
// in for /dev/hidrawN: the test writes HID reports into it and closes it to unplug the device.
// The impl never touches the OpenXR session handles, so placeholder handles stand in for a runtime.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/spacemouse_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/spacemouse_generated.h>
#include <sys/stat.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>

namespace
{

constexpr uint8_t BUTTON_LEFT_BIT = 0x01;
constexpr uint8_t BUTTON_RIGHT_BIT = 0x02;
// 350 raw counts is full scale.
constexpr int16_t FULL_SCALE = 350;

// Non-null values the SpaceMouse impl never dereferences.
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

// A fake SpaceMouse HID device: a FIFO the test holds open for writing.
class FakeSpaceMouse
{
public:
    explicit FakeSpaceMouse(size_t report_size = 7) : path_(temp_path("test_spacemouse_hidraw", "")), size_(report_size)
    {
        REQUIRE(::mkfifo(path_.c_str(), 0600) == 0);
        // O_RDWR keeps a writer attached without blocking, so the tracker's reader never sees EOF
        // until the test unplugs the device.
        fd_ = ::open(path_.c_str(), O_RDWR | O_NONBLOCK);
        REQUIRE(fd_ >= 0);
    }

    ~FakeSpaceMouse()
    {
        unplug();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string& path() const
    {
        return path_;
    }

    // Report 1: translation (and, for the combined layout, rotation).
    void motion(int16_t x, int16_t y, int16_t z, int16_t rx = 0, int16_t ry = 0, int16_t rz = 0)
    {
        std::array<int16_t, 6> axes{ x, y, z, rx, ry, rz };
        std::array<uint8_t, 13> report{};
        report[0] = 1;
        for (size_t i = 0; i < axes.size(); ++i)
        {
            report[1 + 2 * i] = static_cast<uint8_t>(axes[i] & 0xFF);
            report[2 + 2 * i] = static_cast<uint8_t>((axes[i] >> 8) & 0xFF);
        }
        send(report.data());
    }

    // Report 3: button bitmask.
    void buttons(uint8_t mask)
    {
        std::array<uint8_t, 13> report{};
        report[0] = 3;
        report[1] = mask;
        send(report.data());
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
    void send(const uint8_t* report)
    {
        REQUIRE(::write(fd_, report, size_) == static_cast<ssize_t>(size_));
    }

    std::string path_;
    size_t size_;
    int fd_ = -1;
};

} // namespace

TEST_CASE("SpaceMouseTracker: reads motion and buttons", "[unit][spacemouse]")
{
    FakeSpaceMouse device;
    auto spacemouse = std::make_shared<core::SpaceMouseTracker>(device.path());
    auto session = core::DeviceIOSession::run({ spacemouse }, placeholder_handles());

    session->update();
    const auto& idle = spacemouse->get_data(*session);
    REQUIRE(idle);
    CHECK(idle->is_valid());
    CHECK(idle->translation()->Get(0) == 0.0F);

    device.motion(FULL_SCALE, -FULL_SCALE, 0);
    device.buttons(BUTTON_LEFT_BIT | BUTTON_RIGHT_BIT);
    session->update();

    const auto& data = spacemouse->get_data(*session);
    REQUIRE(data);
    REQUIRE(data->translation()->size() == 3);
    CHECK(data->translation()->Get(0) == 1.0F);
    CHECK(data->translation()->Get(1) == -1.0F);
    REQUIRE(data->pressed_buttons()->size() == 2);
    CHECK(data->pressed_buttons()->Get(0) == 0);
    CHECK(data->pressed_buttons()->Get(1) == 1);
}

TEST_CASE("SpaceMouseTracker: combined reports carry rotation", "[unit][spacemouse]")
{
    FakeSpaceMouse device(13);
    auto spacemouse = std::make_shared<core::SpaceMouseTracker>(device.path(), true);
    auto session = core::DeviceIOSession::run({ spacemouse }, placeholder_handles());

    device.motion(0, 0, 0, 0, 0, FULL_SCALE);
    session->update();

    const auto& data = spacemouse->get_data(*session);
    REQUIRE(data);
    REQUIRE(data->rotation()->size() == 3);
    CHECK(data->rotation()->Get(2) == 1.0F);
}

TEST_CASE("SpaceMouseTracker: an unplugged SpaceMouse reports nothing", "[unit][spacemouse]")
{
    FakeSpaceMouse device;
    auto spacemouse = std::make_shared<core::SpaceMouseTracker>(device.path());
    auto session = core::DeviceIOSession::run({ spacemouse }, placeholder_handles());

    device.buttons(BUTTON_LEFT_BIT);
    session->update();
    REQUIRE(spacemouse->get_data(*session));

    device.unplug();
    session->update();
    CHECK_FALSE(spacemouse->get_data(*session));
}

TEST_CASE("SpaceMouseTracker: a missing device reports nothing", "[unit][spacemouse]")
{
    auto spacemouse = std::make_shared<core::SpaceMouseTracker>("/nonexistent/hidraw0");
    auto session = core::DeviceIOSession::run({ spacemouse }, placeholder_handles());

    session->update();
    CHECK_FALSE(spacemouse->get_data(*session));
}

TEST_CASE("SpaceMouseTracker: records and replays", "[unit][spacemouse]")
{
    const auto path = temp_path("test_spacemouse", ".mcap");
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
        FakeSpaceMouse device;
        auto spacemouse = std::make_shared<core::SpaceMouseTracker>(device.path());
        auto session = core::DeviceIOSession::run({ spacemouse }, placeholder_handles(),
                                                  core::McapRecordingConfig{ path, { { spacemouse.get(), "sm" } } });
        device.buttons(BUTTON_LEFT_BIT);
        session->update();
        device.buttons(0);
        session->update();
    }

    core::SpaceMouseTracker replayed;
    auto replay = core::ReplaySession::run(core::McapReplayConfig{ path, { { &replayed, "sm" } } });

    replay->update();
    const auto& first = replayed.get_data(*replay);
    REQUIRE(first);
    REQUIRE(first->pressed_buttons()->size() == 1);
    CHECK(first->pressed_buttons()->Get(0) == 0);

    replay->update();
    const auto& second = replayed.get_data(*replay);
    REQUIRE(second);
    CHECK((second->pressed_buttons() == nullptr || second->pressed_buttons()->size() == 0));
}
