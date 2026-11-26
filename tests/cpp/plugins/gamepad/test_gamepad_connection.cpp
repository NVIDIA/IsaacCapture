// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The connection reads SDL virtual gamepads: attaching one is a pad being plugged in, setting its
// buttons and axes is input, and detaching it is an unplug. The state is read back through
// GamepadInputState's named fields, so the plugin's SDL-to-schema mapping is checked end to end. Real
// pads are hidden from SDL so a pad on the test machine does not count.

#include <SDL3/SDL.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gamepad/gamepad_connection.hpp>

#include <cstdint>
#include <memory>

using plugins::gamepad::ConnectionStatus;
using plugins::gamepad::GamepadConnection;
using plugins::gamepad::GamepadConnectionConfig;

namespace
{

// pid.codes test VID; each pad gets its own product ID, which SDL reports per device.
constexpr uint16_t kVirtualVendor = 0x1209;

//! A connection that sees only the virtual pads this file attaches.
std::unique_ptr<GamepadConnection> make_connection(int player = 1)
{
    SDL_SetHint(SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT, "0x1209/0x0001,0x1209/0x0002");
    return std::make_unique<GamepadConnection>(GamepadConnectionConfig{ player });
}

//! A virtual gamepad with every standard button and axis, in SDL_GamepadButton/SDL_GamepadAxis order.
class VirtualPad
{
public:
    explicit VirtualPad(const char* name, uint16_t product_id = 0x0001)
    {
        SDL_VirtualJoystickDesc desc;
        SDL_INIT_INTERFACE(&desc);
        desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
        desc.vendor_id = kVirtualVendor;
        desc.product_id = product_id;
        desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
        desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
        desc.name = name;
        id_ = SDL_AttachVirtualJoystick(&desc);
        REQUIRE(id_ != 0);
        joystick_ = SDL_OpenJoystick(id_);
        REQUIRE(joystick_ != nullptr);
    }
    ~VirtualPad()
    {
        unplug();
    }
    void press(SDL_GamepadButton button, bool down = true)
    {
        REQUIRE(SDL_SetJoystickVirtualButton(joystick_, button, down));
    }
    void move(SDL_GamepadAxis axis, int16_t value)
    {
        REQUIRE(SDL_SetJoystickVirtualAxis(joystick_, axis, value));
    }
    void unplug()
    {
        if (joystick_ != nullptr)
            SDL_CloseJoystick(joystick_);
        if (id_ != 0)
            SDL_DetachVirtualJoystick(id_);
        joystick_ = nullptr;
        id_ = 0;
    }

private:
    SDL_JoystickID id_ = 0;
    SDL_Joystick* joystick_ = nullptr;
};

} // namespace

TEST_CASE("No gamepad connected: disconnected at rest", "[gamepad][connection]")
{
    auto connection = make_connection();

    connection->poll();

    CHECK(connection->status() == ConnectionStatus::NoDevice);
    CHECK(connection->description().empty());
    CHECK(connection->state().inputs.thumbstick_left_x() == 0.0f);
}

TEST_CASE("A connected gamepad is read in the standard layout until it is unplugged", "[gamepad][connection]")
{
    auto connection = make_connection();
    VirtualPad pad("Test Pad");

    connection->poll();
    REQUIRE(connection->status() == ConnectionStatus::Connected);
    CHECK(connection->description().find("Test Pad") != std::string::npos);
    CHECK(connection->description().find("1209:0001") != std::string::npos);

    pad.press(SDL_GAMEPAD_BUTTON_WEST);
    pad.press(SDL_GAMEPAD_BUTTON_DPAD_UP);
    pad.move(SDL_GAMEPAD_AXIS_LEFTY, -32768); // SDL: stick fully up
    pad.move(SDL_GAMEPAD_AXIS_RIGHTX, 16384);
    pad.move(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 32767);
    connection->poll();

    const auto& state = connection->state();
    CHECK(state.inputs.west_click());
    CHECK(state.inputs.dpad_up_click());
    CHECK_FALSE(state.inputs.south_click());
    CHECK(state.inputs.thumbstick_left_y() == Catch::Approx(1.0f)); // up-positive
    CHECK(state.inputs.thumbstick_right_x() == Catch::Approx(0.5f).margin(1e-4));
    CHECK(state.inputs.trigger_right_value() == Catch::Approx(1.0f));
    CHECK(state.inputs.trigger_left_value() == 0.0f);

    pad.press(SDL_GAMEPAD_BUTTON_WEST, false);
    connection->poll();
    CHECK_FALSE(connection->state().inputs.west_click());

    pad.unplug();
    connection->poll();
    CHECK(connection->status() == ConnectionStatus::NoDevice);
    CHECK_FALSE(connection->state().inputs.dpad_up_click());
    CHECK(connection->state().inputs.thumbstick_left_y() == 0.0f);
}

TEST_CASE("A gamepad plugged in later is picked up", "[gamepad][connection]")
{
    auto connection = make_connection();
    connection->poll();
    REQUIRE(connection->status() == ConnectionStatus::NoDevice);

    VirtualPad pad("Late Pad");
    connection->poll();

    CHECK(connection->status() == ConnectionStatus::Connected);
}

TEST_CASE("The player number selects the Nth connected gamepad and keeps it", "[gamepad][connection]")
{
    auto connection = make_connection(2);
    VirtualPad first("First Pad", 0x0001);
    connection->poll();
    CHECK(connection->status() == ConnectionStatus::NoDevice);

    VirtualPad second("Second Pad", 0x0002);
    connection->poll();
    REQUIRE(connection->status() == ConnectionStatus::Connected);
    CHECK(connection->description().find("1209:0002") != std::string::npos);

    // Player 1 leaving does not move player 2 onto another pad.
    first.unplug();
    connection->poll();
    CHECK(connection->status() == ConnectionStatus::Connected);
    CHECK(connection->description().find("1209:0002") != std::string::npos);
}

TEST_CASE("A player number below 1 is rejected", "[gamepad][connection]")
{
    CHECK_THROWS_AS(GamepadConnection(GamepadConnectionConfig{ 0 }), std::invalid_argument);
}
