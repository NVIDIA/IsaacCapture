// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/gamepad/gamepad_connection.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace plugins
{
namespace gamepad
{

namespace
{

// SDL reports [-32768, 32767]; sticks map to [-1, 1] and triggers to [0, 1].
float stick(SDL_Gamepad* gamepad, SDL_GamepadAxis axis, bool flip = false)
{
    const float value = static_cast<float>(SDL_GetGamepadAxis(gamepad, axis)) / 32767.0f;
    return std::clamp(flip ? -value : value, -1.0f, 1.0f);
}

float trigger(SDL_Gamepad* gamepad, SDL_GamepadAxis axis)
{
    return std::clamp(static_cast<float>(SDL_GetGamepadAxis(gamepad, axis)) / 32767.0f, 0.0f, 1.0f);
}

} // namespace

void GamepadState::clear()
{
    inputs = core::GamepadInputState();
}

GamepadConnection::GamepadConnection(GamepadConnectionConfig config) : config_(config)
{
    if (config_.player < 1)
        throw std::invalid_argument("GamepadConnection: player must be 1 or more");

    // The plugin launcher stops the plugin with a signal; SDL must not intercept it.
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    // There is no window, so the process never has focus: read the pad regardless.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD))
        throw std::runtime_error(std::string("GamepadConnection: SDL_Init failed: ") + SDL_GetError());
    // poll() reads state directly, so nothing drains an event queue.
    SDL_SetJoystickEventsEnabled(false);
    SDL_SetGamepadEventsEnabled(false);
}

GamepadConnection::~GamepadConnection()
{
    if (gamepad_ != nullptr)
        SDL_CloseGamepad(gamepad_);
    SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
}

void GamepadConnection::poll()
{
    // Refreshes every pad's state and detects pads added or removed.
    SDL_UpdateGamepads();
    if (gamepad_ != nullptr && !SDL_GamepadConnected(gamepad_))
        close_device();
    if (gamepad_ == nullptr)
        open_device();
    if (gamepad_ != nullptr)
        read_state();
}

void GamepadConnection::open_device()
{
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    if (ids == nullptr)
    {
        set_status(ConnectionStatus::NoDevice);
        return;
    }
    // Instance IDs only grow, so ascending order is connection order.
    std::sort(ids, ids + count);
    const SDL_JoystickID id = count >= config_.player ? ids[config_.player - 1] : 0;
    SDL_free(ids);
    if (id == 0)
    {
        set_status(ConnectionStatus::NoDevice);
        return;
    }

    gamepad_ = SDL_OpenGamepad(id);
    if (gamepad_ == nullptr)
    {
        set_status(ConnectionStatus::OpenFailed, SDL_GetError());
        return;
    }

    char usb_ids[16];
    std::snprintf(usb_ids, sizeof(usb_ids), "%04x:%04x", SDL_GetGamepadVendor(gamepad_), SDL_GetGamepadProduct(gamepad_));
    const char* name = SDL_GetGamepadName(gamepad_);
    const char* type = SDL_GetGamepadStringForType(SDL_GetGamepadType(gamepad_));
    description_ = std::string(name != nullptr ? name : "Unknown gamepad") + " (" +
                   (type != nullptr ? type : "unknown") + ", " + usb_ids + ")";
    state_.clear();
    set_status(ConnectionStatus::Connected);
}

void GamepadConnection::close_device()
{
    SDL_CloseGamepad(gamepad_);
    gamepad_ = nullptr;
    logger_->info("Gamepad {} disconnected", description_);
    description_.clear();
    state_.clear();
}

void GamepadConnection::read_state()
{
    const auto pressed = [this](SDL_GamepadButton button) { return SDL_GetGamepadButton(gamepad_, button); };
    // SDL reports stick Y down-positive; flip it to up-positive, as OpenXR's thumbstick y is.
    state_.inputs = core::GamepadInputState(
        pressed(SDL_GAMEPAD_BUTTON_SOUTH), pressed(SDL_GAMEPAD_BUTTON_EAST), pressed(SDL_GAMEPAD_BUTTON_WEST),
        pressed(SDL_GAMEPAD_BUTTON_NORTH), pressed(SDL_GAMEPAD_BUTTON_BACK), pressed(SDL_GAMEPAD_BUTTON_START),
        pressed(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER), pressed(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER),
        pressed(SDL_GAMEPAD_BUTTON_LEFT_STICK), pressed(SDL_GAMEPAD_BUTTON_RIGHT_STICK),
        pressed(SDL_GAMEPAD_BUTTON_DPAD_UP), pressed(SDL_GAMEPAD_BUTTON_DPAD_DOWN),
        pressed(SDL_GAMEPAD_BUTTON_DPAD_LEFT), pressed(SDL_GAMEPAD_BUTTON_DPAD_RIGHT),
        stick(gamepad_, SDL_GAMEPAD_AXIS_LEFTX), stick(gamepad_, SDL_GAMEPAD_AXIS_LEFTY, true),
        stick(gamepad_, SDL_GAMEPAD_AXIS_RIGHTX), stick(gamepad_, SDL_GAMEPAD_AXIS_RIGHTY, true),
        trigger(gamepad_, SDL_GAMEPAD_AXIS_LEFT_TRIGGER), trigger(gamepad_, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
}

void GamepadConnection::set_status(ConnectionStatus status, std::string error)
{
    const bool report = !status_logged_ || status != status_;
    status_ = status;
    error_ = std::move(error);
    status_logged_ = true;
    if (!report)
        return;
    switch (status)
    {
    case ConnectionStatus::Connected:
        logger_->info("Opened gamepad {} as player {}", description_, config_.player);
        break;
    case ConnectionStatus::NoDevice:
        logger_->info("No gamepad for player {}; waiting for one to connect", config_.player);
        break;
    case ConnectionStatus::OpenFailed:
        logger_->warn("Cannot open gamepad for player {}: {}", config_.player, error_);
        break;
    }
}

} // namespace gamepad
} // namespace plugins
