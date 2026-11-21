// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <log_bridge/logger.hpp>
#include <schema/gamepad_generated.h>

#include <memory>
#include <string>

struct SDL_Gamepad;

namespace plugins
{
namespace gamepad
{

//! Why the pad is or is not open.
enum class ConnectionStatus
{
    Connected, //!< The pad is open and being read.
    NoDevice, //!< Fewer gamepads are connected than the configured player number.
    OpenFailed, //!< The pad is connected but SDL could not open it.
};

struct GamepadConnectionConfig
{
    //! 1-based: read the Nth connected gamepad, in connection order. The pad is kept until it
    //! disconnects, then the Nth one is taken again.
    int player = 1;
};

struct GamepadState
{
    //! Sticks in [-1, 1] with right and up positive, triggers in [0, 1] (see gamepad.fbs).
    core::GamepadInputState inputs;

    //! The state of a pad that is not open: no buttons held, axes at rest.
    void clear();
};

/*!
 * @brief Reads one gamepad through SDL3 in the standard layout, reopening after a disconnect.
 *
 * SDL resolves each pad's own button and axis numbering through its built-in drivers and
 * SDL_GameControllerDB mappings. poll() refreshes SDL, opens the configured player's pad when one
 * is connected, and reads its state; a closed pad's state is cleared.
 */
class GamepadConnection
{
public:
    //! Initializes SDL's gamepad subsystem; throws std::runtime_error when it cannot.
    explicit GamepadConnection(GamepadConnectionConfig config);
    ~GamepadConnection();

    GamepadConnection(const GamepadConnection&) = delete;
    GamepadConnection& operator=(const GamepadConnection&) = delete;

    void poll();

    ConnectionStatus status() const
    {
        return status_;
    }
    //! Why the last open failed; empty unless status() is OpenFailed.
    const std::string& error() const
    {
        return error_;
    }
    //! The open pad's name, type and USB IDs; empty while none is open.
    const std::string& description() const
    {
        return description_;
    }
    const GamepadState& state() const
    {
        return state_;
    }

private:
    void open_device();
    void close_device();
    void read_state();
    void set_status(ConnectionStatus status, std::string error = {});

    GamepadConnectionConfig config_;
    SDL_Gamepad* gamepad_ = nullptr;
    ConnectionStatus status_ = ConnectionStatus::NoDevice;
    bool status_logged_ = false; //!< The current status() has been logged once.
    std::string error_;
    std::string description_;
    GamepadState state_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.plugins.gamepad.GamepadConnection");
};

} // namespace gamepad
} // namespace plugins
