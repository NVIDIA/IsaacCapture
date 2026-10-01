// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/gamepad_tracker_base.hpp>
#include <schema/gamepad_generated.h>

#include <optional>
#include <string>
#include <string_view>

namespace core
{

//! The first Linux joystick-API gamepad (`/dev/input/by-path/*-joystick`, i.e. a `/dev/input/jsN`
//! node; the lowest `/dev/input/jsN` when no pad has a by-path link, e.g. over Bluetooth), or
//! nullopt when none is connected.
std::optional<std::string> discover_gamepad_device();

/*!
 * @brief In-process gamepad: reads a Linux joystick-API device (`/dev/input/jsN`) directly.
 *
 * Needs no OpenXR extension, no plugin process and no `input` group membership: udev grants
 * the logged-in seat user access to joysticks. The device is opened when the session starts,
 * and reopened (rediscovered, when no path was given) after a disconnect. The session
 * publishes its state once per update and records it to MCAP; in replay the recorded state
 * is returned instead.
 */
class GamepadTracker : public ITracker
{
public:
    //! @param device_path Joystick device to read (e.g. `/dev/input/js0`); empty picks the first
    //!        one discover_gamepad_device() finds.
    explicit GamepadTracker(std::string device_path = {});

    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    //! The configured device path; empty when the device is discovered.
    const std::string& device_path() const
    {
        return device_path_;
    }

    //! Empty while no gamepad is connected (or, in replay, when the recording has no sample).
    const Serialized<GamepadOutput>& get_data(const ITrackerSession& session) const;

private:
    static constexpr const char* TRACKER_NAME = "GamepadTracker";

    std::string device_path_;
};

} // namespace core
