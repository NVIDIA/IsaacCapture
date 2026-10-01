// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/spacemouse_tracker_base.hpp>
#include <schema/spacemouse_generated.h>

#include <optional>
#include <string>
#include <string_view>

namespace core
{

//! A connected 3Dconnexion SpaceMouse-family HID device.
struct SpaceMouseDevice
{
    std::string device_path; //!< `/dev/hidrawN`
    //! The device packs translation and rotation into one 13-byte report (3Dconnexion Universal
    //! Receiver) instead of two separate 7-byte reports.
    bool combined_report = false;
};

//! The first connected SpaceMouse of a validated model (matched on the hidraw device's HID_NAME
//! under `/sys/class/hidraw`), or nullopt when none is connected.
std::optional<SpaceMouseDevice> discover_spacemouse_device();

/*!
 * @brief In-process SpaceMouse: reads a 3Dconnexion HID device (`/dev/hidrawN`) directly.
 *
 * Needs no OpenXR extension and no plugin process. Reading `/dev/hidrawN` needs a udev rule
 * granting the user access (see examples/teleop/python/spacemouse_printer_example.py). The device
 * is opened when the session starts, and reopened (rediscovered, when no path was given) after a
 * disconnect. The session publishes its state once per update and records it to MCAP; in replay
 * the recorded state is returned instead.
 *
 * Validated devices: SpaceMouse Compact, SpaceMouse Wireless, SpaceNavigator, SpaceNavigator for Notebooks,
 * 3Dconnexion Universal Receiver.
 */
class SpaceMouseTracker : public ITracker
{
public:
    //! @param device_path HID device to read (e.g. `/dev/hidraw3`); empty picks the first one
    //!        discover_spacemouse_device() finds.
    //! @param combined_report Only used with an explicit @p device_path: the device packs
    //!        translation and rotation into one report (3Dconnexion Universal Receiver).
    explicit SpaceMouseTracker(std::string device_path = {}, bool combined_report = false);

    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    //! The configured device path; empty when the device is discovered.
    const std::string& device_path() const
    {
        return device_path_;
    }

    bool combined_report() const
    {
        return combined_report_;
    }

    //! Empty while no SpaceMouse is connected (or, in replay, when the recording has no sample).
    const Serialized<SpaceMouseOutput>& get_data(const ITrackerSession& session) const;

private:
    static constexpr const char* TRACKER_NAME = "SpaceMouseTracker";

    std::string device_path_;
    bool combined_report_;
};

} // namespace core
