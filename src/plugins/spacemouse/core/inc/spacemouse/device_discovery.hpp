// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>

namespace plugins
{
namespace spacemouse
{

//! A connected SpaceMouse's HID device node.
struct DeviceLocation
{
    std::filesystem::path path; //!< e.g. `/dev/hidraw3`
    //! The device packs translation and rotation into one report (3Dconnexion Universal Receiver).
    bool combined_report = false;
};

/*!
 * @brief The first SpaceMouse of a validated model, matched on each hidraw device's `HID_NAME`.
 *
 * Scans `<hidraw_class_dir>/hidrawN/device/uevent` in name order and returns `<dev_dir>/hidrawN`
 * for the first match, or nullopt when none is connected.
 */
std::optional<DeviceLocation> find_device(const std::filesystem::path& hidraw_class_dir = "/sys/class/hidraw",
                                          const std::filesystem::path& dev_dir = "/dev");

} // namespace spacemouse
} // namespace plugins
