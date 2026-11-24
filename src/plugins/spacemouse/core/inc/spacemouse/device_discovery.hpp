// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

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
 * @brief The axis report layout a HID report descriptor declares, if apply_report() can read it.
 *
 * @return true when report 1 declares X, Y, Z, Rx, Ry and Rz (combined); false when report 1
 *         declares X, Y, Z and report 2 Rx, Ry, Rz (separate); nullopt for any other layout,
 *         including an interface with no motion axes or a malformed descriptor.
 */
std::optional<bool> combined_report_from_descriptor(std::span<const uint8_t> descriptor);

/*!
 * @brief The first SpaceMouse of a validated model that carries the motion axes.
 *
 * Scans `<hidraw_class_dir>/hidrawN` in numeric order, matching `HID_NAME` in `device/uevent`
 * against the validated models. A matching node whose `device/report_descriptor` declares no
 * supported axis layout is skipped: a receiver exposes several nodes under one name, and only one
 * carries the axes. The report layout comes from the descriptor, or from the model when the
 * descriptor cannot be read. Returns `<dev_dir>/hidrawN`, or nullopt when none is connected.
 */
std::optional<DeviceLocation> find_device(const std::filesystem::path& hidraw_class_dir = "/sys/class/hidraw",
                                          const std::filesystem::path& dev_dir = "/dev");

} // namespace spacemouse
} // namespace plugins
