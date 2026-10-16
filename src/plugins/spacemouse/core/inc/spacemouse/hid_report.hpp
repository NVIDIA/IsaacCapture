// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace plugins
{
namespace spacemouse
{

//! Buttons a button report can set; more than any SpaceMouse model has. Bounds the pushed sample.
inline constexpr std::size_t kMaxButtons = 32;

//! Current axis and button state of a SpaceMouse.
struct SpaceMouseState
{
    std::array<float, 3> translation{}; //!< [x, y, z] in [-1, 1], device axis order.
    std::array<float, 3> rotation{}; //!< [x, y, z] in [-1, 1], device axis order.
    std::vector<uint8_t> buttons; //!< 1 while held; sized by the last button report.

    //! The state of a device at rest: zero axes, no buttons held.
    void clear();
};

/*!
 * @brief Applies one raw HID input report to @p state.
 *
 * Report 1 carries translation (and, for a combined-report device such as the 3Dconnexion
 * Universal Receiver, rotation too), report 2 rotation, and report 3 the button bitmask (bit i of
 * the bytes after the report ID is button i, up to kMaxButtons). Axes are 16-bit little-endian
 * counts, where 350 is full deflection.
 *
 * @return false for a report ID this parser does not handle or a report too short for its ID.
 */
bool apply_report(SpaceMouseState& state, const uint8_t* report, std::size_t size, bool combined_report);

} // namespace spacemouse
} // namespace plugins
