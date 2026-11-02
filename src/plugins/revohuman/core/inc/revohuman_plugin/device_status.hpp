// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pipeline.hpp"
#include "source.hpp"
#include "types.hpp"

#include <schema/plugin_device_status_generated.h>

#include <array>
#include <optional>
#include <string>

namespace plugins
{
namespace revohuman
{

//! One device path in plugin.yaml and what it reports.
struct DeviceStatusRow
{
    std::string path;
    core::PluginDeviceState state = core::PluginDeviceState_UNKNOWN;
    core::PluginDeviceReason reason = core::PluginDeviceReason_NONE;
    std::string error;
};

//! What the plugin knows about one hand when it reports device status.
struct HandStatusInput
{
    HandHealth health;
    std::optional<Source::GloveNote> note; //!< From the source, for a glove that gives no samples.
};

/*!
 * @brief The /hand, /glove and /tactile rows of one hand, in that order.
 *
 * Only the state/reason pairs that TeleopSession accepts are used
 * (teleop_session_manager/status_monitor.py). A device that has not given data yet is UNKNOWN,
 * not DISCONNECTED, so a plugin started before its gloves are plugged in is not reported as having
 * lost them.
 */
std::array<DeviceStatusRow, 3> make_hand_status(Hand hand, const HandStatusInput& input);

} // namespace revohuman
} // namespace plugins
