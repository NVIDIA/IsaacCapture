// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_clock.hpp"
#include "receiver_config.hpp"

#include <string>

namespace plugins
{
namespace jmex
{

//! What the plugin's command line sets.
struct PluginArgs
{
    ReceiverConfig receiver;
    std::string collection_id = "jmex_full_body";
    DeviceClock device_clock = DeviceClock::DEVICE;
};

/*!
 * @brief Parse the plugin's named arguments; the last occurrence of a flag wins.
 *
 * ``device_clock_env`` is ``JMEX_DEVICE_CLOCK``, which ``--device-clock=`` overrides. Throws
 * std::runtime_error naming the argument it cannot use.
 */
PluginArgs parse_plugin_args(int argc, const char* const* argv, const char* device_clock_env);

} // namespace jmex
} // namespace plugins
