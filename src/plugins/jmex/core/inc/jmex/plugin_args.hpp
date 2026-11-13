// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_clock.hpp"
#include "receiver_config.hpp"

#include <optional>
#include <string>

namespace plugins
{
namespace jmex
{

//! What the plugin's command line sets.
struct PluginArgs
{
    ReceiverConfig receiver;
    //! Must equal LiveFullBodyTrackerJmexImpl::DEFAULT_COLLECTION_ID.
    std::string collection_id = "jmex_full_body";
    DeviceClock device_clock = DeviceClock::DEVICE;
};

/*!
 * @brief Parse the plugin's named arguments; the last occurrence of a flag wins.
 *
 * @return Empty when --help or -h is given, whatever else is. Prints nothing.
 * @throws std::runtime_error naming the argument it cannot use.
 */
std::optional<PluginArgs> parse_plugin_args(int argc, const char* const* argv);

//! The flags and their defaults, for --help.
std::string plugin_usage();

} // namespace jmex
} // namespace plugins
