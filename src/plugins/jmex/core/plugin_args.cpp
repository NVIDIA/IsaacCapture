// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/plugin_args.hpp"

#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace plugins
{
namespace jmex
{

namespace
{

constexpr int kMaxPort = 65535;
// The SDK keeps a channel in one byte.
constexpr int kMaxChannel = 255;

//! The value of `--name=value` in @p arg, or nullopt when @p arg is another flag.
std::optional<std::string> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
    {
        return std::string(arg.substr(name.size() + 1));
    }
    return std::nullopt;
}

//! The `--name=` part of @p arg, which errors name the flag by.
std::string flag_name(const std::string& arg)
{
    return arg.substr(0, arg.find('=') + 1);
}

//! @p value as a whole number in [min, max], or an error naming the flag and the value.
//! Partially parsed values are rejected: "2.9" would otherwise select channel 2.
int parse_whole_number(const std::string& arg, const std::string& value, int min, int max)
{
    int parsed = 0;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto [ptr, error] = std::from_chars(begin, end, parsed);
    if (error != std::errc{} || ptr != end || parsed < min || parsed > max)
    {
        throw std::runtime_error(flag_name(arg) + " takes a whole number from " + std::to_string(min) + " to " +
                                 std::to_string(max) + ", got '" + value + "'");
    }
    return parsed;
}

//! @p value; an empty one, which "--bind-ip=$IP" leaves when IP is unset, is an error naming the flag.
std::string require_value(const std::string& arg, const std::string& value)
{
    if (value.empty())
    {
        throw std::runtime_error(flag_name(arg) + " needs a value");
    }
    return value;
}

DeviceClock parse_device_clock(const std::string& arg, const std::string& value)
{
    if (value == "device")
    {
        return DeviceClock::DEVICE;
    }
    if (value == "local")
    {
        return DeviceClock::LOCAL;
    }
    throw std::runtime_error(flag_name(arg) + " takes 'device' or 'local', got '" + value + "'");
}

} // namespace

std::optional<PluginArgs> parse_plugin_args(int argc, const char* const* argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            return std::nullopt;
        }
    }

    PluginArgs args;
    // Do not add positional arguments: the PluginManager passes --plugin-root-id=<id> ahead of
    // plugin.yaml's and PluginConfig's.
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (auto value = flag_value(arg, "--channel"))
        {
            args.receiver.channel = parse_whole_number(arg, *value, 0, kMaxChannel);
        }
        else if (auto value = flag_value(arg, "--tcp-port"))
        {
            args.receiver.tcp_port = parse_whole_number(arg, *value, 1, kMaxPort);
        }
        else if (auto value = flag_value(arg, "--udp-start-port"))
        {
            args.receiver.udp_start_port = parse_whole_number(arg, *value, 1, kMaxPort);
        }
        else if (auto value = flag_value(arg, "--bind-ip"))
        {
            args.receiver.bind_ip = require_value(arg, *value);
        }
        else if (auto value = flag_value(arg, "--broadcast-ip"))
        {
            args.receiver.broadcast_ip = require_value(arg, *value);
        }
        else if (auto value = flag_value(arg, "--collection-id"))
        {
            args.collection_id = require_value(arg, *value);
        }
        else if (auto value = flag_value(arg, "--device-clock"))
        {
            args.device_clock = parse_device_clock(arg, *value);
        }
        else if (flag_value(arg, "--plugin-root-id"))
        {
            // Not used by this plugin.
        }
        else
        {
            throw std::runtime_error("unknown argument '" + arg + "'; --help lists the options");
        }
    }

    return args;
}

std::string plugin_usage()
{
    const PluginArgs defaults;
    const ReceiverConfig& receiver = defaults.receiver;
    return "Usage: jmex_plugin [--name=value ...]\n"
           "\n"
           "  --channel=N            MOXI channel, 0.." +
           std::to_string(kMaxChannel) + " (default " + std::to_string(receiver.channel) +
           "; Player's file replay is 255)\n"
           "  --collection-id=ID     collection the body pose is published on (default " +
           defaults.collection_id +
           ")\n"
           "  --bind-ip=ADDR         local address Player connects to and advertisements are sent from "
           "(default " +
           kEveryInterface +
           ")\n"
           "  --broadcast-ip=ADDR    where channel advertisements are sent (default " +
           kSdkDefaultBroadcastIp +
           ")\n"
           "  --tcp-port=N           TCP port advertised to Player (default " +
           std::to_string(receiver.tcp_port) +
           ")\n"
           "  --udp-start-port=N     base of the per-channel data ports (default " +
           std::to_string(receiver.udp_start_port) +
           ")\n"
           "  --device-clock=CLOCK   raw device clock: device (Player's frame time) or local (default " +
           device_clock_name(defaults.device_clock) +
           ")\n"
           "  --help, -h             print this and exit\n"
           "\n"
           "README.md says when to set each.\n";
}

} // namespace jmex
} // namespace plugins
