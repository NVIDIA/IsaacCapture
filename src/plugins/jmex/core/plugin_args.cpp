// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/plugin_args.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace jmex
{

namespace
{

constexpr int k_max_port = 65535;
// The SDK keeps a channel in one byte.
constexpr int k_max_channel = 255;

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

//! The flag's value as a whole number in [min, max], or an error naming the flag and the value.
//! Partially parsed values are rejected: "2.9" would otherwise select channel 2.
int parse_whole_number(const std::string& flag, const std::string& value, int min, int max)
{
    const std::string rejection = flag + " takes a whole number from " + std::to_string(min) + " to " +
                                  std::to_string(max) + ", got '" + value + "'";
    int parsed = 0;
    try
    {
        std::size_t consumed = 0;
        parsed = std::stoi(value, &consumed);
        if (consumed != value.size())
        {
            throw std::invalid_argument("trailing characters after the number");
        }
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(rejection);
    }
    if (parsed < min || parsed > max)
    {
        throw std::runtime_error(rejection);
    }
    return parsed;
}

//! The flag's value; an empty one, which "--bind-ip=$IP" leaves when IP is unset, is an error naming
//! the flag.
std::string require_value(const std::string& flag, const std::string& value)
{
    if (value.empty())
    {
        throw std::runtime_error(flag + " needs a value");
    }
    return value;
}

} // namespace

PluginArgs parse_plugin_args(int argc, const char* const* argv, const char* device_clock_env)
{
    PluginArgs args;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            args.help = true;
            return args;
        }
    }

    args.device_clock = choose_device_clock(nullptr, device_clock_env);

    // Do not add positional arguments: the PluginManager passes --plugin-root-id=<id> ahead of
    // plugin.yaml's and PluginConfig's.
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--channel="))
        {
            args.receiver.channel =
                parse_whole_number("--channel=", arg.substr(std::string("--channel=").size()), 0, k_max_channel);
        }
        else if (starts_with(arg, "--tcp-port="))
        {
            args.receiver.tcp_port =
                parse_whole_number("--tcp-port=", arg.substr(std::string("--tcp-port=").size()), 1, k_max_port);
        }
        else if (starts_with(arg, "--udp-start-port="))
        {
            args.receiver.udp_start_port = parse_whole_number(
                "--udp-start-port=", arg.substr(std::string("--udp-start-port=").size()), 1, k_max_port);
        }
        else if (starts_with(arg, "--bind-ip="))
        {
            args.receiver.bind_ip = require_value("--bind-ip=", arg.substr(std::string("--bind-ip=").size()));
        }
        else if (starts_with(arg, "--broadcast-ip="))
        {
            args.receiver.broadcast_ip =
                require_value("--broadcast-ip=", arg.substr(std::string("--broadcast-ip=").size()));
        }
        else if (starts_with(arg, "--collection-id="))
        {
            args.collection_id = require_value("--collection-id=", arg.substr(std::string("--collection-id=").size()));
        }
        else if (starts_with(arg, "--device-clock="))
        {
            args.device_clock =
                choose_device_clock(arg.substr(std::string("--device-clock=").size()).c_str(), device_clock_env);
        }
        else if (starts_with(arg, "--plugin-root-id="))
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
           std::to_string(k_max_channel) + " (default " + std::to_string(receiver.channel) +
           "; Player's file replay is 255)\n"
           "  --collection-id=ID     collection the body pose is published on (default " +
           defaults.collection_id +
           ")\n"
           "  --bind-ip=ADDR         local address Player connects to and advertisements are sent from "
           "(default every interface)\n"
           "  --broadcast-ip=ADDR    where channel advertisements are sent (default 255.255.255.255)\n"
           "  --tcp-port=N           TCP port advertised to Player (default " +
           std::to_string(receiver.tcp_port) +
           ")\n"
           "  --udp-start-port=N     base of the per-channel data ports (default " +
           std::to_string(receiver.udp_start_port) +
           ")\n"
           "  --device-clock=CLOCK   raw device clock: device (Player's frame time, default) or local; "
           "JMEX_DEVICE_CLOCK sets it too\n"
           "  --help, -h             print this and exit\n"
           "\n"
           "README.md says when to set each.\n";
}

} // namespace jmex
} // namespace plugins
