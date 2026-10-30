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
// The SDK keeps a channel in one byte; 0 is no channel.
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

} // namespace

PluginArgs parse_plugin_args(int argc, const char* const* argv, const char* device_clock_env)
{
    PluginArgs args;

    args.device_clock = choose_device_clock(nullptr, device_clock_env);

    // Do not add positional arguments: the PluginManager injects --plugin-root-id=<id> ahead of
    // plugin.yaml's and PluginConfig's.
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--channel="))
        {
            args.receiver.channel =
                parse_whole_number("--channel=", arg.substr(std::string("--channel=").size()), 1, k_max_channel);
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
            args.receiver.bind_ip = arg.substr(std::string("--bind-ip=").size());
        }
        else if (starts_with(arg, "--broadcast-ip="))
        {
            args.receiver.broadcast_ip = arg.substr(std::string("--broadcast-ip=").size());
        }
        else if (starts_with(arg, "--collection-id="))
        {
            args.collection_id = arg.substr(std::string("--collection-id=").size());
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
            throw std::runtime_error("unknown argument '" + arg + "': every option is --name=value");
        }
    }

    // Checked once both are final, whichever came last. Both are range-checked above, so the sum
    // cannot overflow.
    const int data_port = args.receiver.udp_start_port + 1 + args.receiver.channel;
    if (data_port > k_max_port)
    {
        throw std::runtime_error("--udp-start-port=" + std::to_string(args.receiver.udp_start_port) + " puts channel " +
                                 std::to_string(args.receiver.channel) + "'s data port at " +
                                 std::to_string(data_port) + ", past " + std::to_string(k_max_port));
    }

    return args;
}

} // namespace jmex
} // namespace plugins
