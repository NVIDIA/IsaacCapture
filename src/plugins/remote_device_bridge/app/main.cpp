// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <plugin_utils/openxr_plugin_session.hpp>
#include <remote_device_bridge/bridge_server.hpp>
#include <remote_plugin_session/remote_address.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "lock-free atomic bool is required for signal safety");

std::atomic<bool> g_stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

std::optional<std::string> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
    {
        return std::string(arg.substr(name.size() + 1));
    }
    return std::nullopt;
}

//! Whole-string integer in [min, max of T]: std::stoi accepts trailing junk and std::stoul wraps "-1".
template <typename T>
T parse_integer(std::string_view flag, const std::string& value, T min)
{
    size_t used = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &used);
    }
    catch (const std::exception&)
    {
        used = 0;
    }
    if (used == 0 || used != value.size() || parsed < static_cast<long long>(min) ||
        static_cast<unsigned long long>(parsed) > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
    {
        throw std::invalid_argument(std::string(flag) + " must be an integer >= " + std::to_string(min) + ", got '" +
                                    value + "'");
    }
    return static_cast<T>(parsed);
}

void print_usage()
{
    const remote_device_bridge::BridgeOptions defaults;
    std::cout
        << "Usage: remote_device_bridge [--listen=<ip:port>] [--ws-listen=<ip:port>] [--app-name=<str>]\n"
           "                            [--wrist-rate-hz=<n>] [--max-clients=<n>] [--max-sample-age-ms=<n>]\n"
           "                            [--max-schema-bytes=<n>] [--clock-source=two-way|one-way]\n"
           "                            [--stats-csv=<path>]\n"
        << "  --listen             bind address (default " << defaults.listen.host << ":" << defaults.listen.port << ")\n"
        << "  --ws-listen          also accept WebSocket clients (--remote=ws://...) here; no TLS (default off,\n"
        << "                       port " << remote_plugin_session::kDefaultBridgeWsPort << " when omitted)\n"
        << "  --app-name           OpenXR app name; defaults to the name the client sends in HELLO\n"
        << "  --wrist-rate-hz      wrist sample rate towards the client (default " << defaults.wrist_rate_hz << ")\n"
        << "  --max-clients        concurrent clients; further ones get HELLO_ACK 'bridge full' (default "
        << defaults.max_clients << ")\n"
        << "  --max-sample-age-ms  drop samples older than this instead of pushing them; 0 = off (default "
        << defaults.max_sample_age->count() << ")\n"
        << "  --max-schema-bytes   refuse schema channels whose sample size exceeds this (default "
        << defaults.max_schema_bytes << ", the\n"
        << "                       largest the runtime accepts)\n"
        << "  --clock-source       two-way: PING/PONG min-RTT clock offset, one-way if the client never answers\n"
        << "                       within 2 s; one-way: frame-header minimum, no PINGs (default two-way)\n"
        << "  --stats-csv          write one row per pushed hand sample to this file (truncated at start)\n";
}

remote_device_bridge::BridgeOptions parse_args(int argc, char** argv)
{
    remote_device_bridge::BridgeOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (auto value = flag_value(arg, "--listen"))
        {
            options.listen = remote_plugin_session::parse_endpoint(*value, remote_plugin_session::kDefaultBridgePort);
        }
        else if (auto value = flag_value(arg, "--ws-listen"))
        {
            options.ws_listen =
                remote_plugin_session::parse_endpoint(*value, remote_plugin_session::kDefaultBridgeWsPort);
        }
        else if (auto value = flag_value(arg, "--app-name"))
        {
            if (value->empty())
            {
                throw std::invalid_argument("--app-name must not be empty");
            }
            options.app_name = *value;
        }
        else if (auto value = flag_value(arg, "--wrist-rate-hz"))
        {
            options.wrist_rate_hz = parse_integer<int>("--wrist-rate-hz", *value, 1);
        }
        else if (auto value = flag_value(arg, "--max-clients"))
        {
            options.max_clients = parse_integer<int>("--max-clients", *value, 1);
        }
        else if (auto value = flag_value(arg, "--max-sample-age-ms"))
        {
            const int ms = parse_integer<int>("--max-sample-age-ms", *value, 0);
            options.max_sample_age =
                ms == 0 ? std::nullopt : std::optional<std::chrono::milliseconds>(std::chrono::milliseconds(ms));
        }
        else if (auto value = flag_value(arg, "--max-schema-bytes"))
        {
            options.max_schema_bytes = parse_integer<size_t>("--max-schema-bytes", *value, 1);
        }
        else if (auto value = flag_value(arg, "--clock-source"))
        {
            if (*value != "two-way" && *value != "one-way")
            {
                throw std::invalid_argument("--clock-source must be two-way or one-way");
            }
            options.clock_source = *value == "two-way" ? remote_device_bridge::ClockSource::TwoWay :
                                                         remote_device_bridge::ClockSource::OneWay;
        }
        else if (auto value = flag_value(arg, "--stats-csv"))
        {
            options.stats_csv = *value;
        }
        else if (flag_value(arg, "--plugin-root-id"))
        {
            // Injected by the PluginManager; not used by the bridge.
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage();
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown argument '" + arg + "'");
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv)
try
{
    const remote_device_bridge::BridgeOptions options = parse_args(argc, argv);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    remote_device_bridge::BridgeServer server(options, plugin_utils::make_openxr_plugin_session_factory);
    server.run(g_stop_requested);
    return 0;
}
catch (const std::exception& e)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.main");
    logger->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.main");
    logger->error("{}: Unknown error occurred", argv[0]);
    return 1;
}
