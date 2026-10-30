// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <plugin_utils/openxr_plugin_session.hpp>
#include <remote_device_bridge/bridge_server.hpp>
#include <remote_plugin_session/remote_address.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

void print_usage()
{
    std::cout << "Usage: remote_device_bridge [--listen=<ip:port>] [--ws-listen=<ip:port>] [--app-name=<str>]\n"
                 "                            [--wrist-rate-hz=<n>] [--max-clients=<n>] [--max-sample-age-ms=<n>]\n"
                 "                            [--max-schema-bytes=<n>] [--clock-source=two-way|one-way]\n"
                 "                            [--stats-csv=<path>]\n"
                 "  --listen             bind address (default 0.0.0.0:47500)\n"
                 "  --ws-listen          also accept WebSocket clients (--remote=ws://...) here; no TLS (default off)\n"
                 "  --app-name           OpenXR app name; defaults to the name the client sends in HELLO\n"
                 "  --wrist-rate-hz      wrist sample rate towards the client (default 90)\n"
                 "  --max-clients        concurrent clients; further ones get HELLO_ACK 'bridge full' (default 8)\n"
                 "  --max-sample-age-ms  drop samples older than this instead of pushing them; 0 = off (default 250)\n"
                 "  --max-schema-bytes   refuse schema channels whose sample size exceeds this (default 32768, the\n"
                 "                       largest the runtime accepts)\n"
                 "  --clock-source       two-way: PING/PONG min-RTT clock offset, one-way if the client never answers\n"
                 "                       within 2 s; one-way: frame-header minimum, no PINGs (default two-way)\n"
                 "  --stats-csv          write one row per pushed hand sample to this file (truncated at start)\n";
}

remote_device_bridge::BridgeOptions parse_args(int argc, char** argv)
{
    remote_device_bridge::BridgeOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--listen="))
        {
            options.listen = remote_plugin_session::parse_endpoint(
                arg.substr(std::string("--listen=").size()), remote_plugin_session::kDefaultBridgePort);
        }
        else if (starts_with(arg, "--ws-listen="))
        {
            options.ws_listen = remote_plugin_session::parse_endpoint(
                arg.substr(std::string("--ws-listen=").size()), remote_plugin_session::kDefaultBridgeWsPort);
        }
        else if (starts_with(arg, "--app-name="))
        {
            options.app_name = arg.substr(std::string("--app-name=").size());
        }
        else if (starts_with(arg, "--wrist-rate-hz="))
        {
            options.wrist_rate_hz = std::stoi(arg.substr(std::string("--wrist-rate-hz=").size()));
        }
        else if (starts_with(arg, "--max-clients="))
        {
            options.max_clients = std::stoi(arg.substr(std::string("--max-clients=").size()));
        }
        else if (starts_with(arg, "--max-sample-age-ms="))
        {
            options.max_sample_age_ms = std::stoi(arg.substr(std::string("--max-sample-age-ms=").size()));
        }
        else if (starts_with(arg, "--max-schema-bytes="))
        {
            // stoul would wrap "-1" to ULONG_MAX and silently disable the cap.
            const std::string value = arg.substr(std::string("--max-schema-bytes=").size());
            size_t used = 0;
            long long bytes = 0;
            try
            {
                bytes = std::stoll(value, &used);
            }
            catch (const std::exception&)
            {
                used = 0;
            }
            if (used == 0 || used != value.size() || bytes <= 0)
            {
                throw std::invalid_argument("--max-schema-bytes must be a positive integer (bytes); default 32768");
            }
            options.max_schema_bytes = static_cast<size_t>(bytes);
        }
        else if (starts_with(arg, "--clock-source="))
        {
            const std::string value = arg.substr(std::string("--clock-source=").size());
            if (value != "two-way" && value != "one-way")
            {
                throw std::invalid_argument("--clock-source must be two-way or one-way");
            }
            options.clock_source = value == "two-way" ? remote_device_bridge::ClockSource::TwoWay :
                                                        remote_device_bridge::ClockSource::OneWay;
        }
        else if (starts_with(arg, "--stats-csv="))
        {
            options.stats_csv = arg.substr(std::string("--stats-csv=").size());
        }
        else if (starts_with(arg, "--plugin-root-id="))
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

    remote_device_bridge::BridgeServer server(
        options, [](const std::string& app_name, const core::PluginSessionRequirements& requirements)
        { return plugin_utils::make_openxr_plugin_session_factory(app_name)(requirements, {}); });
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
