// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_plugin.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

std::atomic<bool> g_stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

std::vector<std::string> split_csv(const std::string& value)
{
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (!item.empty())
        {
            result.push_back(item);
        }
    }
    return result;
}

std::vector<int> parse_baudrates(const std::string& value)
{
    std::vector<int> result;
    for (const auto& item : split_csv(value))
    {
        result.push_back(std::stoi(item));
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
try
{
    std::vector<std::string> ports;
    std::vector<int> baudrates;
    std::string plugin_root_id = "realhand_ffg_glove";
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument.starts_with("--ports="))
        {
            ports = split_csv(argument.substr(8));
        }
        else if (argument.starts_with("--baudrates="))
        {
            baudrates = parse_baudrates(argument.substr(12));
        }
        else if (argument.starts_with("--plugin-root-id="))
        {
            plugin_root_id = argument.substr(17);
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    plugins::realhand_ffg_glove::RealHandFFGGlovePlugin plugin(std::move(ports), std::move(baudrates), plugin_root_id);
    std::cout << "RealHand FFG Glove plugin running. Press Ctrl+C to stop." << std::endl;

    constexpr auto frame_period = std::chrono::nanoseconds(1'000'000'000 / 90);
    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        const auto frame_start = std::chrono::steady_clock::now();
        plugin.update();
        std::this_thread::sleep_until(frame_start + frame_period);
    }
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << argv[0] << ": " << error.what() << std::endl;
    return 1;
}
