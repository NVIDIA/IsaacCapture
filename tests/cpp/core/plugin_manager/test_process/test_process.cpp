// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int)
{
    stop_requested = 1;
}

std::vector<std::string> plugin_arguments(int argc, char** argv)
{
    std::vector<std::string> arguments;
    for (int index = 1; index < argc; ++index)
    {
        std::string argument = argv[index];
        if (!argument.starts_with("--plugin-root-id="))
        {
            arguments.push_back(std::move(argument));
        }
    }
    return arguments;
}

} // namespace

int main(int argc, char** argv)
{
    const std::vector<std::string> arguments = plugin_arguments(argc, argv);
    if (arguments.empty())
    {
        return 2;
    }

    if (arguments[0] == "wait")
    {
        std::signal(SIGINT, request_stop);
        while (!stop_requested)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return 0;
    }

    if (arguments.size() < 3)
    {
        return 2;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(arguments[2])));
    if (arguments[0] == "exit")
    {
        return std::stoi(arguments[1]);
    }
    if (arguments[0] == "signal")
    {
        std::raise(std::stoi(arguments[1]));
        return 3;
    }
    return 2;
}
