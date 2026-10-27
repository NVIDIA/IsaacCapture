// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

// Counts SIGINTs; only the signal handler writes it.
volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int)
{
    stop_requested = stop_requested + 1;
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

    // finalize <marker> <ms>: write <marker>.ready once SIGINT is handled; after SIGINT,
    // keep working for <ms>, then write <marker>. A second SIGINT abandons the work, as a
    // second Ctrl-C does in many plugins.
    if (arguments[0] == "finalize")
    {
        std::signal(SIGINT, request_stop);
        std::ofstream(arguments[1] + ".ready") << "ready\n";
        while (!stop_requested)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const auto done = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::stoi(arguments[2]));
        while (std::chrono::steady_clock::now() < done)
        {
            if (stop_requested > 1)
            {
                return 4;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::ofstream(arguments[1]) << "finalized\n";
        return 0;
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
