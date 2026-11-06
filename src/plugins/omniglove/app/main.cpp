// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <omniglove/omniglove_plugin.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <memory>
#include <string>
#include <thread>

namespace
{

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "lock-free atomic bool is required for signal safety");

std::atomic<bool> stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        stop_requested.store(true, std::memory_order_relaxed);
    }
}

} // namespace

int main(int argc, char** argv)
try
{
    std::string plugin_root_id = "omniglove";
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        constexpr const char* prefix = "--plugin-root-id=";
        if (argument.rfind(prefix, 0) == 0)
        {
            plugin_root_id = argument.substr(std::char_traits<char>::length(prefix));
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.omniglove.main");
    auto plugin = std::make_unique<plugins::omniglove::OmniglovePlugin>(plugin_root_id);
    while (!stop_requested.load(std::memory_order_relaxed) && plugin->is_running())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return plugin->has_failed() ? 1 : 0;
}
catch (const std::exception& error)
{
    isaaccapture::Logger::get("isaaccapture.plugins.omniglove.main")->error("{}: {}", argv[0], error.what());
    return 1;
}
catch (...)
{
    isaaccapture::Logger::get("isaaccapture.plugins.omniglove.main")->error("{}: unknown error", argv[0]);
    return 1;
}
