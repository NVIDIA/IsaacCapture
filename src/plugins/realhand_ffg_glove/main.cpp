// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_options.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace
{
std::atomic<bool> g_stop_requested{ false };
void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
        g_stop_requested.store(true, std::memory_order_relaxed);
}
constexpr auto kUsage = R"(Usage: realhand_ffg_glove_plugin [options]
  --ports=PATH,...            Verified glove serial ports
  --baudrates=RATE,...        2000000,1000000,921600,460800
  --calibration=PATH          Human-hand calibration file
  --wrist-source=SOURCE       auto, controller, hand_tracking
  --left-aim-to-wrist=POSE    px,py,pz,qx,qy,qz,qw (meters, XYZW)
  --right-aim-to-wrist=POSE   px,py,pz,qx,qy,qz,qw (meters, XYZW)
  --hand-scale=SCALE          Positive hand-size scale
  --plugin-root-id=ID         Device status root identifier
  --help                     Show this message

Environment defaults (command-line flags take precedence):
  REALHAND_FFG_GLOVE_CALIBRATION
  REALHAND_FFG_GLOVE_PORTS
  REALHAND_FFG_GLOVE_WRIST_SOURCE
)";
} // namespace

int main(int argc, char** argv)
try
{
    const std::vector<std::string> arguments(argv + 1, argv + argc);
    if (std::find(arguments.begin(), arguments.end(), "--help") != arguments.end())
    {
        std::cout << kUsage;
        return 0;
    }
    auto options = plugins::realhand_ffg_glove::parse_options(arguments);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    plugins::realhand_ffg_glove::RealHandFFGGlovePlugin plugin(
        std::move(options.ports), std::move(options.baudrates), std::move(options.root_id), std::move(options.config));
    isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.main")->info("RealHand FFG Glove plugin running");
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
    isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.main")->error("{}: {}", argv[0], error.what());
    return 1;
}
