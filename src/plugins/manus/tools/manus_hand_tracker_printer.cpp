// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "manus_hand_visualizer.hpp"

#include <log_bridge/logger.hpp>
#include <manus/manus_hand_tracking_plugin.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

// This tool is a standalone diagnostic CLI whose whole product is joint data
// on the terminal (docs/source/device/manus.rst). It therefore keeps
// std::cout, which the repo root AGENTS.md reserves for exactly this: operator
// banners and progress lines a log file would ruin. Routing it through a
// logger made it silent in the one situation it is most often run in -- a
// shell that inherited ISAACCAPTURE_LOG_SOCKET from a session leader, where
// local_sinks() returns a forwarding sink and no console sink at all, so every
// line including "Waiting for gloves..." went to the leader's log file and the
// tool looked hung. Diagnostics -- the visualizer failure, the fatal handlers
// -- stay on the logger.

int main(int argc, char** argv)
try
{
    (void)argc;
    (void)argv;

    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.manus_hand_tracker_printer");
    std::cout << "[Manus] Initializing Manus Tracker..." << std::endl;

    plugins::manus::ManusPluginConfig config;
    config.app_name = "ManusHandPrinter";
    auto& tracker = plugins::manus::ManusTracker::instance(config);

    // Start Vulkan visualizer in a background thread.
    // If X11 or Vulkan is unavailable the thread exits cleanly and printing
    // continues without the window.
    // std::jthread automatically requests stop and joins on destruction,
    // preventing the thread from outliving the tracker singleton.
    std::jthread vis_thread(
        [&tracker, logger](std::stop_token st)
        {
            try
            {
                plugins::manus::HandVisualizer vis;
                vis.run(tracker, std::move(st));
            }
            catch (const std::exception& e)
            {
                logger->warn("Visualizer failed: {} — running without visualizer", e.what());
            }
        });

    std::cout << "[Manus] Press Ctrl+C to stop. Printing joint data..." << std::endl;

    int frame = 0;
    bool waiting_printed = false;
    while (true)
    {
        // Get glove data from Manus SDK
        auto left_nodes = tracker.get_left_hand_nodes();
        auto right_nodes = tracker.get_right_hand_nodes();

        if (left_nodes.empty() && right_nodes.empty())
        {
            if (!waiting_printed)
            {
                std::cout << "[Manus] Waiting for gloves..." << std::endl;
                waiting_printed = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        waiting_printed = false;

        std::cout << "\n[Manus] === Frame " << frame << " ===" << std::endl;

        // Helper lambda to print hand data
        auto print_hand = [](const std::string& side, const std::vector<SkeletonNode>& nodes)
        {
            if (nodes.empty())
            {
                return;
            }

            std::cout << "[Manus] " << side << " hand (" << nodes.size() << " joints):" << std::endl;

            for (size_t i = 0; i < std::min(nodes.size(), static_cast<size_t>(5)); ++i)
            {
                const auto& pos = nodes[i].transform.position;
                const auto& ori = nodes[i].transform.rotation;

                std::cout << "[Manus]   Joint " << i << ": "
                          << "pos=[" << std::fixed << std::setprecision(3) << pos.x << ", " << pos.y << ", " << pos.z
                          << "] "
                          << "ori=[" << ori.x << ", " << ori.y << ", " << ori.z << ", " << ori.w << "]" << std::endl;
            }

            if (nodes.size() > 5)
            {
                std::cout << "[Manus]   ... (" << (nodes.size() - 5) << " more joints)" << std::endl;
            }
        };

        print_hand("left", left_nodes);
        print_hand("right", right_nodes);

        std::cout << std::flush;

        frame++;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
catch (const std::exception& e)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.manus_hand_tracker_printer");
    logger->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.manus_hand_tracker_printer");
    logger->error("{}: Unknown error occurred", argv[0]);
    return 1;
}
