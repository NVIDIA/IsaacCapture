// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
 * @file main.cpp
 * @brief j-mex AgileMaster device plugin: streams one MOXI channel over the OpenXR tensor
 *        transport as ``FullBodyPose``. Flags and setup: README.md.
 */

#include "receiver_session.hpp"

#include <jmex/device_clock.hpp>
#include <jmex/full_body_publisher.hpp>
#include <jmex/plugin_args.hpp>
#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

using namespace plugins::jmex;

namespace
{

constexpr const char* LOGGER_NAME = "isaaccapture.plugins.jmex.main";

// MOXI Player sends at 60 Hz; polling at four times that picks each frame up within about 4 ms.
constexpr int kPollHz = 240;

//! Set by SIGINT/SIGTERM so the poll loop returns and ~ReceiverSession() shuts the SDK down in
//! order, instead of the process being killed mid-stride.
std::atomic<bool> g_stop{ false };
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler stores to this");

extern "C" void on_stop_signal(int /*signum*/)
{
    // Everything a signal handler is allowed to do here: one lock-free store, no I/O.
    g_stop.store(true, std::memory_order_relaxed);
}

} // namespace

int main(int argc, char** argv)
try
{
    const auto logger = isaaccapture::Logger::get(LOGGER_NAME);
    const PluginArgs args = parse_plugin_args(argc, argv, std::getenv("JMEX_DEVICE_CLOCK"));

    const int channel = args.receiver.channel;
    const std::string collection_id = args.collection_id;

    logger->info("j-mex AgileMaster plugin (channel: {}, collection: {}, device clock: {}, SDK: {})", channel,
                 collection_id, device_clock_name(args.device_clock), moxi_sdk_version());
    logger->info("receiver on tcp {}:{}, advertising to {}, channel data from port {}",
                 args.receiver.bind_ip.empty() ? "every interface" : args.receiver.bind_ip, args.receiver.tcp_port,
                 args.receiver.broadcast_ip.empty() ? "255.255.255.255" : args.receiver.broadcast_ip,
                 args.receiver.udp_start_port + 1 + channel);

    // OpenXR first: a missing runtime is the common setup mistake, and failing on it before the
    // MOXI receiver starts means we never take the process-global TCP port just to give it back.
    FullBodyPublisher full_body_publisher(collection_id, args.device_clock);

    // Only now: the publisher's startup wait for an OpenXR system reads no stop flag, so until it
    // returns, SIGINT and SIGTERM keep their default and end the process. Nothing needs undoing
    // yet.
    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

    ReceiverSession receiver(args.receiver);

    logger->info("Waiting for MOXI Player to pair on channel {}...", channel);

    const auto poll_interval = std::chrono::nanoseconds(1000000000 / kPollHz);
    auto next_poll = std::chrono::steady_clock::now();
    int64_t published = 0;

    int exit_code = 0;
    try
    {
        while (!g_stop.load(std::memory_order_relaxed))
        {
            auto frame = receiver.poll();
            if (frame && full_body_publisher.publish(*frame, g_stop))
            {
                if (published == 0)
                {
                    logger->info("full body on collection '{}'", collection_id);
                }
                ++published;
            }

            // After a stall, re-base rather than catch up: the missed polls would all be empty.
            const auto now = std::chrono::steady_clock::now();
            next_poll += poll_interval;
            if (next_poll < now)
            {
                next_poll = now + poll_interval;
            }
            std::this_thread::sleep_until(next_poll);
        }
    }
    catch (const std::exception& e)
    {
        logger->error("{}: {}", argv[0], e.what());
        exit_code = 1;
    }

    // Log the counters on every exit: after a fatal outage they are the only record of it.
    logger->info(
        "stopping -- published {} frames, dropped {}, push failures {}, sessions re-established {}, recoveries "
        "abandoned {}",
        published, receiver.dropped_frames(), full_body_publisher.push_failures(),
        full_body_publisher.session_recoveries(), full_body_publisher.recoveries_abandoned());
    return exit_code;
}
catch (const std::exception& e)
{
    isaaccapture::Logger::get(LOGGER_NAME)->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    isaaccapture::Logger::get(LOGGER_NAME)->error("{}: Unknown error", argv[0]);
    return 1;
}
