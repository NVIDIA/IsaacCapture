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
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

using namespace plugins::jmex;

namespace
{

constexpr const char* kLoggerName = "isaaccapture.plugins.jmex.main";

// Above Player's frame rate.
constexpr int kPollHz = 240;

//! Set by SIGINT/SIGTERM: the poll loop returns and destroying the session shuts the SDK down.
std::atomic<bool> g_stop{ false };
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler stores to this");

extern "C" void on_stop_signal(int /*signum*/)
{
    // Everything a signal handler is allowed to do here: one lock-free store, no I/O.
    g_stop.store(true, std::memory_order_relaxed);
}

//! Logs the run's counters when it goes out of scope, however the run ends.
class CountersOnExit
{
public:
    CountersOnExit(const int64_t& published, const ReceiverSession& receiver, const FullBodyPublisher& publisher)
        : published_(published), receiver_(receiver), publisher_(publisher)
    {
    }

    CountersOnExit(const CountersOnExit&) = delete;
    CountersOnExit& operator=(const CountersOnExit&) = delete;

    ~CountersOnExit()
    {
        isaaccapture::Logger::get(kLoggerName)
            ->info(
                "stopping -- published {} frames, dropped {}, push failures {}, sessions re-established {}, "
                "recoveries abandoned {}",
                published_, receiver_.dropped_frames(), publisher_.push_failures(), publisher_.session_recoveries(),
                publisher_.recoveries_abandoned());
    }

private:
    const int64_t& published_;
    const ReceiverSession& receiver_;
    const FullBodyPublisher& publisher_;
};

} // namespace

int main(int argc, char** argv)
try
{
    const std::optional<PluginArgs> parsed = parse_plugin_args(argc, argv);
    if (!parsed)
    {
        std::cout << plugin_usage();
        return 0;
    }
    const PluginArgs& args = *parsed;
    const auto logger = isaaccapture::Logger::get(kLoggerName);

    const int channel = args.receiver.channel;
    const std::string collection_id = args.collection_id;

    logger->info("j-mex AgileMaster plugin (channel: {}, collection: {}, device clock: {}, SDK: {})", channel,
                 collection_id, device_clock_name(args.device_clock), moxi_sdk_version());
    logger->info("receiver on tcp {}:{}, advertising to {}, data port base {}",
                 args.receiver.bind_ip.value_or(kEveryInterface), args.receiver.tcp_port,
                 args.receiver.broadcast_ip.value_or(kSdkDefaultBroadcastIp), args.receiver.udp_start_port);

    // Before the MOXI receiver, so a missing OpenXR runtime fails the run before the receiver
    // takes its TCP port.
    FullBodyPublisher full_body_publisher(collection_id, args.device_clock);

    // The publisher's startup wait for an OpenXR system reads no stop flag: until it returns,
    // SIGINT and SIGTERM keep their default and end the process.
    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

    ReceiverSession receiver(args.receiver);
    int64_t published = 0;
    const CountersOnExit counters(published, receiver, full_body_publisher);

    logger->info("Waiting for MOXI Player to pair on channel {}...", channel);

    const auto poll_interval = std::chrono::nanoseconds(1000000000 / kPollHz);
    auto next_poll = std::chrono::steady_clock::now();
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
    return 0;
}
catch (const std::exception& e)
{
    isaaccapture::Logger::get(kLoggerName)->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    isaaccapture::Logger::get(kLoggerName)->error("{}: Unknown error", argv[0]);
    return 1;
}
