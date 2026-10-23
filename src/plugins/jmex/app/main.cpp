// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
 * @file main.cpp
 * @brief j-mex AgileMaster device plugin: streams one MOXI channel over the OpenXR tensor
 *        transport as ``FullBodyPose``. Flags and setup: README.md.
 */

#include <jmex/device_clock.hpp>
#include <jmex/full_body_publisher.hpp>
#include <jmex/receiver_session.hpp>
#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
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

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

struct PluginArgs
{
    ReceiverConfig receiver;
    std::string collection_id = "jmex_full_body";
    DeviceClock device_clock = DeviceClock::DEVICE;
};

//! Parse a flag's value as a whole number, or say which flag rejected what.
int parse_port(const std::string& flag, const std::string& value)
{
    try
    {
        std::size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size())
        {
            throw std::invalid_argument("trailing characters after the number");
        }
        return parsed;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(flag + " takes a number, got '" + value + "'");
    }
}

//! Parse named arguments; the last occurrence of a flag wins. Do not add positional arguments: the
//! PluginManager injects ``--plugin-root-id=<id>`` ahead of plugin.yaml's and PluginConfig's.
PluginArgs parse_args(int argc, char** argv)
{
    PluginArgs args;

    // The environment first, so that a --device-clock= below overrides it.
    const char* const device_clock_env = std::getenv("JMEX_DEVICE_CLOCK");
    args.device_clock = choose_device_clock(nullptr, device_clock_env);

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--channel="))
        {
            // Reject partially parsed values; otherwise "2.9" silently selects channel 2, making a
            // channel mismatch look like Player never connected.
            args.receiver.channel = parse_port("--channel=", arg.substr(std::string("--channel=").size()));
        }
        else if (starts_with(arg, "--tcp-port="))
        {
            args.receiver.tcp_port = parse_port("--tcp-port=", arg.substr(std::string("--tcp-port=").size()));
        }
        else if (starts_with(arg, "--udp-start-port="))
        {
            args.receiver.udp_start_port =
                parse_port("--udp-start-port=", arg.substr(std::string("--udp-start-port=").size()));
        }
        else if (starts_with(arg, "--bind-ip="))
        {
            args.receiver.bind_ip = arg.substr(std::string("--bind-ip=").size());
        }
        else if (starts_with(arg, "--broadcast-ip="))
        {
            args.receiver.broadcast_ip = arg.substr(std::string("--broadcast-ip=").size());
        }
        else if (starts_with(arg, "--collection-id="))
        {
            args.collection_id = arg.substr(std::string("--collection-id=").size());
        }
        else if (starts_with(arg, "--device-clock="))
        {
            args.device_clock =
                choose_device_clock(arg.substr(std::string("--device-clock=").size()).c_str(), device_clock_env);
        }
        else if (starts_with(arg, "--plugin-root-id="))
        {
            // Injected by the PluginManager; not used by this plugin.
        }
        else
        {
            isaaccapture::Logger::get(LOGGER_NAME)->warn("ignoring unknown argument '{}'", arg);
        }
    }

    return args;
}

} // namespace

int main(int argc, char** argv)
try
{
    const auto logger = isaaccapture::Logger::get(LOGGER_NAME);
    const PluginArgs args = parse_args(argc, argv);

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
