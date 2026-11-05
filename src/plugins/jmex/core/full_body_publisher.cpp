// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/full_body_publisher.hpp"

#include "inc/jmex/device_clock.hpp"
#include "inc/jmex/full_body_frame.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/full_body_generated.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace plugins
{
namespace jmex
{

namespace
{

//! Names the OpenXR instance and the collection's pushing application. One constant because the
//! runtime reports them separately and two spellings of the same plugin read as two plugins.
constexpr const char* APP_NAME = "JmexPlugin";

//! How long to wait before each attempt to re-establish a lost session, in milliseconds: about
//! 23 s in total, short first, then 4 s.
constexpr int SESSION_BACKOFF_MS[] = { 500, 1000, 2000, 4000, 4000, 4000, 4000, 4000 };

//! How many times a lost session is re-established before the plugin gives up: one per wait.
constexpr int SESSION_RECOVERY_ATTEMPTS = static_cast<int>(std::size(SESSION_BACKOFF_MS));

//! A backoff is slept in slices this long, so a stop request costs at most one of them.
constexpr int BACKOFF_SLICE_MS = 50;

spdlog::logger& logger()
{
    static const std::shared_ptr<spdlog::logger> instance =
        isaaccapture::Logger::get("isaaccapture.plugins.jmex.FullBodyPublisher");
    return *instance;
}

//! The whole schedule added up, for the message that reports giving up.
int backoff_budget_ms()
{
    int total = 0;
    for (const int wait_ms : SESSION_BACKOFF_MS)
    {
        total += wait_ms;
    }
    return total;
}

//! The BackoffWait a publisher gets when none is passed in: @a total_ms, slept a slice at a time.
bool wait_unless_stopped(int total_ms, const std::atomic<bool>& stop)
{
    for (int remaining = total_ms; remaining > 0; remaining -= BACKOFF_SLICE_MS)
    {
        if (stop.load(std::memory_order_relaxed))
        {
            return false;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(remaining < BACKOFF_SLICE_MS ? remaining : BACKOFF_SLICE_MS));
    }
    return !stop.load(std::memory_order_relaxed);
}

//! An OpenXR session and the pusher on it. The pusher is declared last so it is destroyed first:
//! its collection handle lives inside the session, and once the runtime has gone, destroying the
//! two the other way round would destroy the handle against a session that no longer answers.
struct OpenXRTransport
{
    OpenXRTransport(const std::string& collection_id, bool wait_for_system)
        : session(APP_NAME, core::SchemaPusher::get_required_extensions(), wait_for_system),
          pusher(session.get_handles(),
                 core::SchemaPusherConfig{ .collection_id = collection_id,
                                           .max_flatbuffer_size = FullBodyPublisher::MAX_FLATBUFFER_SIZE,
                                           // Must match what FullBodyTracker reads.
                                           .tensor_identifier = "full_body",
                                           .localized_name = "j-mex Full Body",
                                           .app_name = APP_NAME })
    {
    }

    core::OpenXRSession session;
    core::SchemaPusher pusher;
};

FullBodyPublisher::TransportFactory openxr_transport_factory(const std::string& collection_id)
{
    return [collection_id](bool wait_for_system) -> FullBodyPublisher::PushBufferFunction
    {
        // Shared only because std::function needs a callable it can copy.
        const auto transport = std::make_shared<OpenXRTransport>(collection_id, wait_for_system);
        return [transport](const uint8_t* buffer, size_t size, int64_t sample_time_ns, int64_t raw_device_time_ns)
        { transport->pusher.push_buffer(buffer, size, sample_time_ns, raw_device_time_ns); };
    };
}

} // namespace

FullBodyPublisher::FullBodyPublisher(const std::string& collection_id, DeviceClock device_clock)
    : FullBodyPublisher(openxr_transport_factory(collection_id), device_clock)
{
}

FullBodyPublisher::FullBodyPublisher(TransportFactory create_transport, DeviceClock device_clock, BackoffWait wait)
    : create_transport_(std::move(create_transport)),
      wait_(wait ? std::move(wait) : BackoffWait(wait_unless_stopped)),
      device_clock_(device_clock)
{
    establish_session(/*wait_for_system=*/true);
}

void FullBodyPublisher::establish_session(bool wait_for_system)
{
    // Let go of the old transport before asking for a new one, so two sessions never stand at
    // once. Assigning the new one straight over it would bring the new one up first.
    push_buffer_ = nullptr;
    push_buffer_ = create_transport_(wait_for_system);
}

bool FullBodyPublisher::recover_session(const std::atomic<bool>& stop)
{
    for (int attempt = 0; attempt < SESSION_RECOVERY_ATTEMPTS; ++attempt)
    {
        if (!wait_(SESSION_BACKOFF_MS[attempt], stop))
        {
            return false;
        }

        try
        {
            establish_session(/*wait_for_system=*/false);
            ++session_recoveries_;
            logger().info("OpenXR session re-established after {} attempt(s); publishing again", attempt + 1);
            return true;
        }
        catch (const std::exception& e)
        {
            // Expected while the runtime is down, so only the first and last attempts are
            // reported rather than every one.
            if (attempt == 0 || attempt == SESSION_RECOVERY_ATTEMPTS - 1)
            {
                logger().warn(
                    "session re-establish attempt {}/{} failed: {}", attempt + 1, SESSION_RECOVERY_ATTEMPTS, e.what());
            }
        }
    }

    ++recoveries_abandoned_;
    return false;
}

bool FullBodyPublisher::publish(const FullBodyFrame& frame, const std::atomic<bool>& stop)
{
    // Empty only after a recovery that gave up with a stop already pending, when the caller's loop
    // is about to leave on the same flag.
    if (!push_buffer_)
    {
        return false;
    }

    const int64_t sample_time_ns = core::os_monotonic_now_ns();

    flatbuffers::FlatBufferBuilder builder(MAX_FLATBUFFER_SIZE);
    builder.Finish(core::CreateFullBodyPose(builder, &frame.joints, frame.all_joint_poses_tracked));

    if (builder.GetSize() > MAX_FLATBUFFER_SIZE)
    {
        logger().warn(
            "serialized FullBodyPose is {} bytes, over the {}-byte budget shared with the consumer's "
            "FullBodyTracker. Sample dropped.",
            builder.GetSize(), MAX_FLATBUFFER_SIZE);
        return false;
    }

    // push_buffer throws when the push fails, and a restarted CloudXR runtime arrives as exactly
    // that rather than as anything session-shaped. Not rate limited: every failure costs a backoff
    // before anything is retried, so this cannot produce more than a line or two a second.
    try
    {
        push_buffer_(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns,
                     raw_device_time_ns(device_clock_, sample_time_ns, frame.player_time_ns));
        return true;
    }
    catch (const std::exception& e)
    {
        ++push_failures_;
        logger().warn(
            "push failed at sequence id {}: {} -- re-establishing the OpenXR session", frame.sequence_id, e.what());

        if (!recover_session(stop))
        {
            if (stop.load(std::memory_order_relaxed))
            {
                return false; // stopping anyway; main's loop exits on the same flag
            }
            throw std::runtime_error("OpenXR session unrecoverable after " + std::to_string(SESSION_RECOVERY_ATTEMPTS) +
                                     " attempts over " + std::to_string(backoff_budget_ms() / 1000) +
                                     " s -- restart the CloudXR runtime, then restart this plugin");
        }
        // Recovered. This frame's pose is stale by a backoff now, so it is not re-pushed; the next
        // one Player delivers goes out on the new session.
        return false;
    }
}

} // namespace jmex
} // namespace plugins
