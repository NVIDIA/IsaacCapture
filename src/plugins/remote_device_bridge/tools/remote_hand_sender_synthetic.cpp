// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Stands in for a hand plugin: streams two synthetic hands over RemotePluginSession
// so the bridge loopback can be exercised without glove hardware. --schema-bytes adds
// a schema channel pushing fixed-size payloads at its own rate (bandwidth/rate load).
// --hands=0 / --wrist=0 drop the hand push channels / wrist pull so N clients can carry
// distinct device types (the runtime allows one push-device owner per hand).
//
// Its terminal output is the product (connect banner, the 1 s "[sender]" line, the stop summary), so it stays on
// std::cout like manus_hand_tracker_printer; warnings and the fatal handlers go to the logger.

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/hand_tracking_pusher.hpp>
#include <remote_plugin_session/bridge_limits.hpp>
#include <remote_plugin_session/remote_plugin_session.hpp>
#include <remote_plugin_session/websocket_transport.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "lock-free atomic bool is required for signal safety");

std::atomic<bool> g_stop_requested{ false };
constexpr double kTwoPi = 6.283185307179586;

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.remote_hand_sender_synthetic");
}

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

struct Options
{
    remote_plugin_session::RemoteAddress remote{ { "127.0.0.1", 47500 } };
    int rate_hz = 90;
    double duration_s = 0.0; // 0 = run until Ctrl-C
    size_t schema_bytes = 0; // 0 = no schema channel
    int schema_rate_hz = 180;
    bool hands = true;
    bool wrist = true;
};

bool parse_bool(const std::string& arg, const std::string& value)
{
    if (value != "0" && value != "1")
    {
        throw std::invalid_argument(arg + " expects 0 or 1");
    }
    return value == "1";
}

std::string mode_string(const Options& options)
{
    std::string mode;
    int count = 0;
    for (const auto& [on, name] : { std::pair{ options.hands, "hands" }, std::pair{ options.wrist, "wrist" },
                                    std::pair{ options.schema_bytes > 0, "schema" } })
    {
        if (on)
        {
            mode += (count++ == 0 ? "" : "+") + std::string(name);
        }
    }
    return count == 1 ? mode + "-only" : mode;
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--remote="))
        {
            options.remote = remote_plugin_session::parse_remote(arg.substr(std::string("--remote=").size()), 47500);
        }
        else if (starts_with(arg, "--rate-hz="))
        {
            options.rate_hz = std::stoi(arg.substr(std::string("--rate-hz=").size()));
        }
        else if (starts_with(arg, "--duration-s="))
        {
            options.duration_s = std::stod(arg.substr(std::string("--duration-s=").size()));
        }
        else if (starts_with(arg, "--schema-bytes="))
        {
            options.schema_bytes = std::stoul(arg.substr(std::string("--schema-bytes=").size()));
        }
        else if (starts_with(arg, "--schema-rate-hz="))
        {
            options.schema_rate_hz = std::stoi(arg.substr(std::string("--schema-rate-hz=").size()));
        }
        else if (starts_with(arg, "--hands="))
        {
            options.hands = parse_bool("--hands", arg.substr(std::string("--hands=").size()));
        }
        else if (starts_with(arg, "--wrist="))
        {
            options.wrist = parse_bool("--wrist", arg.substr(std::string("--wrist=").size()));
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: remote_hand_sender_synthetic [--remote=<host:port>|ws://<host:port>[/path]] [--rate-hz=<n>] "
                   "[--duration-s=<s>] [--schema-bytes=<n>] [--schema-rate-hz=<n>] [--hands=0|1] "
                   "[--wrist=0|1]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown argument '" + arg + "'");
        }
    }
    if (options.rate_hz <= 0 || options.schema_rate_hz <= 0)
    {
        throw std::invalid_argument("--rate-hz and --schema-rate-hz must be positive");
    }
    if (!options.hands && !options.wrist && options.schema_bytes == 0)
    {
        throw std::invalid_argument("nothing to send: enable --hands=1, --wrist=1 or --schema-bytes=<n>");
    }
    return options;
}

XrQuaternionf quat_multiply(const XrQuaternionf& a, const XrQuaternionf& b)
{
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

XrVector3f quat_rotate(const XrQuaternionf& q, const XrVector3f& v)
{
    const float t1 = 2.0f * (q.y * v.z - q.z * v.y);
    const float t2 = 2.0f * (q.z * v.x - q.x * v.z);
    const float t3 = 2.0f * (q.x * v.y - q.y * v.x);
    return { v.x + q.w * t1 + (q.y * t3 - q.z * t2), v.y + q.w * t2 + (q.z * t1 - q.x * t3),
             v.z + q.w * t3 + (q.x * t2 - q.y * t1) };
}

XrQuaternionf quat_from_axis_angle(const XrVector3f& axis, float angle)
{
    const float half = 0.5f * angle;
    const float s = std::sin(half);
    return { axis.x * s, axis.y * s, axis.z * s, std::cos(half) };
}

XrPosef compose(const XrPosef& parent, const XrPosef& local)
{
    const XrVector3f p = quat_rotate(parent.orientation, local.position);
    return { quat_multiply(parent.orientation, local.orientation),
             { parent.position.x + p.x, parent.position.y + p.y, parent.position.z + p.z } };
}

// Resting hand in the wrist frame (OpenXR: +X right of the left palm, +Y back of hand, -Z fingers).
// Each finger is a chain of segments along -Z; curl pitches every segment about +X towards the palm.
struct FingerLayout
{
    int first_joint; // metacarpal (thumb) or metacarpal joint index in XrHandJointEXT order
    int segments; // number of segments after the first joint
    XrVector3f base;
    std::array<float, 4> lengths;
    float yaw; // spread about +Y, radians
};

constexpr std::array<FingerLayout, 5> kFingers = { {
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, 3, { 0.020f, -0.005f, -0.020f }, { 0.040f, 0.032f, 0.028f, 0.0f }, 0.9f },
    { XR_HAND_JOINT_INDEX_METACARPAL_EXT, 4, { 0.018f, 0.0f, -0.020f }, { 0.060f, 0.040f, 0.025f, 0.022f }, 0.12f },
    { XR_HAND_JOINT_MIDDLE_METACARPAL_EXT, 4, { 0.005f, 0.0f, -0.020f }, { 0.062f, 0.044f, 0.028f, 0.024f }, 0.0f },
    { XR_HAND_JOINT_RING_METACARPAL_EXT, 4, { -0.010f, 0.0f, -0.020f }, { 0.058f, 0.040f, 0.026f, 0.022f }, -0.10f },
    { XR_HAND_JOINT_LITTLE_METACARPAL_EXT, 4, { -0.024f, 0.0f, -0.022f }, { 0.050f, 0.032f, 0.020f, 0.020f }, -0.22f },
} };

float joint_radius(int joint)
{
    if (joint <= XR_HAND_JOINT_WRIST_EXT)
    {
        return 0.015f;
    }
    switch (joint)
    {
    case XR_HAND_JOINT_THUMB_TIP_EXT:
    case XR_HAND_JOINT_INDEX_TIP_EXT:
    case XR_HAND_JOINT_MIDDLE_TIP_EXT:
    case XR_HAND_JOINT_RING_TIP_EXT:
    case XR_HAND_JOINT_LITTLE_TIP_EXT:
        return 0.006f;
    default:
        return 0.009f;
    }
}

void generate_hand(XrHandJointLocationEXT* joints, const XrPosef& wrist, bool is_left, float curl)
{
    const float mirror = is_left ? 1.0f : -1.0f;
    const XrPosef identity{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };

    std::array<XrPosef, XR_HAND_JOINT_COUNT_EXT> local;
    local.fill(identity);
    local[XR_HAND_JOINT_PALM_EXT].position = { 0.0f, 0.0f, -0.045f };

    for (const FingerLayout& finger : kFingers)
    {
        const bool is_thumb = finger.first_joint == XR_HAND_JOINT_THUMB_METACARPAL_EXT;
        XrPosef cursor{ quat_from_axis_angle({ 0.0f, 1.0f, 0.0f }, mirror * finger.yaw),
                        { mirror * finger.base.x, finger.base.y, finger.base.z } };
        local[finger.first_joint] = cursor;
        for (int s = 0; s < finger.segments; ++s)
        {
            // Curl is applied progressively along the chain so tips bend most.
            const float pitch = is_thumb ? 0.35f * curl : (0.55f + 0.20f * static_cast<float>(s)) * curl;
            const XrQuaternionf bend = quat_from_axis_angle({ 1.0f, 0.0f, 0.0f }, -pitch);
            cursor = compose(cursor, XrPosef{ bend, { 0.0f, 0.0f, -finger.lengths[static_cast<size_t>(s)] } });
            local[finger.first_joint + 1 + s] = cursor;
        }
    }

    for (int j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j)
    {
        joints[j].pose = compose(wrist, local[static_cast<size_t>(j)]);
        joints[j].radius = joint_radius(j);
        joints[j].locationFlags = XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                  XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT;
    }
}

} // namespace

int main(int argc, char** argv)
try
{
    const Options options = parse_args(argc, argv);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "remote_hand_sender_synthetic: connecting to " << (options.remote.websocket ? "ws://" : "")
              << options.remote.endpoint.host << ":" << options.remote.endpoint.port
              << (options.remote.websocket ? options.remote.path : "") << std::endl;
    remote_plugin_session::RemotePluginSession session(
        remote_plugin_session::connect_transport(options.remote), "remote_hand_sender_synthetic",
        core::PluginSessionRequirements{ .schema_push = options.schema_bytes > 0,
                                         .hand_tracking_push = options.hands,
                                         .wrist_tracking_pull = options.wrist });

    std::unique_ptr<core::IPluginPullChannel> pull_channel;
    std::unique_ptr<core::IWristTrackingSource> wrist_source;
    if (options.wrist)
    {
        pull_channel = session.create_pull_channel();
        wrist_source = pull_channel->create_wrist_tracking_source(core::WristTrackingSourceConfig{});
    }
    std::unique_ptr<core::HandTrackingPusher> left_pusher;
    std::unique_ptr<core::HandTrackingPusher> right_pusher;
    if (options.hands)
    {
        left_pusher =
            std::make_unique<core::HandTrackingPusher>(session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT));
        right_pusher =
            std::make_unique<core::HandTrackingPusher>(session.create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT));
    }
    std::unique_ptr<core::ISchemaPushChannel> schema_channel;
    std::vector<uint8_t> schema_payload(options.schema_bytes, 0x5A);
    if (options.schema_bytes > 0)
    {
        core::SchemaPusherConfig config;
        config.collection_id = "synthetic_tactile";
        config.max_flatbuffer_size = options.schema_bytes;
        if (options.schema_bytes > remote_plugin_session::kDefaultMaxSchemaBytes)
        {
            logger()->warn(
                "warning: --schema-bytes={} exceeds the bridge's default cap ({} bytes); expect a 'bridge error: ... "
                "channel refused' disconnect",
                options.schema_bytes, remote_plugin_session::kDefaultMaxSchemaBytes);
        }
        schema_channel = session.create_schema_push_channel(config);
    }
    std::cout << "connected; mode=" << mode_string(options) << ", tick " << options.rate_hz << " Hz";
    if (schema_channel)
    {
        std::cout << " + " << options.schema_bytes << " B schema samples at " << options.schema_rate_hz << " Hz";
    }
    std::cout << std::endl;

    const auto period = std::chrono::nanoseconds(1'000'000'000 / options.rate_hz);
    const auto schema_period = std::chrono::nanoseconds(1'000'000'000 / options.schema_rate_hz);
    const auto start = std::chrono::steady_clock::now();
    auto next_tick = start;
    auto next_schema = start;
    auto next_status = start + std::chrono::seconds(1);
    uint64_t frames_sent = 0;
    uint64_t schema_sent = 0;
    int exit_code = 0;

    XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - start).count();
        if (options.duration_s > 0.0 && elapsed_s >= options.duration_s)
        {
            break;
        }
        if (!session.connected())
        {
            logger()->error("session disconnected; exiting");
            exit_code = 1;
            break;
        }
        if (schema_channel && now >= next_schema)
        {
            std::memcpy(schema_payload.data(), &schema_sent, std::min(sizeof(schema_sent), schema_payload.size()));
            const int64_t t_ns = core::os_monotonic_now_ns();
            schema_channel->push_buffer(schema_payload.data(), schema_payload.size(), t_ns, t_ns);
            ++schema_sent;
            next_schema += schema_period;
        }
        if (now < next_tick)
        {
            std::this_thread::sleep_until(schema_channel ? std::min(next_tick, next_schema) : next_tick);
            continue;
        }

        if (pull_channel)
        {
            pull_channel->update();
        }
        const int64_t sample_time_ns = core::os_monotonic_now_ns();
        // Slow sine (4 s period) so a viewer sees the fingers open and close.
        const float curl = 0.5f + 0.5f * static_cast<float>(std::sin(elapsed_s * kTwoPi / 4.0));

        core::WristTrackingSample wrist[2];
        for (const bool is_left : { true, false })
        {
            core::WristTrackingSample& sample = wrist[is_left ? 0 : 1];
            sample = wrist_source ? wrist_source->query(is_left, sample_time_ns) : core::WristTrackingSample{};
            if (!left_pusher)
            {
                continue;
            }
            // Fallback when the bridge has no wrist pose yet: hands a little apart in front of the user.
            const XrPosef pose = sample.valid ?
                                     sample.pose :
                                     XrPosef{ { 0.0f, 0.0f, 0.0f, 1.0f }, { is_left ? -0.15f : 0.15f, 1.2f, -0.4f } };
            generate_hand(joints, pose, is_left, curl);
            (is_left ? left_pusher : right_pusher)->push(joints, sample_time_ns);
        }
        frames_sent += left_pusher ? 1 : 0;

        if (now >= next_status)
        {
            std::cout << "[sender] mode=" << mode_string(options) << " frames=" << frames_sent
                      << " schema=" << schema_sent;
            if (wrist_source)
            {
                std::cout << " wrist L(valid=" << wrist[0].valid << ",tracked=" << wrist[0].tracked
                          << ") R(valid=" << wrist[1].valid << ",tracked=" << wrist[1].tracked << ")";
            }
            else
            {
                std::cout << " wrist=off";
            }
            std::cout << " offset_ns=" << session.clock_offset_ns() << std::endl;
            next_status += std::chrono::seconds(1);
        }

        next_tick += period;
    }

    std::cout << "stopping after " << frames_sent << " frames, " << schema_sent << " schema samples" << std::endl;
    left_pusher.reset();
    right_pusher.reset();
    schema_channel.reset();
    wrist_source.reset();
    pull_channel.reset();
    return exit_code;
}
catch (const std::exception& e)
{
    logger()->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    logger()->error("{}: Unknown error occurred", argv[0]);
    return 1;
}
