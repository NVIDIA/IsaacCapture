// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
 * @brief Mock tactile glove: pushes a synthetic TactileShape and TactileData stream (internal only).
 *
 * Stands in for a tactile plugin. The shape goes to "tactile_shape/<side>" at start and every
 * --shape-period seconds; frames go to "tactile/<side>" at --rate Hz. TactileShapeTracker and
 * TactileTracker (or TactileShapeSource / TactileSource) read them with the same collection ids.
 *
 * Usage: tactile_pusher [--side=right] [--taxels=64] [--rate=60] [--seconds=0] [--shape-period=1]
 * (--seconds=0 runs until interrupted).
 */

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/tactile_generated.h>
#include <schema/tactile_shape_generated.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

// Must not exceed the readers' max_flatbuffer_size (the trackers.toml defaults).
constexpr size_t kDataMaxSize = 16384;
constexpr size_t kShapeMaxSize = 16384;

constexpr uint32_t kNumJoints = 26; // XR_HAND_JOINT_COUNT_EXT
constexpr uint32_t kRootParent = 0x7FFFFFFF; // XR_HAND_JOINT_MAX_ENUM_EXT
constexpr uint32_t kWrist = 1;
constexpr uint32_t kShapeVersion = 1;
constexpr double kPi = 3.14159265358979323846;

// First joint (metacarpal) of each finger in XrHandJointEXT order; the thumb has no
// intermediate joint, so its chain is one shorter.
constexpr std::array<uint32_t, 5> kFingerFirstJoint = { 2, 6, 11, 16, 21 };
constexpr std::array<uint32_t, 5> kFingerTip = { 5, 10, 15, 20, 25 };

struct Options
{
    std::string side = "right";
    uint32_t taxels = 64;
    double rate_hz = 60.0;
    double seconds = 0.0;
    double shape_period_s = 1.0;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto value = [&arg](const char* prefix) -> const char*
        { return arg.rfind(prefix, 0) == 0 ? arg.c_str() + std::char_traits<char>::length(prefix) : nullptr; };
        if (const char* v = value("--side="))
            options.side = v;
        else if (const char* v = value("--taxels="))
            options.taxels = static_cast<uint32_t>(std::stoul(v));
        else if (const char* v = value("--rate="))
            options.rate_hz = std::stod(v);
        else if (const char* v = value("--seconds="))
            options.seconds = std::stod(v);
        else if (const char* v = value("--shape-period="))
            options.shape_period_s = std::stod(v);
        else
            throw std::invalid_argument("unknown argument: " + arg);
    }
    if (options.taxels == 0 || options.rate_hz <= 0.0 || options.shape_period_s <= 0.0)
    {
        throw std::invalid_argument("--taxels, --rate and --shape-period must be positive");
    }
    return options;
}

int64_t monotonic_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A flat hand: fingers along +y, spread along x; every taxel sits on a fingertip pad and is
// skinned 70/30 to that finger's distal joint and tip.
core::TactileShapeT make_shape(const Options& options)
{
    core::TactileShapeT shape;
    shape.joint_count = kNumJoints;
    shape.joint_bind_poses.resize(kNumJoints, core::Pose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 1)));
    shape.joint_parents.assign(kNumJoints, kWrist);
    shape.joint_parents[kWrist] = kRootParent;
    shape.joint_bind_poses[0] = core::Pose(core::Point(0, 0.05f, 0), core::Quaternion(0, 0, 0, 1)); // palm
    for (size_t f = 0; f < kFingerFirstJoint.size(); ++f)
    {
        for (uint32_t j = kFingerFirstJoint[f]; j <= kFingerTip[f]; ++j)
        {
            const float x = -0.04f + 0.02f * static_cast<float>(f);
            const float y = 0.03f * static_cast<float>(j - kFingerFirstJoint[f] + 1);
            shape.joint_bind_poses[j] = core::Pose(core::Point(x, y, 0), core::Quaternion(0, 0, 0, 1));
            if (j > kFingerFirstJoint[f])
            {
                shape.joint_parents[j] = j - 1;
            }
        }
    }

    for (uint32_t i = 0; i < options.taxels; ++i)
    {
        const size_t finger = i % kFingerTip.size();
        const uint32_t tip = kFingerTip[finger];
        const uint32_t distal = tip - 1;
        const core::Point& tip_position = shape.joint_bind_poses[tip].position();
        const float offset = 0.002f * static_cast<float>(i / kFingerTip.size());

        core::TaxelJointIndices indices;
        core::TaxelJointWeights weights;
        indices.mutable_joints()->Mutate(0, distal);
        indices.mutable_joints()->Mutate(1, tip);
        indices.mutable_joints()->Mutate(2, distal);
        indices.mutable_joints()->Mutate(3, distal);
        weights.mutable_weights()->Mutate(0, 0.7f);
        weights.mutable_weights()->Mutate(1, 0.3f);

        shape.taxel_positions.emplace_back(tip_position.x(), tip_position.y() - offset, -0.005f);
        shape.taxel_normals.emplace_back(0.0f, 0.0f, -1.0f);
        shape.taxel_tangents.emplace_back(1.0f, 0.0f, 0.0f);
        shape.taxel_joint_indices.push_back(indices);
        shape.taxel_joint_weights.push_back(weights);
    }
    shape.taxel_count = options.taxels;
    shape.device_id = "mock_tactile_" + options.side;
    shape.shape_version = kShapeVersion;
    return shape;
}

// Normal force in [0, 2] N with a travelling phase, plus a small shear along x.
core::TactileDataT make_frame(const Options& options, double t)
{
    core::TactileDataT frame;
    frame.taxel_force.reserve(options.taxels);
    for (uint32_t i = 0; i < options.taxels; ++i)
    {
        const double phase = kPi * t + 0.1 * i; // 0.5 Hz
        frame.taxel_force.emplace_back(
            static_cast<float>(0.1 * std::cos(phase)), 0.0f, static_cast<float>(1.0 + std::sin(phase)));
    }
    frame.taxel_count = options.taxels;
    frame.device_id = "mock_tactile_" + options.side;
    frame.shape_version = kShapeVersion;
    return frame;
}

template <typename TableT>
void push(core::SchemaPusher& pusher, const typename TableT::NativeTableType& native, int64_t now_ns)
{
    flatbuffers::FlatBufferBuilder builder(pusher.config().max_flatbuffer_size);
    builder.Finish(TableT::Pack(builder, &native));
    if (builder.GetSize() > pusher.config().max_flatbuffer_size)
    {
        throw std::runtime_error("encoded " + std::string(TableT::GetFullyQualifiedName()) + " is " +
                                 std::to_string(builder.GetSize()) + " bytes, over the collection size " +
                                 std::to_string(pusher.config().max_flatbuffer_size) + "; use fewer --taxels");
    }
    pusher.push_buffer(builder.GetBufferPointer(), builder.GetSize(), now_ns, now_ns);
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = parse_options(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << argv[0] << ": " << e.what() << "\n"
                  << "usage: " << argv[0]
                  << " [--side=right] [--taxels=64] [--rate=60] [--seconds=0] [--shape-period=1]\n";
        return 2;
    }

    auto logger = isaaccapture::Logger::get("isaaccapture.examples.TactilePusher");
    try
    {
        auto oxr_session =
            std::make_shared<core::OpenXRSession>("TactilePusher", core::SchemaPusher::get_required_extensions());
        const auto handles = oxr_session->get_handles();

        const std::string data_collection = "tactile/" + options.side;
        const std::string shape_collection = "tactile_shape/" + options.side;
        core::SchemaPusher data_pusher(handles, core::SchemaPusherConfig{ .collection_id = data_collection,
                                                                          .max_flatbuffer_size = kDataMaxSize,
                                                                          .tensor_identifier = "tactile",
                                                                          .localized_name = "Mock Tactile Data",
                                                                          .app_name = "TactilePusher" });
        core::SchemaPusher shape_pusher(handles, core::SchemaPusherConfig{ .collection_id = shape_collection,
                                                                           .max_flatbuffer_size = kShapeMaxSize,
                                                                           .tensor_identifier = "tactile_shape",
                                                                           .localized_name = "Mock Tactile Shape",
                                                                           .app_name = "TactilePusher" });

        const core::TactileShapeT shape = make_shape(options);
        logger->info("Pushing {} taxels at {} Hz to '{}', shape v{} to '{}' every {} s", options.taxels,
                     options.rate_hz, data_collection, kShapeVersion, shape_collection, options.shape_period_s);

        const auto period = std::chrono::duration<double>(1.0 / options.rate_hz);
        const int64_t start_ns = monotonic_now_ns();
        int64_t next_shape_ns = start_ns;
        int64_t next_report_ns = start_ns;
        uint64_t frames = 0;
        uint64_t shapes = 0;
        auto next_tick = std::chrono::steady_clock::now();
        while (true)
        {
            const int64_t now_ns = monotonic_now_ns();
            const double t = static_cast<double>(now_ns - start_ns) * 1e-9;
            if (options.seconds > 0.0 && t >= options.seconds)
            {
                break;
            }
            if (now_ns >= next_shape_ns)
            {
                push<core::TactileShape>(shape_pusher, shape, now_ns);
                ++shapes;
                next_shape_ns += static_cast<int64_t>(options.shape_period_s * 1e9);
            }
            push<core::TactileData>(data_pusher, make_frame(options, t), now_ns);
            ++frames;
            if (now_ns >= next_report_ns)
            {
                logger->info("t={:.1f} s: {} frames, {} shapes pushed", t, frames, shapes);
                next_report_ns += 1'000'000'000;
            }
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }
        logger->info("Done: {} frames, {} shapes pushed", frames, shapes);
        return 0;
    }
    catch (const std::exception& e)
    {
        logger->error("{}", e.what());
        return 1;
    }
}
