// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mcap/recording_traits.hpp>
#include <mcap/tracker_channels.hpp>
#include <mcap/writer.hpp>
#include <schema/controller_generated.h>
#include <schema/full_body_generated.h>
#include <schema/hand_generated.h>
#include <schema/head_generated.h>
#include <schema/pedals_generated.h>
#include <schema/timestamp_generated.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using ControllerChannels = core::McapTrackerChannels<core::ControllerSnapshotRecord>;
using HandChannels = core::McapTrackerChannels<core::HandPoseRecord>;
using HeadChannels = core::McapTrackerChannels<core::HeadPoseRecord>;
using PedalChannels = core::McapTrackerChannels<core::Generic3AxisPedalOutputRecord>;
using FullBodyChannels = core::McapTrackerChannels<core::FullBodyPoseRecord>;

constexpr int kDefaultFrameCount = 1800;
constexpr int64_t kFramePeriodNs = 16'666'667;
constexpr int kFingerCurlPeriodFrames = 120;

// Per-frame position drift (meters) applied to every sample source so replayed
// poses vary over time instead of staying static; the fixture only needs
// valid/finite/changing values, so the exact magnitude is arbitrary.
constexpr float kDriftRatePerFrameM = 0.0005f;

// Plausible standing-pose heights (meters) that give the sample poses physical meaning.
constexpr float kHeadHeightM = 1.60f;
constexpr float kControllerGripHeightM = 1.10f;
constexpr float kControllerAimHeightM = 1.20f;
constexpr float kHandHeightM = 1.10f;
constexpr float kFullBodyPelvisHeightM = 0.88f;

struct PositionOffset
{
    float x;
    float y;
    float z;
};

// OpenXR hand-joint offsets from the wrist in Y-up, -Z-forward coordinates.
// The table follows core::HandJoint order and describes an open left hand;
// make_hand_sample mirrors X for the right hand.
constexpr std::array<PositionOffset, static_cast<std::size_t>(core::HandJoint_NUM_JOINTS)> kHandJointOffsets = {
    PositionOffset{ 0.000f, 0.015f, -0.035f }, // PALM
    PositionOffset{ 0.000f, 0.000f, 0.000f }, // WRIST
    PositionOffset{ 0.025f, 0.005f, -0.015f }, // THUMB_METACARPAL
    PositionOffset{ 0.035f, 0.010f, -0.030f }, // THUMB_PROXIMAL
    PositionOffset{ 0.040f, 0.012f, -0.050f }, // THUMB_DISTAL
    PositionOffset{ 0.042f, 0.013f, -0.062f }, // THUMB_TIP
    PositionOffset{ 0.018f, 0.003f, -0.055f }, // INDEX_METACARPAL
    PositionOffset{ 0.020f, 0.000f, -0.095f }, // INDEX_PROXIMAL
    PositionOffset{ 0.020f, 0.000f, -0.125f }, // INDEX_INTERMEDIATE
    PositionOffset{ 0.019f, 0.000f, -0.145f }, // INDEX_DISTAL
    PositionOffset{ 0.019f, 0.000f, -0.155f }, // INDEX_TIP
    PositionOffset{ 0.005f, 0.002f, -0.055f }, // MIDDLE_METACARPAL
    PositionOffset{ 0.005f, 0.000f, -0.100f }, // MIDDLE_PROXIMAL
    PositionOffset{ 0.005f, 0.000f, -0.135f }, // MIDDLE_INTERMEDIATE
    PositionOffset{ 0.005f, 0.000f, -0.160f }, // MIDDLE_DISTAL
    PositionOffset{ 0.005f, 0.000f, -0.175f }, // MIDDLE_TIP
    PositionOffset{ -0.008f, 0.003f, -0.055f }, // RING_METACARPAL
    PositionOffset{ -0.010f, 0.000f, -0.095f }, // RING_PROXIMAL
    PositionOffset{ -0.012f, 0.000f, -0.128f }, // RING_INTERMEDIATE
    PositionOffset{ -0.013f, 0.000f, -0.150f }, // RING_DISTAL
    PositionOffset{ -0.014f, 0.000f, -0.163f }, // RING_TIP
    PositionOffset{ -0.022f, 0.005f, -0.050f }, // LITTLE_METACARPAL
    PositionOffset{ -0.025f, 0.000f, -0.080f }, // LITTLE_PROXIMAL
    PositionOffset{ -0.027f, 0.000f, -0.103f }, // LITTLE_INTERMEDIATE
    PositionOffset{ -0.029f, 0.000f, -0.120f }, // LITTLE_DISTAL
    PositionOffset{ -0.030f, 0.000f, -0.130f }, // LITTLE_TIP
};

// Standing body-joint offsets from the pelvis in Y-up, -Z-forward coordinates.
// The table follows core::BodyJoint order.
constexpr std::array<PositionOffset, static_cast<std::size_t>(core::BodyJoint_NUM_JOINTS)> kBodyJointOffsets = {
    PositionOffset{ 0.00f, 0.00f, 0.00f }, // PELVIS
    PositionOffset{ -0.10f, -0.05f, 0.00f }, // LEFT_HIP
    PositionOffset{ 0.10f, -0.05f, 0.00f }, // RIGHT_HIP
    PositionOffset{ 0.00f, 0.12f, 0.00f }, // SPINE1
    PositionOffset{ -0.10f, -0.45f, 0.02f }, // LEFT_KNEE
    PositionOffset{ 0.10f, -0.45f, 0.02f }, // RIGHT_KNEE
    PositionOffset{ 0.00f, 0.28f, 0.00f }, // SPINE2
    PositionOffset{ -0.10f, -0.78f, 0.01f }, // LEFT_ANKLE
    PositionOffset{ 0.10f, -0.78f, 0.01f }, // RIGHT_ANKLE
    PositionOffset{ 0.00f, 0.45f, 0.00f }, // SPINE3
    PositionOffset{ -0.10f, -0.82f, -0.15f }, // LEFT_FOOT
    PositionOffset{ 0.10f, -0.82f, -0.15f }, // RIGHT_FOOT
    PositionOffset{ 0.00f, 0.62f, 0.00f }, // NECK
    PositionOffset{ -0.08f, 0.60f, 0.00f }, // LEFT_COLLAR
    PositionOffset{ 0.08f, 0.60f, 0.00f }, // RIGHT_COLLAR
    PositionOffset{ 0.00f, 0.82f, 0.00f }, // HEAD
    PositionOffset{ -0.20f, 0.57f, 0.00f }, // LEFT_SHOULDER
    PositionOffset{ 0.20f, 0.57f, 0.00f }, // RIGHT_SHOULDER
    PositionOffset{ -0.42f, 0.38f, -0.02f }, // LEFT_ELBOW
    PositionOffset{ 0.42f, 0.38f, -0.02f }, // RIGHT_ELBOW
    PositionOffset{ -0.60f, 0.20f, -0.04f }, // LEFT_WRIST
    PositionOffset{ 0.60f, 0.20f, -0.04f }, // RIGHT_WRIST
    PositionOffset{ -0.67f, 0.18f, -0.08f }, // LEFT_HAND
    PositionOffset{ 0.67f, 0.18f, -0.08f }, // RIGHT_HAND
};

// Identity orientation shared by every sample pose.
core::Quaternion identity_quaternion()
{
    return core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
}

float finger_curl_amount(int frame)
{
    const int phase = frame % kFingerCurlPeriodFrames;
    const int half_period = kFingerCurlPeriodFrames / 2;
    if (phase <= half_period)
    {
        return static_cast<float>(phase) / static_cast<float>(half_period);
    }
    return static_cast<float>(kFingerCurlPeriodFrames - phase) / static_cast<float>(half_period);
}

PositionOffset curled_hand_joint_offset(int joint, PositionOffset offset, float curl)
{
    switch (joint)
    {
    case core::HandJoint_THUMB_DISTAL:
        offset.y -= 0.010f * curl;
        offset.z += 0.010f * curl;
        break;
    case core::HandJoint_THUMB_TIP:
        offset.y -= 0.025f * curl;
        offset.z += 0.025f * curl;
        break;
    case core::HandJoint_INDEX_INTERMEDIATE:
    case core::HandJoint_MIDDLE_INTERMEDIATE:
    case core::HandJoint_RING_INTERMEDIATE:
    case core::HandJoint_LITTLE_INTERMEDIATE:
        offset.y -= 0.015f * curl;
        offset.z += 0.015f * curl;
        break;
    case core::HandJoint_INDEX_DISTAL:
    case core::HandJoint_MIDDLE_DISTAL:
    case core::HandJoint_RING_DISTAL:
    case core::HandJoint_LITTLE_DISTAL:
        offset.y -= 0.035f * curl;
        offset.z += 0.035f * curl;
        break;
    case core::HandJoint_INDEX_TIP:
    case core::HandJoint_MIDDLE_TIP:
    case core::HandJoint_RING_TIP:
    case core::HandJoint_LITTLE_TIP:
        offset.y -= 0.055f * curl;
        offset.z += 0.065f * curl;
        break;
    default:
        break;
    }
    return offset;
}

std::vector<std::string> to_strings(auto channels)
{
    std::vector<std::string> result;
    result.reserve(channels.size());
    for (std::string_view channel : channels)
    {
        result.emplace_back(channel);
    }
    return result;
}

std::shared_ptr<core::ControllerSnapshotT> make_controller_sample(bool left, int frame)
{
    const float side = left ? -1.0f : 1.0f;
    const float delta = kDriftRatePerFrameM * static_cast<float>(frame);
    auto sample = std::make_shared<core::ControllerSnapshotT>();
    sample->grip_pose = std::make_shared<core::ControllerPose>(
        core::Pose(core::Point(0.15f * side, kControllerGripHeightM, -0.10f - delta), identity_quaternion()), true);
    sample->aim_pose = std::make_shared<core::ControllerPose>(
        core::Pose(core::Point(0.20f * side, kControllerAimHeightM, -0.15f - delta), identity_quaternion()), true);
    sample->inputs = std::make_shared<core::ControllerInputState>(
        true, !left, false, left, 0.25f * side, left ? 0.40f : -0.40f, 0.55f, 0.70f);
    return sample;
}

std::shared_ptr<core::HandPoseT> make_hand_sample(bool left, int frame)
{
    const float anchor_x = left ? -0.25f : 0.25f;
    const float mirror_x = left ? 1.0f : -1.0f;
    const float delta = kDriftRatePerFrameM * static_cast<float>(frame);
    const float curl = finger_curl_amount(frame);
    auto sample = std::make_shared<core::HandPoseT>();
    sample->joints = std::make_shared<core::HandJoints>();
    for (int joint = 0; joint < core::HandJoint_NUM_JOINTS; ++joint)
    {
        const auto offset = curled_hand_joint_offset(joint, kHandJointOffsets[static_cast<std::size_t>(joint)], curl);
        const float x = anchor_x + mirror_x * offset.x;
        const float y = kHandHeightM + offset.y;
        const float z = offset.z - delta;
        const core::Point position(x, y, z);
        const core::Pose pose(position, identity_quaternion());
        sample->joints->mutable_poses()->Mutate(joint, core::HandJointPose(pose, true, 0.010f));
    }
    return sample;
}

std::shared_ptr<core::HeadPoseT> make_head_sample(int frame)
{
    // Deterministic, slowly drifting head pose at standing height.
    const float delta = kDriftRatePerFrameM * static_cast<float>(frame);
    auto sample = std::make_shared<core::HeadPoseT>();
    sample->pose = std::make_shared<core::Pose>(core::Point(0.0f, kHeadHeightM, -0.10f - delta), identity_quaternion());
    sample->is_valid = true;
    return sample;
}

std::shared_ptr<core::Generic3AxisPedalOutputT> make_pedal_sample(int frame)
{
    auto sample = std::make_shared<core::Generic3AxisPedalOutputT>();
    sample->left_pedal = 0.20f;
    sample->right_pedal = 0.80f;
    sample->rudder = (frame % 2 == 0) ? 0.15f : -0.15f;
    return sample;
}

std::shared_ptr<core::FullBodyPoseT> make_full_body_sample(int frame)
{
    const float delta = kDriftRatePerFrameM * static_cast<float>(frame);
    auto sample = std::make_shared<core::FullBodyPoseT>();
    sample->joints = std::make_shared<core::BodyJoints>();
    sample->all_joint_poses_tracked = true;
    for (int joint = 0; joint < core::BodyJoint_NUM_JOINTS; ++joint)
    {
        const auto& offset = kBodyJointOffsets[static_cast<std::size_t>(joint)];
        const float x = offset.x;
        const float y = kFullBodyPelvisHeightM + offset.y;
        const float z = offset.z - delta;
        const core::Point position(x, y, z);
        const core::Pose pose(position, identity_quaternion());
        sample->joints->mutable_joints()->Mutate(joint, core::BodyJointPose(pose, true));
    }
    return sample;
}

std::unique_ptr<mcap::McapWriter> open_writer(const std::filesystem::path& path)
{
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path());
    }

    auto writer = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions options("teleop-ros2-integration-test");
    options.compression = mcap::Compression::None;
    const auto status = writer->open(path.string(), options);
    if (!status.ok())
    {
        throw std::runtime_error("failed to open MCAP writer for " + path.string() + ": " + status.message);
    }
    return writer;
}

void write_fixture(const std::filesystem::path& output_path, int frame_count)
{
    auto writer = open_writer(output_path);
    const auto controller_names = to_strings(core::ControllerRecordingTraits::recording_channels);
    const auto hand_names = to_strings(core::HandRecordingTraits::recording_channels);
    const auto head_names = to_strings(core::HeadRecordingTraits::recording_channels);
    const auto pedal_names = to_strings(core::PedalRecordingTraits::recording_channels);
    const auto full_body_names = to_strings(core::FullBodyRecordingTraits::recording_channels);

    ControllerChannels controller_channels(*writer, "controllers", controller_names);
    HandChannels hand_channels(*writer, "hands", hand_names);
    HeadChannels head_channels(*writer, "head", head_names);
    PedalChannels pedal_channels(*writer, "pedals", pedal_names);
    FullBodyChannels full_body_channels(*writer, "full_body", full_body_names);

    for (int frame = 0; frame < frame_count; ++frame)
    {
        const int64_t time_ns = static_cast<int64_t>(frame + 1) * kFramePeriodNs;
        const core::DeviceDataTimestamp timestamp(time_ns, time_ns, time_ns);
        controller_channels.write(
            0, core::pack_record<core::ControllerSnapshotRecord>(make_controller_sample(true, frame).get(), timestamp));
        controller_channels.write(
            1, core::pack_record<core::ControllerSnapshotRecord>(make_controller_sample(false, frame).get(), timestamp));
        hand_channels.write(0, core::pack_record<core::HandPoseRecord>(make_hand_sample(true, frame).get(), timestamp));
        hand_channels.write(1, core::pack_record<core::HandPoseRecord>(make_hand_sample(false, frame).get(), timestamp));
        head_channels.write(0, core::pack_record<core::HeadPoseRecord>(make_head_sample(frame).get(), timestamp));
        pedal_channels.write(
            1, core::pack_record<core::Generic3AxisPedalOutputRecord>(make_pedal_sample(frame).get(), timestamp));
        full_body_channels.write(
            0, core::pack_record<core::FullBodyPoseRecord>(make_full_body_sample(frame).get(), timestamp));
    }

    writer->close();
}

int parse_frame_count(const char* value)
{
    const int frame_count = std::stoi(value);
    if (frame_count <= 0)
    {
        throw std::invalid_argument("frame count must be positive");
    }
    return frame_count;
}

} // namespace

int main(int argc, char** argv)
try
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " <output.mcap> [frame_count]\n";
        return 2;
    }

    const std::filesystem::path output_path(argv[1]);
    const int frame_count = argc == 3 ? parse_frame_count(argv[2]) : kDefaultFrameCount;
    write_fixture(output_path, frame_count);
    std::cout << "Wrote " << frame_count << " teleop ROS 2 replay frames to " << output_path << "\n";
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << "\n";
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error occurred\n";
    return 1;
}
