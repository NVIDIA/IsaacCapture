// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

// Must agree with JointStateTracker::DEFAULT_MAX_FLATBUFFER_SIZE on the consumer side; sizes the
// fixed tensor buffer. 4096 bytes comfortably fits a few dozen named finger joints.
constexpr size_t kMaxFlatbufferSize = 4096;

//! ROS 2 header stamp -> nanoseconds, or 0 if unset (sec == 0 && nanosec == 0).
int64_t stamp_to_ns(const builtin_interfaces::msg::Time& stamp)
{
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

} // namespace

IoExoGloveRos2Plugin::IoExoGloveRos2Plugin(const Options& options)
    : rclcpp::Node("io_exo_glove_ros2_plugin"),
      m_opts(options),
      m_session(
          std::make_shared<core::OpenXRSession>("IoExoGloveRos2Plugin", core::SchemaPusher::get_required_extensions()))
{
    m_left_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.left_collection_id,
                                                            .max_flatbuffer_size = kMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_state",
                                                            .localized_name = "Exoskeleton Glove (Left)",
                                                            .app_name = "IoExoGloveRos2Plugin" });
    m_right_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.right_collection_id,
                                                            .max_flatbuffer_size = kMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_state",
                                                            .localized_name = "Exoskeleton Glove (Right)",
                                                            .app_name = "IoExoGloveRos2Plugin" });

    m_left_sub = create_subscription<sensor_msgs::msg::JointState>(
        m_opts.left_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_left(msg); });
    m_right_sub = create_subscription<sensor_msgs::msg::JointState>(
        m_opts.right_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_right(msg); });

    RCLCPP_INFO(get_logger(), "io_exo_glove_ros2_plugin: left '%s' -> collection '%s', right '%s' -> collection '%s'",
                m_opts.left_topic.c_str(), m_opts.left_collection_id.c_str(), m_opts.right_topic.c_str(),
                m_opts.right_collection_id.c_str());
}

void IoExoGloveRos2Plugin::push_joint_state(core::SchemaPusher& pusher,
                                            const std::string& device_id,
                                            const sensor_msgs::msg::JointState& msg)
{
    // Joint names and positions are already retargeted upstream to the target hand's URDF DOFs
    // and radians, so this is a direct name/position passthrough -- no unit conversion or
    // remapping. Rows with a missing position (name/position length mismatch) are skipped rather
    // than guessed.
    core::JointStateOutputT out;
    out.device_id = device_id;
    out.has_velocity = false;
    out.has_effort = false;
    out.ee_pose_valid = false;
    const size_t count = std::min(msg.name.size(), msg.position.size());
    out.joints.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = msg.name[i];
        joint->position = static_cast<float>(msg.position[i]);
        joint->valid = true;
        out.joints.push_back(std::move(joint));
    }

    const int64_t local_ns = core::os_monotonic_now_ns();
    const int64_t stamp_ns = stamp_to_ns(msg.header.stamp);
    // Prefer the message's own header stamp as the raw device clock when the publisher sets one;
    // otherwise fall back to the local monotonic clock (SchemaPusher's documented convention for
    // "device clock not available").
    const int64_t raw_ns = (stamp_ns != 0) ? stamp_ns : local_ns;

    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    auto offset = core::JointStateOutput::Pack(builder, &out);
    builder.Finish(offset);
    pusher.push_buffer(builder.GetBufferPointer(), builder.GetSize(), local_ns, raw_ns);
}

void IoExoGloveRos2Plugin::on_left(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    push_joint_state(*m_left_pusher, m_opts.left_collection_id, *msg);
}

void IoExoGloveRos2Plugin::on_right(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    push_joint_state(*m_right_pusher, m_opts.right_collection_id, *msg);
}

} // namespace io_exo_glove_ros2
} // namespace plugins
