// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_plugin.hpp"

#include "io_exo_glove_ros2_pack.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/joint_state_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

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
    m_joints_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.joints_collection_id,
                                                            .max_flatbuffer_size = kJointsMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_state",
                                                            .localized_name = "Exoskeleton Glove Joints",
                                                            .app_name = "IoExoGloveRos2Plugin" });
    m_tips_left_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.tips_left_collection_id,
                                                            .max_flatbuffer_size = kTipMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_se3_pose",
                                                            .localized_name = "Exoskeleton Glove Tips (Left)",
                                                            .app_name = "IoExoGloveRos2Plugin" });
    m_tips_right_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.tips_right_collection_id,
                                                            .max_flatbuffer_size = kTipMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_se3_pose",
                                                            .localized_name = "Exoskeleton Glove Tips (Right)",
                                                            .app_name = "IoExoGloveRos2Plugin" });

    m_joints_sub = create_subscription<sensor_msgs::msg::JointState>(
        m_opts.joints_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { push_joint_state(*msg); });
    m_links_sub = create_subscription<tf2_msgs::msg::TFMessage>(m_opts.links_topic, rclcpp::SensorDataQoS(),
                                                                [this](const tf2_msgs::msg::TFMessage::SharedPtr msg)
                                                                { push_fingertip_poses(*msg); });

    RCLCPP_INFO(get_logger(),
                "io_exo_glove_ros2_plugin: joints '%s' -> '%s' (%zu B); tips '%s' -> '%s', '%s' (%zu B each, frame "
                "'%s')",
                m_opts.joints_topic.c_str(), m_opts.joints_collection_id.c_str(), kJointsMaxFlatbufferSize,
                m_opts.links_topic.c_str(), m_opts.tips_left_collection_id.c_str(),
                m_opts.tips_right_collection_id.c_str(), kTipMaxFlatbufferSize, m_opts.tips_reference_frame.c_str());
}

void IoExoGloveRos2Plugin::push_joint_state(const sensor_msgs::msg::JointState& msg)
{
    // ROS 2 specifies that a JointState's arrays are either empty or equally sized. A mismatched
    // message is dropped rather than forwarded as a partial device state: a subset of the joints looks
    // like a valid, smaller configuration downstream, which is worse than skipping the frame.
    if (msg.name.size() != msg.position.size())
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Dropping JointState: %zu names but %zu positions",
                             msg.name.size(), msg.position.size());
        return;
    }

    // The glove's own DOF names and angles are forwarded unchanged: mapping them onto a particular
    // dexterous hand is the host retargeter's job, not this bridge's.
    core::JointStateOutputT out;
    out.device_id = m_opts.joints_collection_id;
    out.has_velocity = false;
    out.has_effort = false;
    out.ee_pose_valid = false;
    out.joints.reserve(msg.name.size());
    for (size_t i = 0; i < msg.name.size(); ++i)
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

    flatbuffers::FlatBufferBuilder builder(kJointsMaxFlatbufferSize);
    auto offset = core::JointStateOutput::Pack(builder, &out);
    builder.Finish(offset);

    // SchemaPusher::push_buffer() throws when the payload exceeds the declared tensor size, and nothing
    // catches that here, so an oversized frame is dropped rather than aborting the subscription callback.
    const size_t serialized_size = builder.GetSize();
    if (serialized_size > kJointsMaxFlatbufferSize)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Dropping JointState: serialized size %zu exceeds %zu bytes", serialized_size,
                             kJointsMaxFlatbufferSize);
        return;
    }

    m_joints_pusher->push_buffer(builder.GetBufferPointer(), serialized_size, local_ns, raw_ns);
}

void IoExoGloveRos2Plugin::push_fingertip_poses(const tf2_msgs::msg::TFMessage& msg)
{
    if (msg.transforms.empty())
    {
        return;
    }

    const int64_t local_ns = core::os_monotonic_now_ns();
    // TFMessage carries no header of its own; every transform is stamped by the same producer.
    const int64_t stamp_ns = stamp_to_ns(msg.transforms.front().header.stamp);
    const int64_t raw_ns = (stamp_ns != 0) ? stamp_ns : local_ns;

    const std::pair<Hand, core::SchemaPusher*> hands[] = { { Hand::Left, m_tips_left_pusher.get() },
                                                           { Hand::Right, m_tips_right_pusher.get() } };
    for (const auto& [hand, pusher] : hands)
    {
        const TipPosePackResult packed =
            pack_fingertip_poses(msg, hand, pusher->config().collection_id, m_opts.tips_reference_frame);
        if (packed.skipped != 0)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "Skipped %zu fingertip frame(s) not in the reference frame '%s'", packed.skipped,
                                 packed.reference_frame.c_str());
        }
        if (packed.poses == 0)
        {
            continue; // Sparse stream: this hand reports no fingertip frames in that frame right now.
        }
        if (packed.buffer.size() > kTipMaxFlatbufferSize)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Dropping fingertip poses: %zu bytes exceed %zu",
                                 packed.buffer.size(), kTipMaxFlatbufferSize);
            continue;
        }

        pusher->push_buffer(packed.buffer.data(), packed.buffer.size(), local_ns, raw_ns);
    }
}

} // namespace io_exo_glove_ros2
} // namespace plugins
