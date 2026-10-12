// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/schema_pusher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace io_exo_glove_ros2
{

/*!
 * @brief Bridges the exoskeleton glove's own ROS 2 streams into Isaac Teleop's device paths.
 *
 * The glove's articulation is bridged, not a pre-retargeted command stream, so the mapping onto a
 * particular dexterous hand stays on this side and recordings are not tied to one hand:
 *
 *  - ``sensor_msgs/msg/JointState`` (default ``/io_esk/joint_data``): the glove's own DOF angles (the
 *    two hands' skeleton joints, radians), forwarded unchanged and read through the generic
 *    joint-space path (``JointStateTracker`` / ``JointStateSource``).
 *
 *  - ``tf2_msgs/msg/TFMessage`` (default ``/io_fusion/tf_exoskeleton``): every link pose, of which
 *    the five fingertips per hand are published as ``JointSe3PoseOutput`` (tensor id
 *    ``joint_se3_pose``, one collection per hand) -- the schema for a sparse, vendor-chosen set of
 *    6-DoF frames, which is what ``HandPose``'s dense OpenXR skeleton is not. See
 *    ``pack_fingertip_poses``; the remaining links are derivable from the joint stream by forward
 *    kinematics and are not duplicated here.
 *
 * Both streams are transport-only: names and values are forwarded as published. The one exception is
 * the fingertip reference frame, which ``JointSe3PoseOutput`` cannot carry per pose: it is fixed by
 * Options::tips_reference_frame (default ``base_link``, the exoskeleton's own root) and a tip that
 * arrives in any other frame is skipped with a warning rather than published in a mixed set.
 */
class IoExoGloveRos2Plugin : public rclcpp::Node
{
public:
    struct Options
    {
        std::string joints_topic = "/io_esk/joint_data";
        std::string links_topic = "/io_fusion/tf_exoskeleton";
        std::string joints_collection_id = "exo_glove_joints";
        std::string tips_left_collection_id = "exo_glove_tips_left";
        std::string tips_right_collection_id = "exo_glove_tips_right";
        //! Reference frame the fingertip poses must be expressed in; the frame a consumer reads them
        //! as. Tips published in another frame are skipped (never silently mixed or relabelled).
        std::string tips_reference_frame = "base_link";
    };

    //! Tensor size for the joint collection, in bytes. It is ``JointStateTracker``'s own default, so a
    //! consumer needs no custom size; the reference 42-joint payload serializes to about 2 KB.
    static constexpr std::size_t kJointsMaxFlatbufferSize = 4096;

    explicit IoExoGloveRos2Plugin(const Options& options);

private:
    //! Forwards the glove's own joint angles unchanged.
    void push_joint_state(const sensor_msgs::msg::JointState& msg);

    //! Publishes one hand's fingertip poses (see pack_fingertip_poses()).
    void push_fingertip_poses(const tf2_msgs::msg::TFMessage& msg);

    Options m_opts;

    // OpenXR session shared by all collections; SchemaPusher is non-copyable/non-movable, so each
    // collection gets its own instance rather than a single multiplexed pusher.
    std::shared_ptr<core::OpenXRSession> m_session; // OpenXR session (connects to the CloudXR runtime)
    std::unique_ptr<core::SchemaPusher> m_joints_pusher; // Glove joint angles
    std::unique_ptr<core::SchemaPusher> m_tips_left_pusher; // Left fingertip poses
    std::unique_ptr<core::SchemaPusher> m_tips_right_pusher; // Right fingertip poses

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_joints_sub;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr m_links_sub;
};

} // namespace io_exo_glove_ros2
} // namespace plugins
