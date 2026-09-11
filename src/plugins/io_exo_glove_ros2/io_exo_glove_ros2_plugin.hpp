// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/schema_pusher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

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
 * @brief Bridges a ROS 2 exoskeleton glove driver to Isaac Teleop's joint-space device path.
 *
 * Subscribes to the left/right ``sensor_msgs/msg/JointState`` topics published by the
 * exoskeleton's ROS 2 driver (finger joint angles, already retargeted upstream to a specific
 * dexterous hand's URDF joint names and units [rad]) and re-publishes each message unchanged --
 * same joint names, same positions -- as a ``core::JointStateOutput`` FlatBuffer via OpenXR
 * ``SchemaPusher``, on the generic joint-space device path (``JointStateTracker`` /
 * ``JointStateSource`` / ``JointStateRetargeter``). One tensor collection per hand so each side
 * can be tracked independently downstream.
 *
 * No unit conversion or name remapping is performed: the ROS 2 driver is expected to already
 * output the target hand's joint names and radians, so this plugin is a transport bridge only.
 */
class IoExoGloveRos2Plugin : public rclcpp::Node
{
public:
    struct Options
    {
        // std::string left_topic = "/io_teleop/joint_cmd_finger_left";
        std::string left_topic = "/io_teleop/Wuji_Hand/joint_cmd_finger_left";
        // std::string right_topic = "/io_teleop/joint_cmd_finger_right";
        std::string right_topic = "/io_teleop/Wuji_Hand/joint_cmd_finger_right";
        std::string left_collection_id = "exo_glove_left";
        std::string right_collection_id = "exo_glove_right";
    };

    explicit IoExoGloveRos2Plugin(const Options& options);

private:
    //! Shared by both callbacks: converts a JointState message to JointStateOutputT and pushes it
    //! through @p pusher, using @p device_id as JointStateOutput.device_id.
    void push_joint_state(core::SchemaPusher& pusher,
                          const std::string& device_id,
                          const sensor_msgs::msg::JointState& msg);

    void on_left(const sensor_msgs::msg::JointState::SharedPtr msg);
    void on_right(const sensor_msgs::msg::JointState::SharedPtr msg);

    Options m_opts;

    // OpenXR session shared by both hands; SchemaPusher is non-copyable/non-movable so each hand
    // gets its own instance rather than a single multiplexed pusher.
    std::shared_ptr<core::OpenXRSession> m_session; // OpenXR session (connects to the CloudXR runtime)
    std::unique_ptr<core::SchemaPusher> m_left_pusher; // Pushes the left hand's samples
    std::unique_ptr<core::SchemaPusher> m_right_pusher; // Pushes the right hand's samples

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_left_sub;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_right_sub;
};

} // namespace io_exo_glove_ros2
} // namespace plugins
