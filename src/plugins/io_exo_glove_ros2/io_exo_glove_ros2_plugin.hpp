// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_hand.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <openxr/openxr.h>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <pusherio/schema_pusher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
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
 *    6-DoF frames. See ``pack_fingertip_poses``.
 *
 *  - The same link poses injected as the OpenXR hand skeleton (``/hand/left``, ``/hand/right``,
 *    ``type: hand_tracker``) through ``plugin_utils::HandInjector``, which is what the device
 *    conformance suite records: it reads Isaac Teleop's OpenXR hand tracker, not a channel of the
 *    plugin's own. The exoskeleton's 21 skeletal links per hand are exactly the 21 OpenXR hand
 *    joints the suite judges (``PALM`` and the four non-thumb metacarpals are optional in OpenXR),
 *    and the skeleton is placed in the room by a wrist pose from the headset's own hand tracking or
 *    from a controller (``plugin_utils::WristPoseSource``), because the glove's own frame is not
 *    anchored in the room. See ``build_hand_skeleton``.
 *
 * The joint and fingertip streams are transport-only: names and values are forwarded as published.
 * The one exception is the fingertip reference frame, which ``JointSe3PoseOutput`` cannot carry per
 * pose: it is fixed by Options::tips_reference_frame (default ``base_link``, the exoskeleton's own
 * root) and a tip that arrives in any other frame is skipped with a warning rather than published in
 * a mixed set.
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
        //! Where the wrist pose that places the injected OpenXR skeleton in the room comes from:
        //! "auto" (optical hand tracking, controller as the fallback), "hand_tracking" or "controller".
        std::string wrist_source = "auto";
        //! How the glove's wrist frame is aligned with that pose: "auto" (captured from the first
        //! frame both are tracked), "identity" (the frames already agree) or an explicit "x,y,z,w"
        //! rotation.
        std::string wrist_alignment = "auto";
        //! Low-pass time constant for the tracked wrist, in milliseconds. Optical hand tracking is
        //! noisy -- and a glove in front of the headset makes it noisier -- while the glove's own
        //! articulation is not, so the anchor is smoothed to keep the injected hand still. 0 disables
        //! the filter.
        double wrist_smoothing_ms = 60.0;
    };

    //! Tensor size for the joint collection, in bytes. It is ``JointStateTracker``'s own default, so a
    //! consumer needs no custom size; the reference 42-joint payload serializes to about 2 KB.
    static constexpr std::size_t kJointsMaxFlatbufferSize = 4096;

    explicit IoExoGloveRos2Plugin(const Options& options);

private:
    //! Hands the plugin bridges, and their index into the per-hand state arrays.
    static constexpr std::size_t kSideCount = 2;

    //! How the injected skeleton is aligned with the tracked wrist pose.
    enum class Alignment
    {
        Auto, //!< Captured from the first frame where the glove and the wrist source are both tracked.
        Identity, //!< The glove's wrist frame already matches the OpenXR wrist frame.
        Explicit, //!< The rotation named by Options::wrist_alignment.
    };

    //! Latest skeleton read from the link stream, with the time it arrived.
    struct HandFrame
    {
        HandSkeleton skeleton;
        std::chrono::steady_clock::time_point stamp{};
        bool valid = false;
    };

    static std::size_t side_index(Hand hand)
    {
        return hand == Hand::Left ? 0 : 1;
    }

    //! Forwards the glove's own joint angles unchanged.
    void push_joint_state(const sensor_msgs::msg::JointState& msg);

    //! Publishes one hand's fingertip poses (see pack_fingertip_poses()).
    void push_fingertip_poses(const tf2_msgs::msg::TFMessage& msg);

    //! Reads both hands' skeletons out of one link message and keeps them for the injection timer.
    void update_hand_frames(const tf2_msgs::msg::TFMessage& msg);

    //! Pumps the wrist source and injects both hands; runs on the node's timer.
    void hand_tick();

    //! Places one hand's skeleton at the tracked wrist and pushes it to the runtime.
    void push_hand(std::size_t side, std::unique_ptr<plugin_utils::HandInjector>& injector, XrTime time, float dt_s);

    //! Runs the tracked wrist through the low-pass filter and returns the pose to place the hand at.
    XrPosef smooth_wrist(std::size_t side, const XrPosef& pose, float dt_s);

    //! Logs, once, which frame each OpenXR joint resolved to; empty entries are not published.
    void log_resolved_slots(const HandSkeleton& skeleton, std::size_t side);

    Options m_opts;

    // OpenXR session shared by all outputs; SchemaPusher is non-copyable/non-movable, so each
    // collection gets its own instance rather than a single multiplexed pusher. The session carries
    // the push-device extension the hand injection needs as well as the tensor extensions.
    std::shared_ptr<core::OpenXRSession> m_session; // OpenXR session (connects to the CloudXR runtime)
    std::unique_ptr<core::SchemaPusher> m_joints_pusher; // Glove joint angles
    std::unique_ptr<core::SchemaPusher> m_tips_left_pusher; // Left fingertip poses
    std::unique_ptr<core::SchemaPusher> m_tips_right_pusher; // Right fingertip poses

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_joints_sub;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr m_links_sub;

    //! Trackers the wrist source reads (usually a controller) and the injection timer pumps.
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;
    std::unique_ptr<plugin_utils::HandInjector> m_left_injector;
    std::unique_ptr<plugin_utils::HandInjector> m_right_injector;

    Alignment m_alignment = Alignment::Auto;
    XrQuaternionf m_alignment_explicit{ 0.0f, 0.0f, 0.0f, 1.0f };
    //! Per-hand rotation captured in Alignment::Auto; empty until both sources are tracked together.
    std::array<std::optional<XrQuaternionf>, kSideCount> m_auto_alignment;

    std::mutex m_hand_mutex;
    std::array<HandFrame, kSideCount> m_hand_frames;
    bool m_slots_logged = false;

    //! Anchor -> wrist offset frozen from the first frame that carried each hand (see
    //! build_hand_skeleton): the wrist is a fixed point on the glove, so re-deriving it every frame
    //! would slide the hand as the fingers move.
    std::array<std::optional<XrVector3f>, kSideCount> m_wrist_offsets;

    //! Filtered wrist pose per hand; @c has is false until the source first tracks it.
    struct SmoothedWrist
    {
        XrPosef pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
        bool has = false;
    };
    std::array<SmoothedWrist, kSideCount> m_smoothed_wrists;
    //! Time of the previous injection tick, for the smoothing filter's time step.
    std::chrono::steady_clock::time_point m_last_tick = std::chrono::steady_clock::now();

    rclcpp::TimerBase::SharedPtr m_hand_timer;
};

} // namespace io_exo_glove_ros2
} // namespace plugins
