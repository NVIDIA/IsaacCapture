// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_plugin.hpp"

#include "io_exo_glove_ros2_pack.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! Injection rate. The scripted conformance take needs 30 Hz or more and the glove streams at
//! 120 Hz, so the timer only has to be comfortably above the floor; a frame that arrives between
//! ticks is injected on the next one.
constexpr auto kHandFramePeriod = std::chrono::milliseconds(8);

//! Data older than this counts as "hand absent": the injector is dropped so the runtime reports
//! ``isActive = false`` instead of a frozen pose (the convention WujiGlovePlugin uses).
constexpr auto kHandStaleThreshold = std::chrono::milliseconds(200);

//! ROS 2 header stamp -> nanoseconds, or 0 if unset (sec == 0 && nanosec == 0).
int64_t stamp_to_ns(const builtin_interfaces::msg::Time& stamp)
{
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

plugin_utils::WristSourceMode parse_wrist_source(const std::string& value)
{
    if (value == "auto")
    {
        return plugin_utils::WristSourceMode::Auto;
    }
    if (value == "hand_tracking")
    {
        return plugin_utils::WristSourceMode::HandTracking;
    }
    if (value == "controller")
    {
        return plugin_utils::WristSourceMode::Controller;
    }
    throw std::runtime_error("wrist_source must be 'auto', 'hand_tracking' or 'controller', not '" + value + "'");
}

const char* wrist_source_name(plugin_utils::WristSourceMode mode)
{
    switch (mode)
    {
    case plugin_utils::WristSourceMode::HandTracking:
        return "hand_tracking";
    case plugin_utils::WristSourceMode::Controller:
        return "controller";
    case plugin_utils::WristSourceMode::Auto:
    default:
        return "auto";
    }
}

//! Parses the "x,y,z,w" rotation spelling; nullopt when the text is not four numbers.
std::optional<XrQuaternionf> parse_quaternion(const std::string& text)
{
    XrQuaternionf rotation{};
    int parsed = 0;
    if (std::sscanf(text.c_str(), "%f,%f,%f,%f%n", &rotation.x, &rotation.y, &rotation.z, &rotation.w, &parsed) != 4 ||
        parsed != static_cast<int>(text.size()))
    {
        return std::nullopt;
    }
    // A rotation has to be usable as one: normalise rather than reject a slightly-off entry.
    const float norm =
        std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y + rotation.z * rotation.z + rotation.w * rotation.w);
    if (!(norm > 0.0f))
    {
        return std::nullopt;
    }
    rotation.x /= norm;
    rotation.y /= norm;
    rotation.z /= norm;
    rotation.w /= norm;
    return rotation;
}

const char* hand_name(std::size_t side)
{
    return side == 0 ? "left" : "right";
}

//! Name of one OpenXR hand joint, for the startup mapping log.
const char* openxr_joint_name(XrHandJointEXT joint)
{
    switch (joint)
    {
    case XR_HAND_JOINT_WRIST_EXT:
        return "WRIST";
    case XR_HAND_JOINT_THUMB_METACARPAL_EXT:
        return "THUMB_METACARPAL";
    case XR_HAND_JOINT_THUMB_PROXIMAL_EXT:
        return "THUMB_PROXIMAL";
    case XR_HAND_JOINT_THUMB_DISTAL_EXT:
        return "THUMB_DISTAL";
    case XR_HAND_JOINT_THUMB_TIP_EXT:
        return "THUMB_TIP";
    case XR_HAND_JOINT_INDEX_PROXIMAL_EXT:
        return "INDEX_PROXIMAL";
    case XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT:
        return "INDEX_INTERMEDIATE";
    case XR_HAND_JOINT_INDEX_DISTAL_EXT:
        return "INDEX_DISTAL";
    case XR_HAND_JOINT_INDEX_TIP_EXT:
        return "INDEX_TIP";
    case XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT:
        return "MIDDLE_PROXIMAL";
    case XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT:
        return "MIDDLE_INTERMEDIATE";
    case XR_HAND_JOINT_MIDDLE_DISTAL_EXT:
        return "MIDDLE_DISTAL";
    case XR_HAND_JOINT_MIDDLE_TIP_EXT:
        return "MIDDLE_TIP";
    case XR_HAND_JOINT_RING_PROXIMAL_EXT:
        return "RING_PROXIMAL";
    case XR_HAND_JOINT_RING_INTERMEDIATE_EXT:
        return "RING_INTERMEDIATE";
    case XR_HAND_JOINT_RING_DISTAL_EXT:
        return "RING_DISTAL";
    case XR_HAND_JOINT_RING_TIP_EXT:
        return "RING_TIP";
    case XR_HAND_JOINT_LITTLE_PROXIMAL_EXT:
        return "LITTLE_PROXIMAL";
    case XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT:
        return "LITTLE_INTERMEDIATE";
    case XR_HAND_JOINT_LITTLE_DISTAL_EXT:
        return "LITTLE_DISTAL";
    case XR_HAND_JOINT_LITTLE_TIP_EXT:
        return "LITTLE_TIP";
    default:
        return "?";
    }
}

} // namespace

IoExoGloveRos2Plugin::IoExoGloveRos2Plugin(const Options& options)
    : rclcpp::Node("io_exo_glove_ros2_plugin"), m_opts(options)
{
    // The wrist source's trackers and extensions have to be known before the session exists, since
    // the session is what the injection and the queries run on.
    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = parse_wrist_source(m_opts.wrist_source);
    const plugin_utils::WristPoseSource::Requirements wrist_requirements =
        plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);
    const std::vector<std::shared_ptr<core::ITracker>> trackers = wrist_requirements.trackers;

    if (m_opts.wrist_alignment == "auto")
    {
        m_alignment = Alignment::Auto;
    }
    else if (m_opts.wrist_alignment == "identity")
    {
        m_alignment = Alignment::Identity;
    }
    else
    {
        const std::optional<XrQuaternionf> explicit_alignment = parse_quaternion(m_opts.wrist_alignment);
        if (!explicit_alignment.has_value())
        {
            throw std::runtime_error("wrist_alignment must be 'auto', 'identity' or 'x,y,z,w', not '" +
                                     m_opts.wrist_alignment + "'");
        }
        m_alignment = Alignment::Explicit;
        m_alignment_explicit = *explicit_alignment;
    }

    std::vector<std::string> extensions = core::SchemaPusher::get_required_extensions();
    extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    const std::vector<std::string> tracker_extensions = core::DeviceIOSession::get_required_extensions(trackers);
    extensions.insert(extensions.end(), tracker_extensions.begin(), tracker_extensions.end());
    for (const char* extension : wrist_requirements.extensions)
    {
        extensions.emplace_back(extension);
    }

    m_session = std::make_shared<core::OpenXRSession>("IoExoGloveRos2Plugin", extensions);
    const core::OpenXRSessionHandles handles = m_session->get_handles();
    m_time_converter.emplace(handles);
    if (!trackers.empty())
    {
        m_deviceio_session = core::DeviceIOSession::run(trackers, handles);
    }
    m_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
        wrist_config, handles, m_deviceio_session.get(), wrist_requirements.controller_tracker);

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
                                                                {
                                                                    push_fingertip_poses(*msg);
                                                                    update_hand_frames(*msg);
                                                                });

    m_hand_timer = create_wall_timer(kHandFramePeriod, [this] { hand_tick(); });

    RCLCPP_INFO(get_logger(),
                "io_exo_glove_ros2_plugin: joints '%s' -> '%s' (%zu B); tips '%s' -> '%s', '%s' (%zu B each, frame "
                "'%s'); hands -> XR_HAND_{LEFT,RIGHT}_EXT (wrist '%s', alignment '%s')",
                m_opts.joints_topic.c_str(), m_opts.joints_collection_id.c_str(), kJointsMaxFlatbufferSize,
                m_opts.links_topic.c_str(), m_opts.tips_left_collection_id.c_str(),
                m_opts.tips_right_collection_id.c_str(), kTipMaxFlatbufferSize, m_opts.tips_reference_frame.c_str(),
                wrist_source_name(wrist_config.mode), m_opts.wrist_alignment.c_str());
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

void IoExoGloveRos2Plugin::update_hand_frames(const tf2_msgs::msg::TFMessage& msg)
{
    if (msg.transforms.empty())
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    std::array<HandSkeleton, kSideCount> to_log;
    bool log_slots = false;
    {
        std::lock_guard<std::mutex> lock(m_hand_mutex);
        for (const Hand hand : { Hand::Left, Hand::Right })
        {
            HandFrame& frame = m_hand_frames[side_index(hand)];
            frame.skeleton = build_hand_skeleton(msg, hand);
            frame.stamp = now;
            frame.valid = frame.skeleton.tracked();
        }
        if (!m_slots_logged && (m_hand_frames[0].valid || m_hand_frames[1].valid))
        {
            // Once, on the first frame that carries a hand: what the injection actually bound to, so a
            // rig whose link names differ is diagnosable from the plugin log alone.
            m_slots_logged = true;
            log_slots = true;
            to_log[0] = m_hand_frames[0].skeleton;
            to_log[1] = m_hand_frames[1].skeleton;
        }
    }

    if (log_slots)
    {
        log_resolved_slots(to_log[0], 0);
        log_resolved_slots(to_log[1], 1);
    }
}

void IoExoGloveRos2Plugin::log_resolved_slots(const HandSkeleton& skeleton, std::size_t side)
{
    if (!skeleton.tracked())
    {
        RCLCPP_WARN(get_logger(), "hand %s: no wrist frame in '%s', nothing is injected for it", hand_name(side),
                    m_opts.links_topic.c_str());
        return;
    }

    const auto slots = make_hand_joint_slots(side == 0 ? Hand::Left : Hand::Right);
    std::string resolved;
    std::string missing;
    for (std::size_t i = 0; i < kHandJointSlotCount; ++i)
    {
        const char* joint = openxr_joint_name(slots[i].joint);
        if (skeleton.resolved[i].empty())
        {
            missing += missing.empty() ? joint : (", " + std::string(joint));
            continue;
        }
        resolved += (resolved.empty() ? "" : ", ") + std::string(joint) + " <- " + skeleton.resolved[i];
    }

    RCLCPP_INFO(get_logger(), "hand %s: %zu/%zu OpenXR joints injected, wrist frame '%s'", hand_name(side),
                skeleton.filled, kHandJointSlotCount, skeleton.wrist_frame.c_str());
    RCLCPP_INFO(get_logger(), "hand %s: %s", hand_name(side), resolved.c_str());
    if (!missing.empty())
    {
        RCLCPP_WARN(get_logger(), "hand %s: no frame for %s (optional in OpenXR, left untracked)", hand_name(side),
                    missing.c_str());
    }
}

void IoExoGloveRos2Plugin::hand_tick()
{
    if (m_deviceio_session != nullptr)
    {
        try
        {
            m_deviceio_session->update();
        }
        catch (const std::exception& e)
        {
            // Losing the tracker session means the wrist pose is gone; drop the injectors so the
            // runtime reports the hands as inactive rather than keeping the last pose.
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "wrist tracker update failed: %s", e.what());
            m_left_injector.reset();
            m_right_injector.reset();
            return;
        }
    }
    if (!m_time_converter.has_value())
    {
        return;
    }

    const XrTime time = m_time_converter->os_monotonic_now();
    push_hand(0, m_left_injector, time);
    push_hand(1, m_right_injector, time);
}

void IoExoGloveRos2Plugin::push_hand(std::size_t side, std::unique_ptr<plugin_utils::HandInjector>& injector, XrTime time)
{
    HandFrame frame;
    {
        std::lock_guard<std::mutex> lock(m_hand_mutex);
        frame = m_hand_frames[side];
    }

    const bool fresh = frame.valid && (std::chrono::steady_clock::now() - frame.stamp) < kHandStaleThreshold;
    if (!fresh)
    {
        injector.reset();
        return;
    }
    if (injector == nullptr)
    {
        const core::OpenXRSessionHandles handles = m_session->get_handles();
        injector = std::make_unique<plugin_utils::HandInjector>(
            handles.instance, handles.session, side == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT, handles.space);
    }

    plugin_utils::WristSample wrist;
    if (m_wrist_source != nullptr)
    {
        wrist = m_wrist_source->query(side == 0, time);
    }

    if (!wrist.valid)
    {
        // No wrist source tracked: publish the skeleton wrist-relative, at the space origin, with the
        // valid-but-untracked flags. Consumers see the shape and know the placement is not tracked,
        // which is honest; the runtime is not left with a frozen pose.
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "hand %s: no wrist pose from the '%s' source, injecting the skeleton untracked",
                             hand_name(side), m_opts.wrist_source.c_str());
        injector->push(frame.skeleton.joints.data(), time);
        return;
    }

    // The glove reports the hand in its own frame, which is not anchored in the room; the wrist source
    // provides that anchor. The two frames differ by a rigid rotation that is fixed per hand, one
    // it is known from both sources at once.
    XrQuaternionf alignment{ 0.0f, 0.0f, 0.0f, 1.0f };
    if (m_alignment == Alignment::Explicit)
    {
        alignment = m_alignment_explicit;
    }
    else if (m_alignment == Alignment::Auto)
    {
        if (!m_auto_alignment[side].has_value() && wrist.tracked)
        {
            m_auto_alignment[side] = multiply_quaternions(
                wrist.pose.orientation, conjugate_quaternion(frame.skeleton.wrist_pose.orientation));
            const XrQuaternionf captured = *m_auto_alignment[side];
            RCLCPP_INFO(get_logger(),
                        "hand %s: wrist alignment captured as x,y,z,w = %.4f,%.4f,%.4f,%.4f; put it in "
                        "wrist_alignment to make it reproducible",
                        hand_name(side), captured.x, captured.y, captured.z, captured.w);
        }
        alignment = *m_auto_alignment[side];
    }

    std::array<XrHandJointLocationEXT, kOpenXrHandJointCount> joints = frame.skeleton.joints;
    for (std::size_t i = 0; i < joints.size(); ++i)
    {
        XrHandJointLocationEXT& joint = joints[i];
        // The wrist is where the anchor says it is; every other joint is a glove pose, so it needs the
        // frame alignment before being placed there.
        if (i == static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT))
        {
            continue;
        }
        if ((joint.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) == 0)
        {
            continue;
        }
        joint.pose = oxr_utils::multiply_poses(wrist.pose, rotate_pose(alignment, joint.pose));
        if (wrist.tracked)
        {
            joint.locationFlags |= kPoseTrackedFlags;
        }
    }
    joints[XR_HAND_JOINT_WRIST_EXT].pose = wrist.pose;
    joints[XR_HAND_JOINT_WRIST_EXT].locationFlags = wrist.tracked ? kPoseTrackedFlags : kPoseValidFlags;

    injector->push(joints.data(), time);
}

} // namespace io_exo_glove_ros2
} // namespace plugins
