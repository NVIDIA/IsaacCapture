// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_plugin.hpp"

#include "io_exo_glove_ros2_aim.hpp"
#include "io_exo_glove_ros2_kinematics.hpp"
#include "io_exo_glove_ros2_pack.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

//! Name prefixes the glove's 42-joint message uses to say which hand a joint belongs to.
constexpr const char* kJointNamePrefixes[] = { "joint_Left", "joint_Right" };

//! `--calibrate-aim` sampling: about 200 Hz for 1 s, matching the rate the controller is tracked and
//! the glove streams at, and retried a few times while the operator settles the pose.
constexpr auto kAimSamplePeriod = std::chrono::milliseconds(5);
constexpr auto kAimWindowLength = std::chrono::milliseconds(1000);
constexpr auto kAimTrackingTimeout = std::chrono::seconds(3);
constexpr auto kAimRetryPause = std::chrono::milliseconds(500);
constexpr int kAimMaxAttempts = 3;

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

//! Parses the "x,y,z,qx,qy,qz,qw" pose spelling; the quaternion is normalised, a zero one rejected.
//! Commas and whitespace both separate the fields, because --calibrate-aim prints the value
//! space-separated so the line it emits can be pasted into the config file unchanged.
XrPosef parse_pose(const std::string& text, const char* what)
{
    std::string separated = text;
    std::replace(separated.begin(), separated.end(), ',', ' ');

    XrPosef pose{};
    int parsed = 0;
    const int fields =
        std::sscanf(separated.c_str(), "%f %f %f %f %f %f %f%n", &pose.position.x, &pose.position.y, &pose.position.z,
                    &pose.orientation.x, &pose.orientation.y, &pose.orientation.z, &pose.orientation.w, &parsed);
    if (fields != 7 || parsed != static_cast<int>(separated.size()))
    {
        throw std::runtime_error(std::string(what) + " must be 'x,y,z,qx,qy,qz,qw', not '" + text + "'");
    }
    const float norm = std::sqrt(pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
                                 pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
    if (!(norm > 0.0f))
    {
        throw std::runtime_error(std::string(what) + " has a zero quaternion: '" + text + "'");
    }
    pose.orientation.x /= norm;
    pose.orientation.y /= norm;
    pose.orientation.z /= norm;
    pose.orientation.w /= norm;
    return pose;
}

const char* hand_name(std::size_t side)
{
    return side == 0 ? "left" : "right";
}

} // namespace

bool IoExoGloveRos2Plugin::Throttler::ready()
{
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (now - m_last < m_period)
    {
        return false;
    }
    m_last = now;
    return true;
}

IoExoGloveRos2Plugin::IoExoGloveRos2Plugin(const Options& options)
    : rclcpp::Node("io_exo_glove_ros2_plugin"),
      m_opts(options),
      m_logger(isaaccapture::Logger::get("isaaccapture.plugins.io_exo_glove_ros2."
                                         "IoExoGloveRos2Plugin"))
{
    // The old per-joint zero offsets are accepted (an existing config file must still load) but do
    // nothing: the tip chord cannot be inverted or biased by a session's open pose.
    if (!m_opts.left_joint_zero_offsets.empty() || !m_opts.right_joint_zero_offsets.empty())
    {
        m_logger->warn("left/right_joint_zero_offsets are ignored: flexion is derived from the fingertip chords");
    }

    // The wrist source's trackers and extensions have to be known before the session exists, since
    // the session is what the injection and the queries run on.
    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = parse_wrist_source(m_opts.wrist_source);
    m_detect_stale_wrist = wrist_config.mode != plugin_utils::WristSourceMode::Controller;
    wrist_config.left_aim_to_wrist = parse_pose(m_opts.left_aim_to_wrist, "left_aim_to_wrist");
    wrist_config.right_aim_to_wrist = parse_pose(m_opts.right_aim_to_wrist, "right_aim_to_wrist");
    const plugin_utils::WristPoseSource::Requirements wrist_requirements =
        plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);
    const std::vector<std::shared_ptr<core::ITracker>> trackers = wrist_requirements.trackers;

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
    // Under "auto" the primary source only falls back to the controller when optical tracking is
    // absent. A frozen optical pose is still "tracked", so a second controller-only source is kept
    // for the stale-wrist handover; it shares the same DeviceIOSession and controller tracker.
    if (wrist_config.mode == plugin_utils::WristSourceMode::Auto && wrist_requirements.controller_tracker != nullptr &&
        m_deviceio_session != nullptr)
    {
        plugin_utils::WristSourceConfig controller_config = wrist_config;
        controller_config.mode = plugin_utils::WristSourceMode::Controller;
        m_controller_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
            controller_config, handles, m_deviceio_session.get(), wrist_requirements.controller_tracker);
    }

    m_joints_left_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.joints_left_collection_id,
                                                            .max_flatbuffer_size = kJointsMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_state",
                                                            .localized_name = "Exoskeleton Glove Joints (Left)",
                                                            .app_name = "IoExoGloveRos2Plugin" });
    m_joints_right_pusher = std::make_unique<core::SchemaPusher>(
        m_session->get_handles(), core::SchemaPusherConfig{ .collection_id = m_opts.joints_right_collection_id,
                                                            .max_flatbuffer_size = kJointsMaxFlatbufferSize,
                                                            .tensor_identifier = "joint_state",
                                                            .localized_name = "Exoskeleton Glove Joints (Right)",
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

    m_logger->info(
        "io_exo_glove_ros2_plugin: joints '{}' -> '{}', '{}' ({} B each); tips '{}' -> '{}', '{}' ({} B each, frame "
        "'{}'); hands -> XR_HAND_{{LEFT,RIGHT}}_EXT (wrist '{}', smoothing {:.0f} ms, aim_to_wrist left '{}' right '{}')",
        m_opts.joints_topic.c_str(), m_opts.joints_left_collection_id.c_str(),
        m_opts.joints_right_collection_id.c_str(), kJointsMaxFlatbufferSize, m_opts.links_topic.c_str(),
        m_opts.tips_left_collection_id.c_str(), m_opts.tips_right_collection_id.c_str(), kTipMaxFlatbufferSize,
        m_opts.tips_reference_frame.c_str(), wrist_source_name(wrist_config.mode), m_opts.wrist_smoothing_ms,
        m_opts.left_aim_to_wrist.c_str(), m_opts.right_aim_to_wrist.c_str());
}

void IoExoGloveRos2Plugin::calibrate_aim(float behind_m, float height_m, bool aim_flip)
{
    // The measured transform replaces the configured one, so folding the configured offset into the
    // query would calibrate against itself. main() builds this session with identity offsets; the
    // controller source is still the one that supplies the raw aim pose.
    if (parse_wrist_source(m_opts.wrist_source) != plugin_utils::WristSourceMode::Controller)
    {
        throw std::runtime_error("--calibrate-aim requires --wrist-source=controller, not '" + m_opts.wrist_source + "'");
    }
    if (m_wrist_source == nullptr || m_deviceio_session == nullptr || !m_time_converter.has_value())
    {
        throw std::runtime_error("--calibrate-aim: the controller wrist source is unavailable");
    }

    // Pump the trackers and read both raw aim poses: the same update-then-query order the tick uses.
    const auto sample = [this](std::array<plugin_utils::WristSample, kSideCount>& samples)
    {
        m_deviceio_session->update();
        const XrTime time = m_time_converter->os_monotonic_now();
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            samples[side] = m_wrist_source->query(side == 0, time);
        }
    };

    // Both controllers have to be actively tracked before the window starts: averaging a cached pose
    // would produce a plausible-looking offset from a controller that is not actually being held.
    std::array<bool, kSideCount> seen{ false, false };
    const std::chrono::steady_clock::time_point tracking_deadline =
        std::chrono::steady_clock::now() + kAimTrackingTimeout;
    while (std::chrono::steady_clock::now() < tracking_deadline && !(seen[0] && seen[1]))
    {
        std::array<plugin_utils::WristSample, kSideCount> samples{};
        sample(samples);
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            seen[side] = seen[side] || (samples[side].valid && samples[side].tracked);
        }
        std::this_thread::sleep_for(kAimSamplePeriod);
    }
    if (!(seen[0] && seen[1]))
    {
        std::string missing;
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            if (!seen[side])
            {
                missing += missing.empty() ? hand_name(side) : std::string(" and ") + hand_name(side);
            }
        }
        throw std::runtime_error("--calibrate-aim: no tracked controller aim pose from the " + missing +
                                 " hand; make sure the controller is paired and tracked");
    }

    for (int attempt = 1; attempt <= kAimMaxAttempts; ++attempt)
    {
        m_logger->info(
            "aim calibration: hold both hands flat, palm down, fingers forward for {} ms "
            "(attempt {}/{})",
            kAimWindowLength.count(), attempt, kAimMaxAttempts);

        std::array<std::vector<XrPosef>, kSideCount> windows;
        const std::chrono::steady_clock::time_point window_end = std::chrono::steady_clock::now() + kAimWindowLength;
        while (std::chrono::steady_clock::now() < window_end)
        {
            std::array<plugin_utils::WristSample, kSideCount> samples{};
            sample(samples);
            for (std::size_t side = 0; side < kSideCount; ++side)
            {
                if (samples[side].valid && samples[side].tracked)
                {
                    windows[side].push_back(samples[side].pose);
                }
            }
            std::this_thread::sleep_for(kAimSamplePeriod);
        }

        bool still = true;
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            if (windows[side].empty())
            {
                m_logger->warn("aim calibration: the {} controller was not tracked during the window", hand_name(side));
                still = false;
                continue;
            }
            if (!aim_window_still(windows[side], kAimMaxDriftM, kAimMaxDriftRad))
            {
                const AimDrift drift = aim_window_drift(windows[side]);
                m_logger->warn(
                    "aim calibration: the {} controller moved {:.1f} mm / {:.1f} deg during the window; "
                    "hold still",
                    hand_name(side), drift.position_m * 1000.0f, drift.angle_rad * (180.0f / 3.14159265358979323846f));
                still = false;
            }
        }
        if (!still)
        {
            if (attempt == kAimMaxAttempts)
            {
                break;
            }
            m_logger->warn("aim calibration: window rejected, retrying -- hold both hands flat and still");
            std::this_thread::sleep_for(kAimRetryPause);
            continue;
        }

        std::array<XrPosef, kSideCount> aim_to_wrist_offsets{};
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            const XrPosef aim = average_aim_pose(windows[side]);
            aim_to_wrist_offsets[side] = aim_to_wrist(aim, wrist_from_aim(aim, behind_m, height_m, aim_flip));
        }

        // The printed lines are the config value exactly; --print-config reports the same spelling.
        std::cout << "left_aim_to_wrist:  \"" << format_aim_to_wrist(aim_to_wrist_offsets[0]) << "\"\n"
                  << "right_aim_to_wrist: \"" << format_aim_to_wrist(aim_to_wrist_offsets[1]) << "\"\n"
                  << "Paste both lines into config/io_exo_glove_ros2.yaml, then record normally." << std::endl;
        return;
    }

    throw std::runtime_error(
        "--calibrate-aim: could not collect a still 1 s window; rerun and hold both hands "
        "flat and still");
}

void IoExoGloveRos2Plugin::push_joint_state(const sensor_msgs::msg::JointState& msg)
{
    // ROS 2 specifies that a JointState's arrays are either empty or equally sized. A mismatched
    // message is dropped rather than forwarded as a partial device state: a subset of the joints looks
    // like a valid, smaller configuration downstream, which is worse than skipping the frame.
    if (msg.name.size() != msg.position.size())
    {
        if (m_joint_size_warn.ready())
        {
            m_logger->warn("Dropping JointState: {} names but {} positions", msg.name.size(), msg.position.size());
        }
        return;
    }

    // The glove publishes both hands in one message. Split it by joint name so each hand keeps its own
    // collection: with one shared collection a consumer cannot tell a glove that dropped out from a
    // hand that is standing still, and JointStateSource would read the missing hand as 0.0 radians.
    std::array<std::vector<std::size_t>, kSideCount> per_side;
    std::size_t unassigned = 0;
    for (std::size_t i = 0; i < msg.name.size(); ++i)
    {
        bool assigned = false;
        for (std::size_t side = 0; side < kSideCount; ++side)
        {
            if (msg.name[i].rfind(kJointNamePrefixes[side], 0) == 0)
            {
                per_side[side].push_back(i);
                assigned = true;
                break;
            }
        }
        if (!assigned)
        {
            ++unassigned;
        }
    }
    if (unassigned != 0 && m_joint_split_warn.ready())
    {
        m_logger->warn("Dropping {} joint(s) of JointState: name matches neither '{}' nor '{}'", unassigned,
                       kJointNamePrefixes[0], kJointNamePrefixes[1]);
    }

    const int64_t local_ns = core::os_monotonic_now_ns();
    const int64_t stamp_ns = stamp_to_ns(msg.header.stamp);
    // Prefer the message's own header stamp as the raw device clock when the publisher sets one;
    // otherwise fall back to the local monotonic clock (SchemaPusher's documented convention for
    // "device clock not available").
    const int64_t raw_ns = (stamp_ns != 0) ? stamp_ns : local_ns;

    const std::pair<std::size_t, core::SchemaPusher*> sides[] = { { 0, m_joints_left_pusher.get() },
                                                                  { 1, m_joints_right_pusher.get() } };
    for (const auto& [side, pusher] : sides)
    {
        if (per_side[side].empty())
        {
            continue; // That glove is not in this message: its collection goes quiet, as it should.
        }

        // The glove's own DOF names and angles are forwarded unchanged: mapping them onto a particular
        // dexterous hand is the host retargeter's job, not this bridge's.
        core::JointStateOutputT out;
        out.device_id = pusher->config().collection_id;
        out.has_velocity = false;
        out.has_effort = false;
        out.ee_pose_valid = false;
        out.joints.reserve(per_side[side].size());
        for (const std::size_t index : per_side[side])
        {
            auto joint = std::make_shared<core::JointStateT>();
            joint->name = msg.name[index];
            joint->position = static_cast<float>(msg.position[index]);
            joint->valid = true;
            out.joints.push_back(std::move(joint));
        }

        flatbuffers::FlatBufferBuilder builder(kJointsMaxFlatbufferSize);
        auto offset = core::JointStateOutput::Pack(builder, &out);
        builder.Finish(offset);

        // SchemaPusher::push_buffer() throws when the payload exceeds the declared tensor size, and
        // nothing catches that here, so an oversized frame is dropped rather than aborting the
        // subscription callback.
        const std::size_t serialized_size = builder.GetSize();
        if (serialized_size > kJointsMaxFlatbufferSize)
        {
            if (m_joint_size_warn.ready())
            {
                m_logger->warn("Dropping JointState for the {} hand: serialized size {} exceeds {} bytes",
                               hand_name(side), serialized_size, kJointsMaxFlatbufferSize);
            }
            continue;
        }
        pusher->push_buffer(builder.GetBufferPointer(), serialized_size, local_ns, raw_ns);
    }
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
        if (packed.skipped != 0 && m_tip_frame_warn.ready())
        {
            m_logger->warn("Skipped {} fingertip frame(s) not in the reference frame '{}'", packed.skipped,
                           packed.reference_frame.c_str());
        }
        if (packed.poses == 0)
        {
            continue; // Sparse stream: this hand reports no fingertip frames in that frame right now.
        }
        if (packed.buffer.size() > kTipMaxFlatbufferSize)
        {
            if (m_tip_size_warn.ready())
            {
                m_logger->warn(
                    "Dropping fingertip poses: {} bytes exceed {}", packed.buffer.size(), kTipMaxFlatbufferSize);
            }
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

    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::array<HandSkeleton, kSideCount> to_log;
    bool log_slots = false;
    {
        std::lock_guard<std::mutex> lock(m_hand_mutex);
        for (const Hand hand : { Hand::Left, Hand::Right })
        {
            const std::size_t side = side_index(hand);
            HandFrame& frame = m_hand_frames[side];
            const HandAngles angles = read_hand_angles(msg, hand);
            frame.skeleton = build_anatomical_skeleton(angles, hand);
            frame.chord_m = angles.chord_m;
            frame.stamp = now;
            frame.valid = frame.skeleton.tracked();
        }
        if (!m_slots_logged && (m_hand_frames[0].valid || m_hand_frames[1].valid))
        {
            // Once, on the first frame that carries a hand: what the injection bound to, so a rig
            // whose frame names differ is diagnosable from the plugin log alone.
            m_slots_logged = true;
            log_slots = true;
            to_log[0] = m_hand_frames[0].skeleton;
            to_log[1] = m_hand_frames[1].skeleton;
        }
    }

    if (!log_slots)
    {
        return;
    }
    for (std::size_t side = 0; side < kSideCount; ++side)
    {
        if (!to_log[side].tracked())
        {
            m_logger->warn("hand {}: no mechanism frame in '{}', nothing is injected for it", hand_name(side),
                           m_opts.links_topic.c_str());
            continue;
        }
        m_logger->info("hand {}: {} of {} OpenXR joints injected from mechanism '{}'", hand_name(side),
                       to_log[side].filled, kOpenXrHandJointCount, to_log[side].wrist_frame.c_str());
    }
}

void IoExoGloveRos2Plugin::hand_tick()
{
    if (m_deviceio_session != nullptr)
    {
        // DeviceIOSession::update() documents a throw as fatal: the application is expected to
        // terminate. Let it propagate out of the timer callback and rclcpp::spin, where main's
        // handler logs it and returns 1; catching it here would keep a recording running with no
        // hands and nothing failing.
        m_deviceio_session->update();
    }
    if (!m_time_converter.has_value())
    {
        return;
    }

    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    const float dt_s = std::chrono::duration<float>(now - m_last_tick).count();
    m_last_tick = now;

    const XrTime time = m_time_converter->os_monotonic_now();
    push_hand(0, m_left_injector, time, dt_s);
    push_hand(1, m_right_injector, time, dt_s);
}

XrPosef IoExoGloveRos2Plugin::smooth_wrist(std::size_t side, const XrPosef& pose, float dt_s)
{
    SmoothedWrist& smoothed = m_smoothed_wrists[side];
    if (!smoothed.has)
    {
        smoothed.pose = pose;
        smoothed.has = true;
        return smoothed.pose;
    }
    const float tau_s = static_cast<float>(m_opts.wrist_smoothing_ms) / 1000.0f;
    const float alpha = (tau_s > 0.0f && dt_s > 0.0f) ? 1.0f - std::exp(-dt_s / tau_s) : 1.0f;
    smoothed.pose.position =
        XrVector3f{ smoothed.pose.position.x + alpha * (pose.position.x - smoothed.pose.position.x),
                    smoothed.pose.position.y + alpha * (pose.position.y - smoothed.pose.position.y),
                    smoothed.pose.position.z + alpha * (pose.position.z - smoothed.pose.position.z) };
    smoothed.pose.orientation = slerp_quaternions(smoothed.pose.orientation, pose.orientation, alpha);
    return smoothed.pose;
}

void IoExoGloveRos2Plugin::push_hand(std::size_t side,
                                     std::unique_ptr<plugin_utils::HandInjector>& injector,
                                     XrTime time,
                                     float dt_s)
{
    HandFrame frame;
    {
        std::lock_guard<std::mutex> lock(m_hand_mutex);
        frame = m_hand_frames[side];
    }

    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    const double now_s = std::chrono::duration<double>(now.time_since_epoch()).count();
    const bool fresh = frame.valid && (now - frame.stamp) < kHandStaleThreshold;
    if (!fresh)
    {
        // The glove is absent: forget the stale-wrist baseline, so the first frame after it returns
        // starts clean, and drop the injector so the runtime reports the hand inactive.
        m_frozen_wrist[side].reset();
        injector.reset();
        return;
    }

    const auto ensure_injector = [&]()
    {
        if (injector == nullptr)
        {
            const core::OpenXRSessionHandles handles = m_session->get_handles();
            injector = std::make_unique<plugin_utils::HandInjector>(
                handles.instance, handles.session, side == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT, handles.space);
        }
    };

    plugin_utils::WristSample wrist;
    if (m_wrist_source != nullptr)
    {
        wrist = m_wrist_source->query(side == 0, time);
    }

    if (!wrist.valid)
    {
        m_frozen_wrist[side].reset();
        // No wrist source tracked: publish the anatomy at the space origin with the valid-but-untracked
        // flags. Consumers see the shape and know the placement is not tracked, which is honest; the
        // runtime is not left with a frozen pose.
        if (m_wrist_warn[side].ready())
        {
            m_logger->warn("hand {}: no wrist pose from the '{}' source, injecting the skeleton untracked",
                           hand_name(side), m_opts.wrist_source.c_str());
        }
        ensure_injector();
        injector->push(frame.skeleton.joints.data(), time);
        return;
    }

    // The raw pose, before smoothing, is what tells a live source from a stalled one: smoothing a
    // repeated pose would still produce a slightly different value and hide the stall. A configured
    // controller source is legitimately static, so it is not checked.
    if (m_detect_stale_wrist)
    {
        const FrozenWristGuard::Result guard =
            m_frozen_wrist[side].update(wrist.pose, frame.chord_m, m_controller_wrist_source != nullptr, now_s);
        if (guard.newly_frozen)
        {
            m_logger->warn("hand {}: the wrist pose has not changed for {:.0f} ms while the fingertip chords moved; {}",
                           hand_name(side), FrozenWristGuard::kFrozenHoldS * 1000.0,
                           m_controller_wrist_source != nullptr ? "switching to the controller wrist source" :
                                                                  "withholding the hand until the wrist moves");
        }
        if (guard.thawed)
        {
            m_logger->info("hand {}: the wrist pose is live again", hand_name(side));
        }
        if (guard.action == FrozenWristGuard::Action::Suppress)
        {
            // Drop the push device: the runtime reports the hand inactive instead of holding the
            // frozen wrist, and the next live frame creates a fresh injector.
            injector.reset();
            return;
        }
        if (guard.action == FrozenWristGuard::Action::SwitchToController && m_controller_wrist_source != nullptr)
        {
            const plugin_utils::WristSample controller = m_controller_wrist_source->query(side == 0, time);
            if (controller.valid && controller.tracked)
            {
                wrist = controller;
                // The controller pose is not the optical one, so the smoother must start from it
                // rather than blend the two sources together.
                m_smoothed_wrists[side].has = false;
            }
            else
            {
                if (m_wrist_warn[side].ready())
                {
                    m_logger->warn(
                        "hand {}: the wrist source is frozen and no controller is tracked; withholding the hand",
                        hand_name(side));
                }
                injector.reset();
                return;
            }
        }
    }

    if (wrist.valid)
    {
        // Optical hand tracking is noisy, and a glove in front of the headset makes it worse; the
        // glove's own articulation is not, so only the anchor is filtered. Smoothing a valid but
        // not-actively-tracked pose too means a source that updates slower than the glove still
        // produces a distinct pose on every injected frame instead of repeating one bit-identically.
        wrist.pose = smooth_wrist(side, wrist.pose, dt_s);
    }

    ensure_injector();

    // The skeleton is anatomical and wrist-relative, so placing it needs no frame alignment: the wrist
    // pose is the one constant the anatomy already assumes, and the glide's own frame contributes only
    // articulation angles. Nothing here depends on when a frame arrived.
    std::array<XrHandJointLocationEXT, kOpenXrHandJointCount> joints = frame.skeleton.joints;
    for (std::size_t i = 0; i < joints.size(); ++i)
    {
        XrHandJointLocationEXT& joint = joints[i];
        if (i == static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT))
        {
            continue;
        }
        joint.pose = oxr_utils::multiply_poses(wrist.pose, joint.pose);
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
