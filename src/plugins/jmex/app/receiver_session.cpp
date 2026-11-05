// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "receiver_session.hpp"

#include <jmex/body_joint_mapping.hpp>
#include <jmex/openxr_convention.hpp>
#include <log_bridge/logger.hpp>
#include <schema/full_body_generated.h>

#include <MoxiRobotReceiver.h>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins
{
namespace jmex
{

namespace
{

// The SDK's entry points are free functions over one internal receiver, so only one session may
// exist per process.
std::atomic<bool> g_session_alive{ false };

spdlog::logger& logger()
{
    static const std::shared_ptr<spdlog::logger> instance =
        isaaccapture::Logger::get("isaaccapture.plugins.jmex.ReceiverSession");
    return *instance;
}

//! Report which of the 24 slots this pairing's model fills.
void log_slot_coverage(int channel, const BodyJointMapping& mapping)
{
    // A shortfall is a warning, the count as much as the per-slot lines after it: a console that
    // shows only warnings should still show the count those lines explain.
    const bool every_slot_filled = mapping.filled_slot_count() == static_cast<size_t>(core::BodyJoint_NUM_JOINTS);
    logger().log(every_slot_filled ? spdlog::level::info : spdlog::level::warn,
                 "channel {} full body filled {} of {} slots", channel, mapping.filled_slot_count(),
                 static_cast<int>(core::BodyJoint_NUM_JOINTS));

    // Log the empty slots here: nothing downstream can name them, since samples carry poses, not
    // joint names.
    if (mapping.filled_slot_count() == 0)
    {
        logger().warn(
            "no joint on this channel is named for a body slot, so the full body channel will publish 24 invalid "
            "slots -- is the standard human skeleton loaded?");
    }
    else
    {
        for (size_t slot = 0; slot < k_standard_human_skeleton_joint_names.size(); ++slot)
        {
            const auto body_joint = static_cast<core::BodyJoint>(slot);
            if (mapping.joint_index(body_joint) < 0)
            {
                logger().warn("no joint named '{}' for slot {}", k_standard_human_skeleton_joint_names[slot],
                              core::EnumNameBodyJoint(body_joint));
            }
        }
    }

    for (const std::string& name : mapping.contested_joint_names())
    {
        logger().warn("joint '{}' matches a slot an earlier joint already fills; the earlier one keeps it", name);
    }
}

} // namespace

std::string moxi_sdk_version()
{
    const char* version = MxGetVersion();
    return (version != nullptr) ? version : "unknown";
}

ReceiverSession::ReceiverSession(const ReceiverConfig& config) : config_(config)
{
    bool expected = false;
    if (!g_session_alive.compare_exchange_strong(expected, true))
    {
        throw std::runtime_error(
            "ReceiverSession: a session already exists in this process; the MOXI "
            "receiver is process-global and cannot be opened twice");
    }

    // The SDK's "no preference" is NULL, not an empty string.
    const char* bind_ip = config_.bind_ip.empty() ? nullptr : config_.bind_ip.c_str();
    const char* broadcast_ip = config_.broadcast_ip.empty() ? nullptr : config_.broadcast_ip.c_str();

    // The robot dialect, although nothing here reads robot joints: the change of basis in
    // openxr_convention.hpp was measured on it. Re-measure before switching dialects.
    if (!MxRStartSystem(bind_ip, config_.tcp_port, broadcast_ip, config_.udp_start_port, MOXIMocapType::ROBOT))
    {
        g_session_alive.store(false);
        throw std::runtime_error("ReceiverSession: MxRStartSystem failed binding " +
                                 (config_.bind_ip.empty() ? std::string("every interface") : config_.bind_ip) +
                                 " on TCP port " + std::to_string(config_.tcp_port) +
                                 " -- the port is already taken, or that is not an address this machine has");
    }

    if (!MxROpenChannel(config_.channel))
    {
        MxRFinishSystem();
        g_session_alive.store(false);
        throw std::runtime_error("ReceiverSession: MxROpenChannel(" + std::to_string(config_.channel) +
                                 ") failed; nothing would ever arrive on this channel");
    }
}

ReceiverSession::~ReceiverSession()
{
    MxRCloseChannel(config_.channel);
    MxRFinishSystem();
    g_session_alive.store(false);
}

bool ReceiverSession::is_streaming() const
{
    return MxCheckChannelConnected(config_.channel);
}

std::optional<FullBodyFrame> ReceiverSession::poll()
{
    MxRUpdateSystem();

    if (!is_streaming())
    {
        // Logged on the channel's own transition, not on the SDK's disconnect event: that one also
        // fires when a connection that never held this channel ends, such as one from a second
        // Player the SDK refused.
        if (was_streaming_)
        {
            logger().info("MOXI Player disconnected; waiting for it to come back");
        }
        was_streaming_ = false;

        // A new pairing may bring a different model, so the joint table is re-read next time
        // rather than carried across the gap.
        joints_discovered_ = false;
        joint_names_.clear();
        sequence_id_ = 0;
        return std::nullopt;
    }
    was_streaming_ = true;

    const uint32_t sequence_id = MxRGetCurrentDataSequenceID(config_.channel);
    if (sequence_id == sequence_id_)
    {
        return std::nullopt; // no new frame; the values on the wire are the ones already handed out
    }

    // Ids step by one per frame, so a larger step counts frames that never reached us; the first
    // frame of a pairing has nothing to compare against. No branch for an id going backwards: only
    // a Player restart resets the count, and that drops the pairing first (!is_streaming() above).
    if (sequence_id_ != 0 && sequence_id > sequence_id_ + 1)
    {
        dropped_frames_ += sequence_id - sequence_id_ - 1;
    }
    sequence_id_ = sequence_id;
    const uint64_t player_time_ns = MxRGetCurrentDataTimestamp(config_.channel);

    if (!joints_discovered_)
    {
        discover_joints();
    }
    if (!joints_discovered_)
    {
        return std::nullopt;
    }

    unconstrained_global_poses(joint_poses_);

    FullBodyFrame frame;
    frame.all_joint_poses_tracked =
        mapping_.apply(joint_poses_, frame.joints) == static_cast<size_t>(core::BodyJoint_NUM_JOINTS);
    frame.sequence_id = sequence_id;
    frame.player_time_ns = player_time_ns;
    return frame;
}

void ReceiverSession::discover_joints()
{
    const int joint_count = MxRGetBoneJointCount(config_.channel);
    if (joint_count <= 0)
    {
        return; // the initial pose has not landed yet; try again on the next frame
    }

    joint_names_.clear();
    joint_names_.reserve(static_cast<size_t>(joint_count));
    for (int bone = 0; bone < joint_count; ++bone)
    {
        // Every joint gets an entry, named or not: the table is indexed the way the per-joint SDK
        // calls are, and a name no slot carries simply fills nothing.
        const char* raw_name = MxRGetBoneJointName(config_.channel, bone);
        joint_names_.emplace_back((raw_name != nullptr) ? raw_name : "");
    }

    mapping_ = BodyJointMapping::resolve(joint_names_);
    joints_discovered_ = true;
    logger().info("channel {} paired, {} joints", config_.channel, joint_count);
    log_slot_coverage(config_.channel, mapping_);
}

void ReceiverSession::unconstrained_global_poses(std::vector<JointPose>& out) const
{
    out.resize(joint_names_.size());

    // Seed with NaN, not identity: the SDK's read returns void, so a joint the frame did not carry
    // keeps the seed, and an identity pose at the origin would go out as a valid slot. NaN survives
    // the change of basis and is_usable() in body_joint_mapping.cpp rejects it.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t bone = 0; bone < joint_names_.size(); ++bone)
    {
        float quat_wxyz[4] = { nan, nan, nan, nan };
        float position[3] = { nan, nan, nan };
        MxRGetBoneJointMotionData(config_.channel, static_cast<int>(bone), quat_wxyz, position, MOXI_DATA_TYPE_GLOBAL);

        // The only conversion from the SDK's conventions to OpenXR's: FullBodyPose consumers expect
        // OpenXR axes from every producer.
        out[bone] = moxi_pose_to_openxr(position, quat_wxyz);
    }
}

} // namespace jmex
} // namespace plugins
