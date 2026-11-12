// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "receiver_session.hpp"

#include <jmex/body_joint_mapping.hpp>
#include <jmex/openxr_convention.hpp>
#include <log_bridge/logger.hpp>
#include <schema/full_body_generated.h>

#include <MoxiRobotReceiver.h>
#include <cassert>
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

spdlog::logger& logger()
{
    static const std::shared_ptr<spdlog::logger> instance =
        isaaccapture::Logger::get("isaaccapture.plugins.jmex.ReceiverSession");
    return *instance;
}

//! Report which FullBodyPose slots this pairing's model fills.
void log_slot_coverage(int channel, const BodyJointMapping& mapping)
{
    const bool every_slot_filled = mapping.filled_slot_count() == static_cast<size_t>(core::BodyJoint_NUM_JOINTS);
    logger().log(every_slot_filled ? spdlog::level::info : spdlog::level::warn,
                 "channel {} full body filled {} of {} slots", channel, mapping.filled_slot_count(),
                 static_cast<int>(core::BodyJoint_NUM_JOINTS));

    // Only this log names the empty slots: samples carry poses, not joint names.
    if (mapping.filled_slot_count() == 0)
    {
        logger().warn(
            "no joint on this channel is named for a body slot, so the full body channel will publish {} invalid "
            "slots -- is the standard human skeleton loaded?",
            static_cast<int>(core::BodyJoint_NUM_JOINTS));
    }
    else
    {
        for (size_t slot = 0; slot < kStandardHumanSkeletonJointNames.size(); ++slot)
        {
            const auto body_joint = static_cast<core::BodyJoint>(slot);
            if (!mapping.joint_index(body_joint))
            {
                logger().warn("no joint named '{}' for slot {}", kStandardHumanSkeletonJointNames[slot],
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
    assert(version != nullptr);
    return version;
}

// The SDK's start and open calls return only a bool, with no reason; the messages below list the
// causes MoxiReceiver.h documents.

ReceiverSession::System::System(const ReceiverConfig& config)
{
    // The SDK's "no preference" is NULL.
    const char* bind_ip = config.bind_ip ? config.bind_ip->c_str() : nullptr;
    const char* broadcast_ip = config.broadcast_ip ? config.broadcast_ip->c_str() : nullptr;

    // The robot dialect, although nothing here reads robot joints: the change of basis in
    // openxr_convention.hpp was measured on it. Re-measure before switching dialects.
    if (!MxRStartSystem(bind_ip, config.tcp_port, broadcast_ip, config.udp_start_port, MOXIMocapType::ROBOT))
    {
        throw std::runtime_error("ReceiverSession: MxRStartSystem returned false for " +
                                 config.bind_ip.value_or(kEveryInterface) + ", TCP port " +
                                 std::to_string(config.tcp_port) +
                                 "; the SDK returns false when the receiver is already running, the port is outside "
                                 "0-65535, or the network cannot be initialized (for example, the port is in use)");
    }
}

ReceiverSession::System::~System()
{
    MxRFinishSystem();
}

ReceiverSession::Channel::Channel(int channel, int udp_start_port) : id_(channel)
{
    if (!MxROpenChannel(id_))
    {
        throw std::runtime_error("ReceiverSession: MxROpenChannel(" + std::to_string(id_) +
                                 ") returned false (UDP base port " + std::to_string(udp_start_port) +
                                 "); the SDK returns false when the receiver is not running, the channel is out of range "
                                 "or already open, or no data port can be bound (for example, the channel's data port "
                                 "is past 65535)");
    }
}

ReceiverSession::Channel::~Channel()
{
    MxRCloseChannel(id_);
}

ReceiverSession::ReceiverSession(const ReceiverConfig& config)
    : system_(config), channel_(config.channel, config.udp_start_port)
{
}

bool ReceiverSession::is_streaming() const
{
    return MxCheckChannelConnected(channel_.id());
}

std::optional<FullBodyFrame> ReceiverSession::poll()
{
    MxRUpdateSystem();

    if (!is_streaming())
    {
        // Do not log this from MxBindDisconnectEvent: it also fires when a connection that never
        // held this channel ends, such as one from a second Player the SDK refused.
        if (was_streaming_)
        {
            logger().info("MOXI Player disconnected; waiting for it to come back");
        }
        was_streaming_ = false;

        // Only this plugin's pairing goes: the SDK keeps the channel's joint table across a disconnect,
        // as it keeps the sequence id (see was_streaming_).
        pairing_.reset();
        return std::nullopt;
    }
    was_streaming_ = true;

    const uint32_t sequence_id = MxRGetCurrentDataSequenceID(channel_.id());
    if (sequence_id == 0)
    {
        return std::nullopt; // the SDK's answer before the channel's first frame
    }

    if (!pairing_)
    {
        pairing_.emplace(Pairing{ sequence_id, std::nullopt });
    }
    else
    {
        if (sequence_id == pairing_->last_sequence_id)
        {
            return std::nullopt; // no new frame; the values on the wire are the ones already handed out
        }

        // Ids step by one per frame, so a larger step counts frames that never reached us. An id
        // going backwards counts none: a pairing can start on the id a disconnect left behind (see
        // was_streaming_), and the new connection's ids start over.
        if (sequence_id > pairing_->last_sequence_id)
        {
            dropped_frames_ += sequence_id - pairing_->last_sequence_id - 1;
        }
        pairing_->last_sequence_id = sequence_id;
    }
    const uint64_t player_time_ns = MxRGetCurrentDataTimestamp(channel_.id());

    if (!pairing_->mapping)
    {
        pairing_->mapping = discover_joints();
        if (!pairing_->mapping)
        {
            return std::nullopt; // try again on the next frame
        }
    }
    const BodyJointMapping& mapping = *pairing_->mapping;

    unconstrained_global_poses(mapping.joint_count(), joint_poses_);

    FullBodyFrame frame;
    frame.all_joint_poses_tracked =
        mapping.apply(joint_poses_, frame.joints) == static_cast<size_t>(core::BodyJoint_NUM_JOINTS);
    frame.sequence_id = sequence_id;
    frame.player_time_ns = player_time_ns;
    return frame;
}

std::optional<BodyJointMapping> ReceiverSession::discover_joints() const
{
    const int joint_count = MxRGetBoneJointCount(channel_.id());
    if (joint_count <= 0)
    {
        return std::nullopt;
    }

    std::vector<std::string> joint_names;
    joint_names.reserve(static_cast<size_t>(joint_count));
    for (int bone = 0; bone < joint_count; ++bone)
    {
        // Every joint gets an entry: the table is indexed the way the per-joint SDK calls are, and a
        // name no slot carries simply fills nothing.
        const char* name = MxRGetBoneJointName(channel_.id(), bone);
        assert(name != nullptr);
        joint_names.emplace_back(name);
    }

    BodyJointMapping mapping(joint_names);
    logger().info("channel {} paired, {} joints", channel_.id(), joint_count);
    log_slot_coverage(channel_.id(), mapping);
    return mapping;
}

void ReceiverSession::unconstrained_global_poses(size_t joint_count, std::vector<core::Pose>& out) const
{
    out.resize(joint_count);

    // Seed with NaN, not identity: the SDK's read returns void, so a joint the frame did not carry
    // keeps the seed, and an identity pose at the origin would go out as a valid slot. NaN survives
    // the change of basis and is_usable() in body_joint_mapping.cpp rejects it.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t bone = 0; bone < joint_count; ++bone)
    {
        float quat_wxyz[4] = { nan, nan, nan, nan };
        float position[3] = { nan, nan, nan };
        MxRGetBoneJointMotionData(channel_.id(), static_cast<int>(bone), quat_wxyz, position, MOXI_DATA_TYPE_GLOBAL);

        out[bone] = moxi_pose_to_openxr(position, quat_wxyz);
    }
}

} // namespace jmex
} // namespace plugins
