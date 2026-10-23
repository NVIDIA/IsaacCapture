// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "body_joint_mapping.hpp"

#include <schema/full_body_generated.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace jmex
{

/*!
 * @brief What the SDK needs from the host to bring a receiver up: the five fields of the SDK's
 *        sample config (MOXIReceiverSDK/samples/MOXIReceiverCfg.h), all properties of this machine
 *        rather than of the device.
 */
struct ReceiverConfig
{
    //! MOXI channel to open, one skeleton per channel. 1..10 is what a live capture assigns; 255
    //! is what Player uses when it replays a recording.
    int channel = 1;

    /*!
     * @brief Local address the receiver's TCP server binds. Empty means every interface.
     *
     * Advertisements go out from every broadcast-capable interface whatever this is set to, and
     * Player connects back to the address it heard one from: bind an address Player can reach.
     */
    std::string bind_ip;

    //! Where channel advertisements are broadcast. Empty means 255.255.255.255, which every MOXI
    //! Player on the subnet hears -- including ones on other machines, which then contend for the
    //! same channel. Naming this host keeps a pairing local.
    std::string broadcast_ip;

    //! TCP port Player connects back on, advertised in every broadcast. Each receiver on one
    //! machine needs its own.
    int tcp_port = 7000;

    //! Base of the per-channel data ports -- channel N receives on this + 1 + N. Only worth moving
    //! to make those ports predictable, for a firewall rule or to tell two receivers apart; a
    //! collision already relocates itself and announces where it landed. It does not move where
    //! advertisements are sent, which Player fixes at 10100.
    int udp_start_port = 10100;
};

//! Version of the MOXI Receiver SDK this plugin is linked against.
std::string moxi_sdk_version();

//! One frame from MOXI Player, in the 24 slots of ``FullBodyPose``.
struct FullBodyFrame
{
    core::BodyJoints joints;
    //! Whether every slot is valid on this frame.
    bool all_joint_poses_tracked = false;
    uint32_t sequence_id = 0;
    //! Unix epoch nanoseconds on Player's machine.
    uint64_t player_time_ns = 0;
};

/*!
 * @brief One robot-dialect MOXI channel, from pairing to each frame's 24 slots.
 *
 * MOXI Player is the TCP *client*: the SDK listens and the Player connects in, so a freshly
 * constructed session is normally not streaming yet, and neither is one whose Player disconnected.
 * Callers poll and wait rather than treating silence as a failure.
 */
class ReceiverSession
{
public:
    /*!
     * @throws std::runtime_error if another ReceiverSession already exists in this process, or if
     *         the SDK refuses to start -- the port is already taken, or `bind_ip` is not an address
     *         this machine has.
     */
    explicit ReceiverSession(const ReceiverConfig& config);
    ~ReceiverSession();

    ReceiverSession(const ReceiverSession&) = delete;
    ReceiverSession& operator=(const ReceiverSession&) = delete;

    /*!
     * @brief Pump the receiver once and hand out the frame it delivered, if that frame is new.
     *
     * A frame comes out only when the channel's sequence id moved *and* the joint table has been
     * read, so a caller publishes at most one sample per delivered frame instead of re-publishing
     * stale values at the poll rate. The joint table is read on the first frame of a pairing that
     * carries one.
     */
    std::optional<FullBodyFrame> poll();

    //! Frames the Player sent that never reached us, counted from sequence id gaps.
    int64_t dropped_frames() const
    {
        return dropped_frames_;
    }

private:
    //! Is the channel currently delivering motion data? False for a channel that is merely open, or
    //! paired but not yet sending.
    bool is_streaming() const;

    //! Read the channel's joint names and match them against the slots. Idempotent per pairing.
    void discover_joints();

    /*!
     * @brief Every joint's unconstrained global pose, in joint table order.
     *
     * @param out Resized to the joint table's size and overwritten.
     */
    void unconstrained_global_poses(std::vector<JointPose>& out) const;

    ReceiverConfig config_;
    uint32_t sequence_id_ = 0;
    int64_t dropped_frames_ = 0;
    //! Whether the channel was streaming at the previous poll, so each disconnect is logged once.
    bool was_streaming_ = false;
    bool joints_discovered_ = false;
    //! Every joint on the channel, in the index order the per-joint SDK calls take. Empty before
    //! the first frame of a pairing.
    std::vector<std::string> joint_names_;
    BodyJointMapping mapping_;
    //! This frame's joint poses, kept as a member so it is not reallocated per frame.
    std::vector<JointPose> joint_poses_;
};

} // namespace jmex
} // namespace plugins
