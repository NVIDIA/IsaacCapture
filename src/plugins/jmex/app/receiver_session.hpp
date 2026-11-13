// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <jmex/body_joint_mapping.hpp>
#include <jmex/full_body_frame.hpp>
#include <jmex/receiver_config.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace jmex
{

//! Version of the MOXI Receiver SDK this plugin is linked against.
std::string moxi_sdk_version();

/*!
 * @brief One robot-dialect MOXI channel, from pairing to each frame's FullBodyPose slots.
 *
 * MOXI Player is the TCP *client*: the SDK listens and the Player connects in, so a freshly
 * constructed session is normally not streaming yet, and neither is one whose Player disconnected.
 * Callers poll and wait rather than treating silence as a failure.
 */
class ReceiverSession
{
public:
    /*!
     * @throws std::runtime_error if the SDK will not start the receiver or open the channel.
     */
    explicit ReceiverSession(const ReceiverConfig& config);

    ReceiverSession(const ReceiverSession&) = delete;
    ReceiverSession& operator=(const ReceiverSession&) = delete;

    /*!
     * @brief Pump the receiver once and hand out the frame it delivered, if that frame is new.
     *
     * A frame comes out only when the channel's sequence id moved *and* the joint table has been
     * read: at most one per delivered frame. The joint table comes with the reference pose and is
     * read once per pairing; Player always disconnects before it switches models.
     */
    std::optional<FullBodyFrame> poll();

    //! Frames the Player sent that never reached us, counted from sequence id gaps.
    int64_t dropped_frames() const
    {
        return dropped_frames_;
    }

private:
    //! The SDK's receiver, started for this process and finished on destruction.
    class System
    {
    public:
        explicit System(const ReceiverConfig& config);
        ~System();
        System(const System&) = delete;
        System& operator=(const System&) = delete;
    };

    //! One open channel, closed on destruction.
    class Channel
    {
    public:
        Channel(int channel, int udp_start_port);
        ~Channel();
        Channel(const Channel&) = delete;
        Channel& operator=(const Channel&) = delete;

        int id() const
        {
            return id_;
        }

    private:
        int id_;
    };

    //! What one Player connection established; reset when the channel stops streaming.
    struct Pairing
    {
        uint32_t last_sequence_id;
        //! Empty until the joint table has been read.
        std::optional<BodyJointMapping> mapping;
    };

    //! Is the channel currently delivering motion data? False for a channel that is merely open, or
    //! paired but not yet sending.
    bool is_streaming() const;

    //! Read the channel's joint table, or nullopt before the channel's first reference pose.
    std::optional<BodyJointMapping> discover_joints() const;

    /*!
     * @brief Every joint's unconstrained global pose, in joint table order.
     *
     * @param out Resized to `joint_count` and overwritten.
     */
    void unconstrained_global_poses(size_t joint_count, std::vector<core::Pose>& out) const;

    System system_;
    Channel channel_;
    std::optional<Pairing> pairing_;
    int64_t dropped_frames_ = 0;
    //! Whether the channel was streaming at the previous poll, so each disconnect is logged once.
    //! Not pairing_.has_value(): motion that arrives before a reference pose sets the channel
    //! streaming but leaves its sequence id alone. That id is 0 from MxROpenChannel, and a disconnect
    //! does not reset it, although the SDK then waits for a new reference pose.
    bool was_streaming_ = false;
    //! This frame's joint poses, kept as a member so it is not reallocated per frame.
    std::vector<core::Pose> joint_poses_;
};

} // namespace jmex
} // namespace plugins
