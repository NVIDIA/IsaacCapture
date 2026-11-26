// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/joint_se3_pose_generated.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

//! Which hand's fingertip frames to pack.
enum class Hand
{
    Left,
    Right,
};

//! Fingertip frame-name suffixes, in ascending JointName order (see kFingertipJoints).
inline constexpr const char* kFingertipFrameSuffixes[] = { "_thumb_tip", "_index_tip", "_middle_tip", "_ring_tip",
                                                           "_pinky_tip" };

//! Fingertip poses packed per hand (one per finger).
inline constexpr std::size_t kFingertipCount = 5;

//! Tensor size for the fingertip collections, in bytes. It matches ``JointSe3PoseTracker``'s own
//! default (the ``joint_se3_pose`` entry in ``src/core/deviceio_trackers/trackers.toml``), so a
//! consumer needs no custom size; five poses are ~200 B and a 26-joint OpenXR hand still fits.
inline constexpr std::size_t kTipMaxFlatbufferSize = 1024;

//! Outcome of pack_fingertip_poses().
struct TipPosePackResult
{
    //! Poses written to @c buffer; 0 when this hand has no fingertip frame in the reference frame.
    std::size_t poses = 0;
    //! Fingertip frames dropped because they are not in the reference frame.
    std::size_t skipped = 0;
    //! Frame the poses are expressed in; empty when nothing was packed.
    std::string reference_frame;
    //! Finished ``JointSe3PoseOutput`` FlatBuffer; empty when nothing was packed.
    std::vector<std::uint8_t> buffer;
};

//! Serializes the fingertip poses of @p hand found in @p msg as a ``JointSe3PoseOutput`` FlatBuffer
//! -- the schema for a sparse, vendor-chosen set of 6-DoF frames (raw vendor fingertips).
//!
//! The exoskeleton publishes every link, so this picks that hand's five
//! ``<side>_{thumb,index,middle,ring,pinky}_tip`` frames and maps them onto the matching
//! ``JointName`` keys. A frame that is absent is simply not tracked (the schema is sparse).
//!
//! One ``JointSe3PoseOutput`` carries a single reference frame for all of its poses and the schema
//! requires the producer to document it, so only tips published in @p reference_frame are packed and
//! anything else is counted in @c TipPosePackResult::skipped. @p reference_frame is therefore
//! required: it is the frame a consumer reads the poses as, and the publisher's own frame is not
//! inferred, because a silently adopted frame is one a consumer cannot know.
TipPosePackResult pack_fingertip_poses(const tf2_msgs::msg::TFMessage& msg,
                                       Hand hand,
                                       const std::string& device_id,
                                       const std::string& reference_frame);

} // namespace io_exo_glove_ros2
} // namespace plugins
