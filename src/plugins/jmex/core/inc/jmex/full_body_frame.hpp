// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/full_body_generated.h>

#include <cstdint>

namespace plugins
{
namespace jmex
{

//! One frame from MOXI Player, in the slots of ``FullBodyPose``.
struct FullBodyFrame
{
    core::BodyJoints joints;
    //! Whether every slot is valid on this frame.
    bool all_joint_poses_tracked = false;
    uint32_t sequence_id = 0;
    //! Nanoseconds: Unix epoch time on Player's machine when the stream carries an accuracy
    //! timestamp, otherwise sequence id x 1e9 / 60 (MxRGetCurrentDataTimestamp).
    uint64_t player_time_ns = 0;
};

} // namespace jmex
} // namespace plugins
