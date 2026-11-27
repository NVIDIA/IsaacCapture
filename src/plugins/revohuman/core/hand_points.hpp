// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Not installed: the SDK hand point names and the point mapping, for hand_model.cpp and the tests.

#pragma once

#include "inc/revohuman_plugin/types.hpp"

#include <cstdint>
#include <revohuman_fk.h>

namespace plugins
{
namespace revohuman
{

//! The SDK hand points of RH_HAND_POINTS_VERSION 1 (rh_hand_point_name): the joint links (each finger from its DIP),
//! the TIPs, then the palm_link origin.
enum SdkPoint : std::uint32_t
{
    kIndexDip,
    kIndexPip,
    kIndexMcp,
    kIndexMpr,
    kMiddleDip,
    kMiddlePip,
    kMiddleMcp,
    kMiddleMpr,
    kRingDip,
    kRingPip,
    kRingMcp,
    kRingMpr,
    kLittleDip,
    kLittlePip,
    kLittleMcp,
    kLittleMpr,
    kThumbDip,
    kThumbPip,
    kThumbMcp,
    kThumbCmr,
    kThumbCmp,
    kIndexTip,
    kMiddleTip,
    kRingTip,
    kLittleTip,
    kThumbTip,
    kPalmLink,
};
static_assert(kThumbCmp + 1 == RH_JOINT_COUNT && kPalmLink + 1 == RH_HAND_POINT_COUNT);

//! The joints HandModel::joint_poses() gives for an SDK hand point set of a fitted model; the tests feed point sets
//! the SDK does not produce. Throws on an incompatible point set. @p retain_unmapped keeps the placeholders' fitted
//! poses for inspection instead of the identity.
HandJointPoses map_hand_points(Hand hand, const RhHandPoints& points, bool retain_unmapped = false);

} // namespace revohuman
} // namespace plugins
