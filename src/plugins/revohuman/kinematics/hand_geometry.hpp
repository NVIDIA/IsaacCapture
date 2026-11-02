// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <revohuman.h>

namespace plugins
{
namespace revohuman
{

constexpr int kHandPointCount = 26;
constexpr std::uint32_t kHandPositionValid = 1u;
constexpr std::uint32_t kHandOrientationValid = 2u;

// OpenXR joint order, wrist-relative meters and normalized (x,y,z,w).
// Flags describe calculated data validity, not spatial tracking quality.
// Four non-thumb metacarpals are preview placeholders with flags == 0.
// Radius is a nominal 10 mm visualization value, not an anatomical measurement.
struct HandPoint
{
    double position[3];
    double orientation[4];
    float radius;
    std::uint32_t flags;
};

// SDK link poses, as rh_hand_points gives them.
struct LinkPose
{
    double p[3];
    double r[9]; // Row-major rotation matrix, same convention as the SDK FK API.
};

constexpr int kSdkPointCount = 27;
constexpr std::uint32_t kSdkPointsVersion = 1u;

// SDK hand point set (rh_hand_points, points version 1) in its own order and {hand}_tracker_link frame:
// 21 joint child links, five TIPs (index, middle, ring, little, thumb) and the palm_link origin. TIPs are
// taken as given; the tracker_link origin supplies PALM and the corrected palm_link origin supplies WRIST. Invalid
// quantities must be NaN, as the SDK writes them.
RhStatus hand_points_from_sdk_points(RhHand hand, const LinkPose points[kSdkPointCount], HandPoint out[kHandPointCount]);

// WRIST in the {hand}_tracker_link frame of an SDK point set: the frame the OpenXR points above are written
// in, palm_link (point 26) shifted toward the palm and turned by the reviewed WRIST rotation.
RhStatus wrist_from_sdk_points(RhHand hand, const LinkPose points[kSdkPointCount], LinkPose* wrist);

// The OpenXR point that moves with SDK link @p link (rh_hand_model_fk order): DIP and PIP links carry the
// finger's distal and intermediate points, a non-thumb MCP link its proximal point. -1 for a link no OpenXR
// point follows (MPR, thumb MCP/CMR/CMP) or out of range.
int openxr_joint_of_sdk_link(std::uint32_t link);

// Id of the reviewed mapping: its corrections and the rules that apply them.
const char* hand_points_mapping_id();
// SHA-256 (rh_hand_model_info urdf_sha256) of the SDK hand model named @p model_name on which the mapping
// was reviewed, or nullptr for a model it was not reviewed on.
const char* hand_points_reviewed_sha256(const char* model_name);

} // namespace revohuman
} // namespace plugins
