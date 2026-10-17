// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "openxr_convention.h"

namespace plugins
{
namespace jmex
{

namespace
{

//! The change of basis on a plain 3-vector: a half turn about (0, 1, 1)/sqrt(2), determinant +1.
void moxi_axes_to_openxr(const float moxi[3], float openxr[3])
{
    openxr[0] = -moxi[0];
    openxr[1] = moxi[2];
    openxr[2] = moxi[1];
}

} // namespace

JointPose moxi_pose_to_openxr(const float moxi_position[3], const float moxi_orientation_wxyz[4])
{
    JointPose pose{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } };

    moxi_axes_to_openxr(moxi_position, pose.position);

    // Carried across the change of basis C, a rotation q becomes C q C^-1: the scalar part stays
    // and the vector part moves by C, so the axis map above serves the orientation too.
    const float moxi_vector_part[3] = { moxi_orientation_wxyz[1], moxi_orientation_wxyz[2], moxi_orientation_wxyz[3] };
    float openxr_vector_part[3] = { 0.0f, 0.0f, 0.0f };
    moxi_axes_to_openxr(moxi_vector_part, openxr_vector_part);

    // Deliberately a separate step: the SDK answers scalar-first and the schema stores it last.
    // Doing the reshuffle and the change of basis in the wrong order raises nothing at all -- it
    // just tilts every published pose.
    pose.orientation_xyzw[0] = openxr_vector_part[0];
    pose.orientation_xyzw[1] = openxr_vector_part[1];
    pose.orientation_xyzw[2] = openxr_vector_part[2];
    pose.orientation_xyzw[3] = moxi_orientation_wxyz[0];

    return pose;
}

} // namespace jmex
} // namespace plugins
