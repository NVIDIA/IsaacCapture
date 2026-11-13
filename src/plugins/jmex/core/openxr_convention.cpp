// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/jmex/openxr_convention.hpp"

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

core::Pose moxi_pose_to_openxr(const float moxi_position[3], const float moxi_orientation_wxyz[4])
{
    float position[3] = { 0.0f, 0.0f, 0.0f };
    moxi_axes_to_openxr(moxi_position, position);

    // Carried across the change of basis C, a rotation q becomes C q C^-1: the scalar part stays
    // and the vector part moves by C, so the axis map above serves the orientation too.
    const float moxi_vector_part[3] = { moxi_orientation_wxyz[1], moxi_orientation_wxyz[2], moxi_orientation_wxyz[3] };
    float openxr_vector_part[3] = { 0.0f, 0.0f, 0.0f };
    moxi_axes_to_openxr(moxi_vector_part, openxr_vector_part);

    // The SDK answers scalar-first and the schema stores the scalar last. Swapping the order of
    // this reshuffle and the change of basis raises nothing and tilts every published pose.
    return core::Pose(
        core::Point(position[0], position[1], position[2]),
        core::Quaternion(openxr_vector_part[0], openxr_vector_part[1], openxr_vector_part[2], moxi_orientation_wxyz[0]));
}

} // namespace jmex
} // namespace plugins
