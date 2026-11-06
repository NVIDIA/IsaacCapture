// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The app's only XR<->MuJoCo frame crossing; Python reaches it through the
// pybind module rather than re-deriving it. Rules:
//
//   - Quaternions cross as xyzw (OpenXR, and so GRIP_ORIENTATION); MuJoCo is
//     wxyz and viz::Pose3D a third spelling. Reorder on every crossing, name
//     every variable q_xyzw or q_wxyz, fill boundary structs field by field.
//   - R_mj_from_xr = Rz(-90deg) * Rx(+90deg): XR -Z -> MJ +x, +Y -> +z,
//     +X -> -y. p_mj = R * p_xr + t; q_mj = q_mj_from_xr (x) q_xr.
//   - Name conversions `_from_` so they read by adjacency
//     (p_mj = mj_from_xr_pos(p_xr)), never the A_T_B robotics form.

#include "mj_api.hpp"

#include <array>

namespace viz
{

// A handedness convention, fixed by the two specs (OpenXR is y-up /
// -z-forward, MuJoCo is REP-103 z-up), so it cannot be wrong at runtime. If a
// scene's static content appears rotated 90 degrees, this is the bug; the ghost
// cannot show it, because it is placed through this transform and the eye pose
// goes into MuJoCo world through the same one, so it cancels.
//
// Deliberately diverges from
// examples/cloudxr_mujoco_teleop/python/isaaccapture_examples/cloudxr_mujoco_teleop/visualize_poses_mujoco_example.py,
// which applies Rx(+90) only and maps XR-forward to MuJoCo +y, which is not REP-103.
// Do not "fix" this constant to match it.
inline constexpr std::array<double, 4> kQuatMjFromXr = { 0.5, 0.5, -0.5, -0.5 }; // wxyz

// A workspace calibration, routinely wrong, and it places static scene content
// only -- it cancels on the ghost, so the shipped scene cannot show it. Two
// independent terms, and zeroing either is a bug:
//   x = -1.0  operator standoff, independent of the reference space.
//   z = -0.73 floor datum: MuJoCo z=0 is a work surface 0.73 m above the floor.
//             Only right against a floor-origin reference space, which this
//             session does not ask for -- viz's origin is the headset's start
//             pose. A scene with static content owns re-tuning it.
inline constexpr std::array<double, 3> kTransMjFromXr = { -1.0, 0.0, -0.73 };

// XR (xyzw) -> MuJoCo world (wxyz). The app's only quaternion crossing.
inline std::array<double, 4> mj_from_xr_quat(const std::array<double, 4>& q_xyzw)
{
    const mjtNum q_wxyz[4] = { q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2] }; // reorder
    std::array<double, 4> out{};
    mujoco::mju_mulQuat(out.data(), kQuatMjFromXr.data(), q_wxyz);
    return out;
}

// XR reference-space point -> MuJoCo world point: R * p + t.
inline std::array<double, 3> mj_from_xr_pos(const std::array<double, 3>& p_xr)
{
    std::array<double, 3> out{};
    mujoco::mju_rotVecQuat(out.data(), p_xr.data(), kQuatMjFromXr.data());
    for (int i = 0; i < 3; ++i)
    {
        out[i] += kTransMjFromXr[i];
    }
    return out;
}

} // namespace viz
