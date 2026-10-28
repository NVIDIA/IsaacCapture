// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

/*!
 * @brief One-shot aim-to-wrist calibration: the fixed controller-aim -> wrist transform measured from
 * one held pose instead of guessed.
 *
 * An OpenXR aim pose points along its -Z axis. The controller is mounted on the glove along the
 * hand's long axis, and during calibration the hand is held flat, palm down, so the aim's -Z runs
 * along the fingers and the session's +Y (Y-up) is the hand's dorsal direction. The wrist joint is
 * therefore @c behind metres back along the fingers and @c height metres below the aim origin, and
 * its orientation is the one whose -Z is that fingers direction and whose +Y is up -- the convention
 * the injected skeleton uses. The offset is then inverse(aim) * wrist for one averaged held pose.
 */

//! Default wrist offset for the calibration: --aim-behind= / --aim-height= override them, in metres.
inline constexpr float kDefaultAimBehindM = 0.04f;
inline constexpr float kDefaultAimHeightM = 0.06f;

//! A held-pose window is rejected at or beyond these: the controller moved too much to average,
//! 5 mm and 2 deg.
inline constexpr float kAimMaxDriftM = 0.005f;
inline constexpr float kAimMaxDriftRad = 2.0f * 3.14159265358979323846f / 180.0f;

//! Wrist pose one aim sample implies (see above). @p aim is the raw controller aim pose in session
//! space, with no wrist offset applied.
XrPosef wrist_from_aim(const XrPosef& aim, float behind_m, float height_m);

//! The fixed controller-aim -> wrist transform: inverse(aim) * wrist.
XrPosef aim_to_wrist(const XrPosef& aim, const XrPosef& wrist);

//! Mean position and slerp-mean orientation of one hand's sample window. An empty window returns the
//! identity pose.
XrPosef average_aim_pose(const std::vector<XrPosef>& samples);

//! Largest position and orientation excursion of @p samples from their first sample.
struct AimDrift
{
    float position_m = 0.0f;
    float angle_rad = 0.0f;
};
AimDrift aim_window_drift(const std::vector<XrPosef>& samples);

//! True while every sample stays inside @p max_drift_m / @p max_drift_rad of the first one. At or
//! beyond either limit the window counts as moved and is rejected, so a 5 mm drift is too much.
bool aim_window_still(const std::vector<XrPosef>& samples, float max_drift_m, float max_drift_rad);

//! The exact `config/io_exo_glove_ros2.yaml` value for an aim-to-wrist transform: position in
//! metres, then the quaternion x,y,z,w, space-separated, without the surrounding quotes. That is
//! what --calibrate-aim prints, ready to paste into the config file.
std::string format_aim_to_wrist(const XrPosef& aim_to_wrist);

} // namespace io_exo_glove_ros2
} // namespace plugins
