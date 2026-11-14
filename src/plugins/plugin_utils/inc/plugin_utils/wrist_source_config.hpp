// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

namespace plugin_utils
{

/** @brief Which device provides the wrist pose. */
enum class WristSourceMode
{
    Auto, //!< Optical hand tracking preferred, controller aim pose as fallback.
    HandTracking, //!< XR_MNDX_xdev_space optical hand tracking only.
    Controller, //!< Controller aim pose (plus the configured rigid offset) only.
};

/** @brief Source selection plus per-device tuning. */
struct WristSourceConfig
{
    WristSourceMode mode = WristSourceMode::Auto;
    // Rigid transform from the controller aim pose to the wrist joint, per
    // hand. Depends on the controller model and how it is mounted on the
    // glove, so each plugin supplies its own calibrated pair (see
    // kLeft/kRightAimToWrist in wuji_glove_plugin.cpp and
    // kLeft/kRightHandOffset in manus_hand_tracking_plugin.cpp).
    XrPosef left_aim_to_wrist{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    XrPosef right_aim_to_wrist{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
};

/** @brief Read a wrist mode; an unset variable uses fallback, an unknown value throws. */
WristSourceMode wrist_source_mode_from_env(const char* name, WristSourceMode fallback);

/** @brief Read and normalize px,py,pz,qx,qy,qz,qw; missing or invalid values throw. */
XrPosef pose_from_env(const char* name);

/** @brief Use fallback only for an unset variable; invalid values throw. */
XrPosef pose_from_env(const char* name, const XrPosef& fallback);

/** @brief The canonical environment-variable spelling of a wrist mode. */
const char* to_string(WristSourceMode mode);

} // namespace plugin_utils
