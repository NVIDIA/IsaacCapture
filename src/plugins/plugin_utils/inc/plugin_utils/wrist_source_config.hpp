// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/wrist_tracking_source.hpp>

namespace plugin_utils
{

using WristSourceMode = core::WristTrackingSourceMode;
using WristSourceConfig = core::WristTrackingSourceConfig;

/** @brief Read a wrist mode; an unset variable uses fallback, an unknown value throws. */
WristSourceMode wrist_source_mode_from_env(const char* name, WristSourceMode fallback);

/** @brief Read and normalize px,py,pz,qx,qy,qz,qw; missing or invalid values throw. */
XrPosef pose_from_env(const char* name);

/** @brief Use fallback only for an unset variable; invalid values throw. */
XrPosef pose_from_env(const char* name, const XrPosef& fallback);

/** @brief The canonical environment-variable spelling of a wrist mode. */
const char* to_string(WristSourceMode mode);

} // namespace plugin_utils
