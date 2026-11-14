// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

namespace plugin_utils
{

// SDK right-hand +Y is palmar; conjugate by a Z half-turn for OpenXR dorsal +Y.
constexpr XrPosef remap_to_openxr_basis(XrPosef pose, bool is_left)
{
    if (!is_left)
    {
        pose.position.x = -pose.position.x;
        pose.position.y = -pose.position.y;
        pose.orientation.x = -pose.orientation.x;
        pose.orientation.y = -pose.orientation.y;
    }
    return pose;
}

} // namespace plugin_utils
