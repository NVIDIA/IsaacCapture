// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_utils/wrist_pose_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

namespace plugin_utils
{

WristSourceMode wrist_source_mode_from_env(const char* name, WristSourceMode fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }
    if (std::strcmp(value, "hand_tracking") == 0)
    {
        return WristSourceMode::HandTracking;
    }
    if (std::strcmp(value, "controller") == 0)
    {
        return WristSourceMode::Controller;
    }
    if (std::strcmp(value, "auto") == 0)
    {
        return WristSourceMode::Auto;
    }
    throw std::runtime_error(std::string(name) + " must be hand_tracking, controller or auto; got '" + value + "'");
}

XrPosef pose_from_env(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(std::string(name) + " is required");
    }
    XrPosef pose{};
    char trailing;
    if (std::sscanf(value, "%f,%f,%f,%f,%f,%f,%f %c", &pose.position.x, &pose.position.y, &pose.position.z,
                    &pose.orientation.x, &pose.orientation.y, &pose.orientation.z, &pose.orientation.w, &trailing) != 7)
    {
        throw std::runtime_error(std::string(name) + " must be px,py,pz,qx,qy,qz,qw");
    }
    const float values[] = { pose.position.x,    pose.position.y,    pose.position.z,   pose.orientation.x,
                             pose.orientation.y, pose.orientation.z, pose.orientation.w };
    if (!std::all_of(std::begin(values), std::end(values), [](float component) { return std::isfinite(component); }))
    {
        throw std::runtime_error(std::string(name) + " contains a non-finite value");
    }
    const double norm = std::hypot(std::hypot(static_cast<double>(pose.orientation.x), pose.orientation.y),
                                   std::hypot(static_cast<double>(pose.orientation.z), pose.orientation.w));
    if (norm < 1.0e-6)
    {
        throw std::runtime_error(std::string(name) + " contains a zero quaternion");
    }
    pose.orientation.x = static_cast<float>(pose.orientation.x / norm);
    pose.orientation.y = static_cast<float>(pose.orientation.y / norm);
    pose.orientation.z = static_cast<float>(pose.orientation.z / norm);
    pose.orientation.w = static_cast<float>(pose.orientation.w / norm);
    return pose;
}

XrPosef pose_from_env(const char* name, const XrPosef& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : pose_from_env(name);
}

} // namespace plugin_utils
