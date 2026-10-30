// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/plugin_session.hpp>

#include <string>

namespace plugin_utils
{

/*!
 * @brief Factory for local plugin sessions backed by an OpenXR session named app_name.
 *
 * The session owns its OpenXR session and pull channel. OpenXR permits action
 * sets to be attached only once, so the pull channel is created with the
 * session. Its owner must destroy every returned push channel before
 * destroying the session.
 */
core::PluginSessionFactory make_openxr_plugin_session_factory(std::string app_name);

} // namespace plugin_utils
