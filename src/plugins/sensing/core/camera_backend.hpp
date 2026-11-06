// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Choosing the camera backend at startup. This is the decision only: probing the hardware is each backend's job,
// and its result arrives here as a BackendProbe. No vendor SDK is involved, so the rules are testable alone.

#pragma once

#include "camera_capture.hpp"

#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/// What one backend found when asked whether it can open the configured rig.
struct BackendProbe
{
    CameraBackend backend = CameraBackend::Sipl;
    /// The backend is built in and its SDK loaded; false means it cannot run here at all.
    bool available = false;
    /// The backend found the configured cameras. Counts only when `available` is also true.
    bool matches = false;
    /// Why not, for the message when this backend was wanted: a missing library, no device, a bad configuration.
    std::string detail;
};

/// "auto", "argus" or "sipl"; nullopt for anything else.
std::optional<CameraBackend> parse_camera_backend(const std::string& text);

const char* camera_backend_name(CameraBackend backend);

/**
 * @brief The backend to open.
 *
 * An explicit request must match or the call throws with that backend's own reason: it never falls back to the
 * other one. Auto selects the only matching backend; none matching, or both matching, throws and names the
 * explicit override, because guessing which camera to open could take the wrong rig.
 *
 * @throws std::runtime_error with the reason.
 */
CameraBackend select_camera_backend(CameraBackend requested, const std::vector<BackendProbe>& probes);

} // namespace sensing
} // namespace plugins
