// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera_backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace plugins
{
namespace sensing
{

std::optional<CameraBackend> parse_camera_backend(const std::string& text)
{
    if (text == "auto")
        return CameraBackend::Auto;
    if (text == "argus")
        return CameraBackend::Argus;
    if (text == "sipl")
        return CameraBackend::Sipl;
    return std::nullopt;
}

const char* camera_backend_name(CameraBackend backend)
{
    switch (backend)
    {
    case CameraBackend::Auto:
        return "auto";
    case CameraBackend::Argus:
        return "argus";
    case CameraBackend::Sipl:
        return "sipl";
    }
    return "unknown";
}

namespace
{

std::string describe(const BackendProbe& probe)
{
    const std::string reason = probe.detail.empty() ? "no reason given" : probe.detail;
    return std::string(camera_backend_name(probe.backend)) +
           (probe.available ? " (found no matching cameras: " : " (unavailable: ") + reason + ")";
}

} // namespace

CameraBackend select_camera_backend(CameraBackend requested, const std::vector<BackendProbe>& probes)
{
    if (requested != CameraBackend::Auto)
    {
        const auto probe =
            std::find_if(probes.begin(), probes.end(), [&](const BackendProbe& p) { return p.backend == requested; });
        if (probe == probes.end())
            throw std::runtime_error(std::string("--camera-backend=") + camera_backend_name(requested) +
                                     ": this build does not include that backend");
        if (!probe->available || !probe->matches)
            throw std::runtime_error(std::string("--camera-backend=") + camera_backend_name(requested) + ": " +
                                     describe(*probe));
        return requested;
    }

    std::vector<CameraBackend> matching;
    for (const auto& probe : probes)
    {
        if (probe.available && probe.matches)
            matching.push_back(probe.backend);
    }
    if (matching.size() == 1)
        return matching.front();

    std::string message;
    if (matching.empty())
    {
        message = "no camera backend matches the configured rig:";
        for (const auto& probe : probes)
            message += " " + describe(probe);
        message += ". Name the backend with --camera-backend=argus or --camera-backend=sipl";
    }
    else
    {
        message =
            "both camera backends match the configured rig; choose one with --camera-backend=argus or "
            "--camera-backend=sipl";
    }
    throw std::runtime_error(message);
}

} // namespace sensing
} // namespace plugins
