// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zed_sender
{
enum class Eye
{
    Left,
    Right,
    Both
};

struct CameraConfig
{
    bool enabled = true;
    // Position in the config's camera list, counting disabled entries, so names stay stable.
    uint32_t index = 0;
    // A nonempty serial selects the camera; otherwise device is a path or "auto".
    std::string serial;
    std::string device = "auto";
    Eye eye = Eye::Left;
    uint32_t captureWidth = 1280;
    uint32_t captureHeight = 720;
    uint32_t captureFps = 60;
    uint32_t timeoutMs = 2000;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t fps = 60;
    uint32_t bitrateKbps = 50000;
    uint32_t gop = 60;
    uint32_t rotation = 0;

    std::string label() const
    {
        return "cam" + std::to_string(index);
    }
    std::string identity() const
    {
        return serial.empty() ? "device " + device : "serial " + serial;
    }
};

struct Config
{
    std::string bind = "0.0.0.0";
    std::string dumpWire;
    uint32_t port = 5000;
    uint32_t reconnectMs = 1000;
    uint32_t frames = 0;
    uint32_t logInterval = 5;
    uint32_t cameraRetryMs = 2000;
    // Every configured camera, enabled or not.
    std::vector<CameraConfig> cameras;

    std::vector<CameraConfig> enabledCameras() const;
};

const char* eyeName(Eye eye);
Config loadConfig(const std::string& path);
void validateConfig(const Config& config);
}
