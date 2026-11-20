// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace zed_sender
{
struct Config
{
    std::string device = "auto";
    std::string bind = "0.0.0.0";
    std::string dumpWire;
    uint32_t port = 5000;
    uint32_t reconnectMs = 1000;
    uint32_t captureWidth = 1280;
    uint32_t captureHeight = 720;
    uint32_t captureFps = 60;
    uint32_t timeoutMs = 2000;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t fps = 60;
    uint32_t bitrateMbps = 50;
    uint32_t gop = 60;
    uint32_t rotation = 0;
    uint32_t frames = 0;
    uint32_t logInterval = 5;
};

Config loadConfig(const std::string& path);
void validateConfig(const Config& config);
}
