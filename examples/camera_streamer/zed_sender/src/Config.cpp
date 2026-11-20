// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Config.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace zed_sender
{
namespace
{
void keys(const YAML::Node& node, std::initializer_list<const char*> allowed, const std::string& context)
{
    if (!node || !node.IsMap())
        throw std::runtime_error(context + " must be a mapping");
    for (const auto& item : node)
    {
        const auto key = item.first.as<std::string>();
        if (std::none_of(allowed.begin(), allowed.end(), [&](const char* name) { return key == name; }))
            throw std::runtime_error("unknown setting: " + context + "." + key);
    }
}

template <typename T>
void read(const YAML::Node& node, const char* key, T& value)
{
    if (node[key])
        value = node[key].as<T>();
}
}

Config loadConfig(const std::string& path)
{
    const YAML::Node root = YAML::LoadFile(path);
    keys(root, { "version", "network", "runtime", "capture", "output" }, "config");
    if (!root["version"] || root["version"].as<unsigned>() != 1)
        throw std::runtime_error("version must be 1");
    Config c;
    const auto network = root["network"];
    if (network)
    {
        keys(network, { "bind", "port", "reconnect_ms", "dump_wire" }, "network");
        read(network, "bind", c.bind);
        read(network, "port", c.port);
        read(network, "reconnect_ms", c.reconnectMs);
        read(network, "dump_wire", c.dumpWire);
    }
    const auto runtime = root["runtime"];
    if (runtime)
    {
        keys(runtime, { "frames", "log_interval_s" }, "runtime");
        read(runtime, "frames", c.frames);
        read(runtime, "log_interval_s", c.logInterval);
    }
    const auto capture = root["capture"];
    if (capture)
    {
        keys(capture, { "device", "width", "height", "fps", "timeout_ms" }, "capture");
        read(capture, "device", c.device);
        read(capture, "width", c.captureWidth);
        read(capture, "height", c.captureHeight);
        read(capture, "fps", c.captureFps);
        read(capture, "timeout_ms", c.timeoutMs);
    }
    c.width = c.captureWidth;
    c.height = c.captureHeight;
    c.fps = c.captureFps;
    c.gop = c.fps;
    const auto output = root["output"];
    if (output)
    {
        keys(output, { "width", "height", "fps", "bitrate_mbps", "gop", "rotation" }, "output");
        read(output, "width", c.width);
        read(output, "height", c.height);
        read(output, "fps", c.fps);
        c.gop = c.fps;
        read(output, "bitrate_mbps", c.bitrateMbps);
        read(output, "gop", c.gop);
        read(output, "rotation", c.rotation);
    }
    validateConfig(c);
    return c;
}

void validateConfig(const Config& c)
{
    if (c.device.empty() || c.port == 0 || c.port > 65535)
        throw std::runtime_error("capture.device must be nonempty; network.port must be 1..65535");
    if (!c.captureFps || c.captureFps > 100 || !c.fps || c.fps > c.captureFps)
        throw std::runtime_error("FPS must satisfy 1 <= output.fps <= capture.fps <= 100");
    for (const auto value : { c.captureWidth, c.captureHeight, c.width, c.height })
        if (value < 64 || value > 4096 || value % 2)
            throw std::runtime_error("per-eye dimensions must be even and in 64..4096");
    if (c.width > c.captureWidth || c.height > c.captureHeight)
        throw std::runtime_error("output dimensions must not exceed capture dimensions");
    if (uint64_t(c.width) * c.captureHeight != uint64_t(c.height) * c.captureWidth)
        throw std::runtime_error("output must preserve the capture aspect ratio");
    if (!c.bitrateMbps || c.bitrateMbps > 200 || !c.gop || c.gop > 10000)
        throw std::runtime_error("bitrate_mbps must be 1..200 per eye; gop must be 1..10000 frames");
    if (c.rotation != 0 && c.rotation != 180)
        throw std::runtime_error("rotation must be 0 or 180");
    if (c.timeoutMs < 100 || c.timeoutMs > 10000 || c.reconnectMs > 60000 || c.logInterval > 3600)
        throw std::runtime_error("invalid capture timeout, reconnect delay, or logging interval");
}
}
