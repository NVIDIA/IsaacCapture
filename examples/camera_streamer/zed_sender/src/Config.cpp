// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Config.h"

#include <algorithm>
#include <initializer_list>
#include <set>
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
bool read(const YAML::Node& node, const char* key, T& value)
{
    if (!node[key])
        return false;
    value = node[key].as<T>();
    return true;
}

Eye parseEye(const std::string& text, const std::string& context)
{
    if (text == "left")
        return Eye::Left;
    if (text == "right")
        return Eye::Right;
    if (text == "both")
        return Eye::Both;
    throw std::runtime_error(context + ".eye must be left, right, or both");
}

// Output settings left unset follow the capture mode, so remember which were given.
struct Explicit
{
    bool width = false, height = false, fps = false, gop = false;
};

void applyCapture(const YAML::Node& node, const std::string& context, bool allowDevice, CameraConfig& c)
{
    if (!node)
        return;
    keys(node, { "device", "width", "height", "fps", "timeout_ms" }, context);
    if (node["device"] && !allowDevice)
        throw std::runtime_error(context + ".device is not allowed with a cameras list; set it per camera");
    read(node, "device", c.device);
    read(node, "width", c.captureWidth);
    read(node, "height", c.captureHeight);
    read(node, "fps", c.captureFps);
    read(node, "timeout_ms", c.timeoutMs);
}

void applyOutput(const YAML::Node& node, const std::string& context, CameraConfig& c, Explicit& given)
{
    if (!node)
        return;
    keys(node, { "width", "height", "fps", "bitrate_mbps", "bitrate_kbps", "gop", "rotation" }, context);
    given.width |= read(node, "width", c.width);
    given.height |= read(node, "height", c.height);
    given.fps |= read(node, "fps", c.fps);
    given.gop |= read(node, "gop", c.gop);
    if (node["bitrate_mbps"] && node["bitrate_kbps"])
        throw std::runtime_error(context + ": set bitrate_mbps or bitrate_kbps, not both");
    uint32_t mbps = 0;
    if (read(node, "bitrate_mbps", mbps))
    {
        if (mbps > 200)
            throw std::runtime_error(context + ".bitrate_mbps must be 1..200");
        c.bitrateKbps = mbps * 1000;
    }
    read(node, "bitrate_kbps", c.bitrateKbps);
    read(node, "rotation", c.rotation);
}

void finishOutput(CameraConfig& c, const Explicit& given)
{
    if (!given.width)
        c.width = c.captureWidth;
    if (!given.height)
        c.height = c.captureHeight;
    if (!given.fps)
        c.fps = c.captureFps;
    if (!given.gop)
        c.gop = c.fps;
}
}

const char* eyeName(Eye eye)
{
    return eye == Eye::Left ? "left" : eye == Eye::Right ? "right" : "both";
}

std::vector<CameraConfig> Config::enabledCameras() const
{
    std::vector<CameraConfig> result;
    for (const auto& camera : cameras)
        if (camera.enabled)
            result.push_back(camera);
    return result;
}

Config loadConfig(const std::string& path)
{
    const YAML::Node root = YAML::LoadFile(path);
    keys(root, { "version", "network", "runtime", "eye", "capture", "output", "cameras" }, "config");
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
        keys(runtime, { "frames", "log_interval_s", "camera_retry_ms" }, "runtime");
        read(runtime, "frames", c.frames);
        read(runtime, "log_interval_s", c.logInterval);
        read(runtime, "camera_retry_ms", c.cameraRetryMs);
    }

    // Top-level eye/capture/output are the defaults every camera starts from.
    const auto cameras = root["cameras"];
    CameraConfig defaults;
    Explicit defaultsGiven;
    if (root["eye"])
        defaults.eye = parseEye(root["eye"].as<std::string>(), "config");
    applyCapture(root["capture"], "capture", !cameras, defaults);
    applyOutput(root["output"], "output", defaults, defaultsGiven);

    if (!cameras)
    {
        finishOutput(defaults, defaultsGiven);
        c.cameras.push_back(defaults);
    }
    else
    {
        if (!cameras.IsSequence() || cameras.size() == 0)
            throw std::runtime_error("cameras must be a nonempty list");
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            const auto node = cameras[i];
            const std::string context = "cameras[" + std::to_string(i) + "]";
            keys(node, { "enabled", "serial", "device", "eye", "capture", "output" }, context);
            CameraConfig camera = defaults;
            Explicit given = defaultsGiven;
            camera.index = uint32_t(i);
            read(node, "enabled", camera.enabled);
            read(node, "serial", camera.serial);
            read(node, "device", camera.device);
            if (node["eye"])
                camera.eye = parseEye(node["eye"].as<std::string>(), context);
            applyCapture(node["capture"], context + ".capture", false, camera);
            applyOutput(node["output"], context + ".output", camera, given);
            finishOutput(camera, given);
            c.cameras.push_back(camera);
        }
    }
    validateConfig(c);
    return c;
}

void validateConfig(const Config& config)
{
    if (config.port == 0 || config.port > 65535)
        throw std::runtime_error("network.port must be 1..65535");
    if (config.reconnectMs > 60000 || config.logInterval > 3600)
        throw std::runtime_error("invalid reconnect delay or logging interval");
    if (config.cameraRetryMs < 100 || config.cameraRetryMs > 60000)
        throw std::runtime_error("runtime.camera_retry_ms must be 100..60000");
    if (config.cameras.empty() || config.cameras.size() > 16)
        throw std::runtime_error("configure 1..16 cameras");
    const auto enabled = config.enabledCameras();
    if (enabled.empty())
        throw std::runtime_error("at least one camera must be enabled");
    std::set<std::string> identities;
    for (const auto& c : enabled)
    {
        const std::string context = c.label() + ": ";
        if (c.serial.empty() && c.device.empty())
            throw std::runtime_error(context + "set serial or device");
        if (c.serial.empty() && c.device == "auto" && enabled.size() > 1)
            throw std::runtime_error(context +
                                     "auto device selection needs a single enabled camera; set serial or device");
        if (!identities.insert(c.identity()).second)
            throw std::runtime_error(context + "duplicate camera " + c.identity());
        if (!c.captureFps || c.captureFps > 100 || !c.fps || c.fps > c.captureFps)
            throw std::runtime_error(context + "FPS must satisfy 1 <= output.fps <= capture.fps <= 100");
        for (const auto value : { c.captureWidth, c.captureHeight, c.width, c.height })
            if (value < 64 || value > 4096 || value % 2)
                throw std::runtime_error(context + "per-eye dimensions must be even and in 64..4096");
        if (c.width > c.captureWidth || c.height > c.captureHeight)
            throw std::runtime_error(context + "output dimensions must not exceed capture dimensions");
        if (uint64_t(c.width) * c.captureHeight != uint64_t(c.height) * c.captureWidth)
            throw std::runtime_error(context + "output must preserve the capture aspect ratio");
        if (c.bitrateKbps < 100 || c.bitrateKbps > 200000 || !c.gop || c.gop > 10000)
            throw std::runtime_error(context + "bitrate must be 100 kbps..200 Mbps per eye; gop must be 1..10000 frames");
        if (c.rotation != 0 && c.rotation != 180)
            throw std::runtime_error(context + "rotation must be 0 or 180");
        if (c.timeoutMs < 100 || c.timeoutMs > 10000)
            throw std::runtime_error(context + "capture.timeout_ms must be 100..10000");
    }
}
}
