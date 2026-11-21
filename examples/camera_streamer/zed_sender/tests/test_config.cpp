// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Config.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
unsigned failures = 0;

void expect(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

void reject(const zed_sender::Config& bad, const char* what)
{
    try
    {
        zed_sender::validateConfig(bad);
        std::cerr << "ACCEPTED invalid config: " << what << '\n';
        ++failures;
    }
    catch (const std::runtime_error&)
    {
    }
}

zed_sender::Config load(const std::string& yaml)
{
    const std::string path = "zed_config_test.yaml";
    std::ofstream(path) << yaml;
    try
    {
        auto config = zed_sender::loadConfig(path);
        std::remove(path.c_str());
        return config;
    }
    catch (...)
    {
        std::remove(path.c_str());
        throw;
    }
}

void rejectYaml(const std::string& yaml, const char* what)
{
    try
    {
        load(yaml);
        std::cerr << "ACCEPTED invalid YAML: " << what << '\n';
        ++failures;
    }
    catch (const std::exception&)
    {
    }
}
}

int main()
{
    zed_sender::Config c;
    c.cameras.emplace_back();
    zed_sender::validateConfig(c);
    c.cameras[0].width = 640;
    c.cameras[0].height = 360;
    c.cameras[0].fps = 30;
    zed_sender::validateConfig(c);

    auto with = [&](const std::function<void(zed_sender::Config&)>& change, const char* what)
    {
        auto bad = c;
        change(bad);
        reject(bad, what);
    };
    with([](zed_sender::Config& b) { b.cameras[0].fps = 61; }, "output fps above capture fps");
    with([](zed_sender::Config& b) { b.cameras[0].width = 641; }, "odd width");
    with([](zed_sender::Config& b) { b.cameras[0].height = 480; }, "aspect ratio change");
    with([](zed_sender::Config& b) { b.cameras[0].bitrateKbps = 0; }, "zero bitrate");
    with([](zed_sender::Config& b) { b.port = 65536; }, "port out of range");
    with([](zed_sender::Config& b) { b.cameras[0].rotation = 90; }, "rotation 90");
    with([](zed_sender::Config& b) { b.cameras[0].timeoutMs = 0; }, "zero timeout");
    with(
        [](zed_sender::Config& b)
        {
            b.cameras[0].width = 2560;
            b.cameras[0].height = 1440;
        },
        "output larger than capture");
    with([](zed_sender::Config& b) { b.cameras[0].enabled = false; }, "no enabled camera");
    with([](zed_sender::Config& b) { b.cameras.push_back(b.cameras[0]); }, "two cameras on auto");
    with([](zed_sender::Config& b) { b.cameraRetryMs = 0; }, "zero camera retry");

    // A config without a cameras list is one camera built from the top-level settings.
    auto single = load(
        "version: 1\neye: both\ncapture: {device: /dev/video9, width: 1280, height: 720, fps: 30}\n"
        "output: {width: 640, height: 360, bitrate_mbps: 6}\n");
    expect(single.cameras.size() == 1 && single.cameras[0].device == "/dev/video9", "single camera device");
    expect(single.cameras[0].eye == zed_sender::Eye::Both, "single camera eye");
    expect(single.cameras[0].fps == 30 && single.cameras[0].gop == 30, "output fps and gop follow capture");
    expect(single.cameras[0].bitrateKbps == 6000, "bitrate_mbps converts to kbps");
    expect(load("version: 1\n").cameras[0].eye == zed_sender::Eye::Left, "default eye is left");

    auto multi = load(
        "version: 1\ncapture: {width: 1280, height: 720, fps: 15}\noutput: {bitrate_kbps: 500}\n"
        "cameras:\n"
        "  - serial: 111\n    output: {width: 320, height: 180}\n"
        "  - serial: 222\n    enabled: false\n"
        "  - serial: 333\n    eye: both\n    output: {width: 640, height: 360, bitrate_kbps: 1500, gop: 5}\n");
    expect(multi.cameras.size() == 3 && multi.enabledCameras().size() == 2, "disabled camera is skipped");
    expect(multi.enabledCameras()[1].index == 2 && multi.enabledCameras()[1].label() == "cam2",
           "disabling a camera does not rename the others");
    expect(multi.cameras[0].serial == "111" && multi.cameras[0].width == 320 && multi.cameras[0].fps == 15 &&
               multi.cameras[0].gop == 15 && multi.cameras[0].bitrateKbps == 500 &&
               multi.cameras[0].eye == zed_sender::Eye::Left,
           "camera inherits defaults");
    expect(multi.cameras[2].eye == zed_sender::Eye::Both && multi.cameras[2].bitrateKbps == 1500 &&
               multi.cameras[2].gop == 5 && multi.cameras[2].width == 640,
           "camera overrides defaults");

    rejectYaml("version: 1\ncapture: {device: /dev/video0}\ncameras:\n  - serial: 1\n", "top-level device with cameras");
    rejectYaml("version: 1\ncameras:\n  - serial: 1\n  - serial: 1\n", "duplicate serial");
    rejectYaml("version: 1\ncameras:\n  - serial: 1\n    eye: middle\n", "unknown eye");
    rejectYaml("version: 1\noutput: {bitrate_mbps: 1, bitrate_kbps: 500}\n", "two bitrate settings");
    rejectYaml("version: 1\ncameras:\n  - serial: 1\n    colour: red\n", "unknown camera setting");
    rejectYaml("version: 1\ncameras: []\n", "empty cameras list");

    if (failures)
        std::cerr << failures << " config test failures\n";
    return failures ? 1 : 0;
}
