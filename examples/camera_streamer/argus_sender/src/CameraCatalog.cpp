/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CameraCatalog.h"

#include "Error.h"

#include <cstdio>
#include <vector>

namespace argus_sender
{
namespace
{

void printSensorMode(uint32_t index, Argus::SensorMode* mode, const char* indent)
{
    Argus::ISensorMode* iMode = Argus::interface_cast<Argus::ISensorMode>(mode);
    if (!iMode)
        return;
    const Argus::Size2D<uint32_t> resolution = iMode->getResolution();
    const Argus::Range<uint64_t> durations = iMode->getFrameDurationRange();
    const double minFps = frameRateFromDurationNs(durations.max());
    const double maxFps = frameRateFromDurationNs(durations.min());
    std::printf("%sMode %u: %ux%u, frameDuration=%llu..%llu ns, fps=%.2f..%.2f\n", indent, index, resolution.width(),
                resolution.height(), static_cast<unsigned long long>(durations.min()),
                static_cast<unsigned long long>(durations.max()), minFps, maxFps);
}

bool getCameraDevices(Argus::CameraProvider* provider, std::vector<Argus::CameraDevice*>* devices)
{
    Argus::ICameraProvider* iProvider = Argus::interface_cast<Argus::ICameraProvider>(provider);
    if (!iProvider)
        ORIGINATE_ERROR("Failed to get ICameraProvider interface");
    if (iProvider->getCameraDevices(devices) != Argus::STATUS_OK)
        ORIGINATE_ERROR("Failed to get camera devices from provider");
    if (devices->empty())
        ORIGINATE_ERROR("No camera devices are available");
    return true;
}

bool getSensorModes(Argus::CameraDevice* device, std::vector<Argus::SensorMode*>* modes)
{
    Argus::ICameraProperties* iProperties = Argus::interface_cast<Argus::ICameraProperties>(device);
    if (!iProperties)
        ORIGINATE_ERROR("Failed to get ICameraProperties interface");
    if (iProperties->getAllSensorModes(modes) != Argus::STATUS_OK)
        ORIGINATE_ERROR("Failed to get sensor modes from camera device");
    if (modes->empty())
        ORIGINATE_ERROR("No sensor modes are available for this camera device");
    return true;
}

} // namespace

double frameRateFromDurationNs(uint64_t durationNs)
{
    return durationNs == 0 ? 0.0 : 1000000000.0 / static_cast<double>(durationNs);
}

uint32_t roundedFrameRateFromDurationNs(uint64_t durationNs)
{
    if (durationNs == 0)
        return 0;
    const uint64_t rounded = (1000000000ULL + durationNs / 2) / durationNs;
    return static_cast<uint32_t>(rounded > 0 ? rounded : 1);
}

bool listCameraModes(Argus::CameraProvider* provider)
{
    std::vector<Argus::CameraDevice*> devices;
    PROPAGATE_ERROR(getCameraDevices(provider, &devices));
    std::printf("Available CameraDevices: %zu\n", devices.size());
    for (uint32_t i = 0; i < devices.size(); ++i)
    {
        std::printf("CameraDevice %u:\n", i);
        std::vector<Argus::SensorMode*> modes;
        if (!getSensorModes(devices[i], &modes))
            continue;
        for (uint32_t mode = 0; mode < modes.size(); ++mode)
            printSensorMode(mode, modes[mode], "  ");
    }
    return true;
}

Argus::CameraDevice* selectCameraDevice(Argus::CameraProvider* provider, uint32_t index)
{
    std::vector<Argus::CameraDevice*> devices;
    if (!getCameraDevices(provider, &devices))
        return NULL;
    if (index >= devices.size())
    {
        std::fprintf(stderr, "Camera device %u was requested but only %zu are available.\n", index, devices.size());
        return NULL;
    }
    return devices[index];
}

Argus::SensorMode* selectSensorMode(Argus::CameraDevice* device, uint32_t index)
{
    std::vector<Argus::SensorMode*> modes;
    if (!getSensorModes(device, &modes))
        return NULL;
    if (index >= modes.size())
    {
        std::fprintf(stderr, "Sensor mode %u was requested but only %zu are available.\n", index, modes.size());
        for (uint32_t mode = 0; mode < modes.size(); ++mode)
            printSensorMode(mode, modes[mode], "  ");
        return NULL;
    }
    return modes[index];
}

} // namespace argus_sender
