// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sipl_capture.hpp"

namespace plugins
{
namespace sensing
{

SiplCapture::SiplCapture(const SiplConfig& config) : m_camera(config)
{
    for (const auto& sensor : m_camera.sensors())
        m_cameras.push_back({ sensor.id, sensor.name, sensor.width, sensor.height, sensor.fps });
}

const std::vector<CameraInfo>& SiplCapture::cameras() const
{
    return m_cameras;
}

void SiplCapture::start()
{
    m_camera.start();
}

void SiplCapture::stop()
{
    m_camera.stop();
}

std::optional<CaptureFrame> SiplCapture::acquire_latest(uint32_t camera_id)
{
    const auto view = m_camera.latest(camera_id);
    if (!view)
        return std::nullopt;

    CaptureFrame frame;
    frame.camera_id = camera_id;
    frame.sequence = view->sequence;
    frame.width = view->width;
    frame.height = view->height;
    frame.monotonic_ns = view->timestamp_ns;
    frame.unix_ns = view->unix_ns;
    frame.sensor_time = view->capture_tsc;
    frame.pixels = CudaRgbaPixels{ view->ptr, view->pitch };
    return frame;
}

} // namespace sensing
} // namespace plugins
