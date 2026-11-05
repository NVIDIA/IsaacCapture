// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The Argus backend behind the sender's capture interface: each camera streams NV12 into a pool of NvBufSurface
// DMA-BUFs that the Jetson hardware encoder reads directly. Built only with SENSING_WITH_ARGUS, which needs the
// Argus headers and libnvargus; the Argus types stay out of this header.

#pragma once

#include "camera_capture.hpp"

#include <memory>
#include <vector>

namespace plugins
{
namespace sensing
{

struct ArgusConfig
{
    /// Argus camera device indices to capture; CameraInfo::id is the device index.
    std::vector<uint32_t> devices;
    /// Index into each device's sensor modes (as listed by query()).
    uint32_t sensor_mode = 0;
    /// Capture buffers per camera. More than the encoder holds (its input plane) plus one being acquired, or the
    /// camera stalls waiting for a buffer.
    uint32_t buffer_count = 10;
};

class ArgusCapture : public CameraCapture
{
public:
    /// @throws std::runtime_error naming the failing Argus or EGL step.
    explicit ArgusCapture(const ArgusConfig& config);
    ~ArgusCapture() override;

    /// Every Argus device with the given sensor mode's size and rate. Opens no camera.
    static std::vector<CameraInfo> query(uint32_t sensor_mode);

    const std::vector<CameraInfo>& cameras() const override;
    void start() override;
    void stop() override;
    std::optional<CaptureFrame> acquire_latest(uint32_t camera_id) override;

private:
    // Shared with the frame leases it issues, so the Argus objects outlive every outstanding frame.
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace sensing
} // namespace plugins
