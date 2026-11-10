// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The SIPL backend behind the sender's capture interface. A thin adapter over SiplCamera: frames are the
// borrowed CUDA RGBA views SiplCamera already produces, passed on without a copy.

#pragma once

#include "inc/sensing_core/camera_capture.hpp"
#include "inc/sensing_core/sipl_camera.hpp"

#include <memory>
#include <vector>

namespace plugins
{
namespace sensing
{

class SiplCapture : public CameraCapture
{
public:
    explicit SiplCapture(const SiplConfig& config);

    const std::vector<CameraInfo>& cameras() const override;
    void start() override;
    void stop() override;
    std::optional<CaptureFrame> acquire_latest(uint32_t camera_id) override;

private:
    SiplCamera m_camera;
    std::vector<CameraInfo> m_cameras;
};

} // namespace sensing
} // namespace plugins
