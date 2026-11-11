/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_SIPL_CAPTURE_H
#define ARGUS_SENDER_SIPL_CAPTURE_H

// SIPL capture of every sensor in one platform configuration, ISP0 native NV12 out. The SIPL and NvSci headers stay
// inside SiplCapture.cpp: they pull in macros that collide with other libraries.

#include "CameraCapture.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace argus_sender
{

struct SiplConfig
{
    // Vendor platform configuration JSON and the named configuration inside it.
    std::string platformConfigJson;
    std::string platformConfigName;
    // One mask per deserializer, in transport order. Required for GMSL.
    std::vector<uint32_t> linkMasks;
    // Directory holding <module>.nito; the ISP has no tuning without it.
    std::string nitoDir = "/var/nvidia/nvcam/settings/sipl";
    int gpuId = 0;
    // ISP0 buffers per sensor. A frame an encoder still holds is one fewer for the ISP.
    uint32_t isp0Buffers = 6;
    uint32_t frameTimeoutMs = 1000;
    // Stamped on every frame; the caller changes it whenever capture is rebuilt.
    uint64_t sessionEpoch = 0;
};

// Descriptors only; touches no hardware. `id` is the SIPL pipeline index, which is neither the GMSL link index nor
// the JSON sensor id.
std::vector<CameraDescriptor> querySiplCameras(const std::string& platformConfigJson,
                                               const std::string& platformConfigName,
                                               const std::vector<uint32_t>& linkMasks);

class SiplCapture : public CameraCapture
{
public:
    explicit SiplCapture(const SiplConfig& config);
    ~SiplCapture() override;

    const std::vector<CameraDescriptor>& cameras() const override;
    void start() override;
    void stop() override;
    bool acquireLatest(uint32_t cameraId, CaptureFrame* frame) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_SIPL_CAPTURE_H
