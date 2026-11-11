/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArgusFrameAdapter.h"

#include "HevcEncoder.h"

#include <NvBufSurface.h>
#include <utility>

namespace argus_sender
{

bool describeArgusFrame(DmaBuffer* buffer,
                        const FrameTiming& timing,
                        uint64_t sessionEpoch,
                        uint32_t cameraId,
                        std::function<void()> release,
                        CaptureFrame* frame)
{
    if (!buffer || !frame)
        return false;
    NvBufSurface* surface = NULL;
    if (NvBufSurfaceFromFd(buffer->getFd(), reinterpret_cast<void**>(&surface)) != 0 || !surface)
        return false;
    const NvBufSurfaceParams& params = surface->surfaceList[0];
    if (params.planeParams.num_planes < 2)
        return false;

    CaptureFrame described;
    described.cameraId = cameraId;
    described.sessionEpoch = sessionEpoch;
    described.sequence = timing.frameNumber;
    described.width = params.width;
    described.height = params.height;
    if (timing.hasSensorSofTimestampTsc && timing.sensorSofTimestampTscNs != 0)
    {
        described.sensorTime = timing.sensorSofTimestampTscNs;
        described.sensorClock = SENSOR_CLOCK_TSC_NS;
    }
    described.readyMonotonicNs = timing.acquiredSteadyNs;
    described.surface.fd = buffer->getFd();
    for (int plane = 0; plane < 2; ++plane)
    {
        described.surface.pitchBytes[plane] = params.planeParams.pitch[plane];
        described.surface.offsetBytes[plane] = params.planeParams.offset[plane];
    }
    described.surface.blockLinear = params.layout == NVBUF_LAYOUT_BLOCK_LINEAR;
    described.lease = FrameLease(std::move(release));
    *frame = std::move(described);
    return true;
}

} // namespace argus_sender
