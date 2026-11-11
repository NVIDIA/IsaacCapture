/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAMERA_CAPTURE_H
#define ARGUS_SENDER_CAMERA_CAPTURE_H

#include "CaptureFrame.h"

#include <cstdint>
#include <vector>

namespace argus_sender
{

// A camera backend that owns its whole rig: some camera SDKs hand out one process-wide object, so there is one
// instance per backend, never one per camera.
class CameraCapture
{
public:
    virtual ~CameraCapture()
    {
    }

    virtual const std::vector<CameraDescriptor>& cameras() const = 0;
    virtual void start() = 0;
    // Drop every frame lease before stopping; a lease outliving the backend is a use after free.
    virtual void stop() = 0;
    // Takes the newest frame of `cameraId` not yet taken. Returns false when none has arrived; never blocks. Throws
    // once the backend has failed.
    virtual bool acquireLatest(uint32_t cameraId, CaptureFrame* frame) = 0;
    // Blocks until some camera has a frame waiting or `timeoutMs` passes; true when one is waiting. Throws once the
    // backend has failed.
    virtual bool waitForFrame(uint32_t timeoutMs) = 0;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_CAMERA_CAPTURE_H
