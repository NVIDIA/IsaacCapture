/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_SENDER_H
#define ARGUS_SENDER_CAPTURE_SENDER_H

#include "CameraCapture.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace argus_sender
{

struct AppConfig;

// One AVS1 stream fed by one camera of a CameraCapture backend and HEVC-encoded at the camera's own size.
struct CaptureSenderStream
{
    uint32_t cameraId = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
};

// A fresh AVS1 session id. A backend stamps it on its frames as their session epoch.
uint64_t createCaptureSessionId();

// Accepts a receiver, then encodes each frame `capture` delivers straight from its DMA-BUF and sends it as its own
// bundle. `streams` parallels config.streams; `sessionId` is the epoch the backend stamps; `backendName` only labels
// log lines. Destroys `capture` once the connection is closed. Same return convention as runConfiguredCapture.
bool runCaptureSender(const AppConfig& config,
                      std::unique_ptr<CameraCapture> capture,
                      const std::vector<CaptureSenderStream>& streams,
                      uint64_t sessionId,
                      const char* backendName,
                      bool* receiverDisconnected,
                      bool appendDump);

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_SENDER_H
