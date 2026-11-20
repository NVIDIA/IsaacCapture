// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zed_sender
{
class V4l2Capture
{
public:
    struct Frame
    {
        std::vector<uint8_t> bytes;
        uint32_t width = 0, height = 0, stride = 0, sequence = 0;
        uint64_t captureReadyRealtimeNs = 0, captureReadyMonotonicNs = 0;
    };
    V4l2Capture();
    ~V4l2Capture();
    V4l2Capture(const V4l2Capture&) = delete;
    V4l2Capture& operator=(const V4l2Capture&) = delete;
    void open(const std::string& device, uint32_t widthSbs, uint32_t height, uint32_t fps);
    bool readLatest(Frame& frame, int timeoutMs);
    // Call only after the acquisition thread exits. Format/FPS restore is best effort.
    void close() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
