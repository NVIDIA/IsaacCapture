// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace zed_sender
{
// An eye that was not selected at open() stays empty.
struct EncodedStereo
{
    std::vector<uint8_t> left, right;
};

class HardwareEncoder
{
public:
    HardwareEncoder();
    ~HardwareEncoder();
    HardwareEncoder(const HardwareEncoder&) = delete;
    HardwareEncoder& operator=(const HardwareEncoder&) = delete;
    void open(uint32_t inputWidth,
              uint32_t inputHeight,
              uint32_t outputWidth,
              uint32_t outputHeight,
              uint32_t fps,
              uint32_t bitrateKbps,
              uint32_t gop,
              bool encodeLeft = true,
              bool encodeRight = true);
    EncodedStereo encode(const uint8_t* yuyv, size_t bytes, size_t stride);
    void close() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
