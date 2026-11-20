// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
namespace zed_sender
{
class CudaConversion
{
public:
    CudaConversion(uint32_t width, uint32_t height);
    ~CudaConversion();
    CudaConversion(const CudaConversion&) = delete;
    CudaConversion& operator=(const CudaConversion&) = delete;
    // Upload packed BT.601-limited YUYV, write RGBA to mapped surface CUDA memory,
    // and wait for completion before another hardware engine reads the surface.
    void convert(const uint8_t* host, size_t stride, uint8_t* deviceRgba, size_t rgbaPitch);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
