// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "CudaConversion.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace zed_sender
{
namespace
{
void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
__device__ uint8_t channel(int value)
{
    return uint8_t(max(0, min(255, (value + 128) >> 8)));
}
__global__ void convertYuyv(
    const uint8_t* source, size_t pitch, uint8_t* dest, size_t outPitch, uint32_t width, uint32_t height)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const uchar4 pair = *reinterpret_cast<const uchar4*>(source + y * pitch + 2 * x);
    const int u = int(pair.y) - 128, v = int(pair.w) - 128;
    for (unsigned i = 0; i < 2; ++i)
    {
        const int luma = (int(i ? pair.z : pair.x) - 16) * 298;
        *reinterpret_cast<uchar4*>(dest + y * outPitch + 4 * (x + i)) =
            make_uchar4(channel(luma + 409 * v), channel(luma - 100 * u - 208 * v), channel(luma + 516 * u), 255);
    }
}
}
struct CudaConversion::Impl
{
    uint32_t width, height;
    uint8_t* source = nullptr;
    size_t pitch = 0;
    cudaStream_t stream = nullptr;
    Impl(uint32_t w, uint32_t h) : width(w), height(h)
    {
    }
    ~Impl()
    {
        if (stream)
            cudaStreamSynchronize(stream);
        if (source)
            cudaFree(source);
        if (stream)
            cudaStreamDestroy(stream);
    }
};
CudaConversion::CudaConversion(uint32_t width, uint32_t height) : impl(new Impl(width, height))
{
    checkCuda(cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking), "create conversion stream");
    checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&impl->source), &impl->pitch, size_t(width) * 2, height),
              "allocate YUYV staging");
}
CudaConversion::~CudaConversion() = default;
void CudaConversion::convert(const uint8_t* host, size_t stride, uint8_t* deviceRgba, size_t rgbaPitch)
{
    auto& p = *impl;
    checkCuda(cudaMemcpy2DAsync(
                  p.source, p.pitch, host, stride, size_t(p.width) * 2, p.height, cudaMemcpyHostToDevice, p.stream),
              "upload YUYV");
    convertYuyv<<<dim3((p.width / 2 + 31) / 32, (p.height + 7) / 8), dim3(32, 8), 0, p.stream>>>(
        p.source, p.pitch, deviceRgba, rgbaPitch, p.width, p.height);
    checkCuda(cudaGetLastError(), "launch YUYV conversion");
    checkCuda(cudaStreamSynchronize(p.stream), "complete YUYV conversion");
}
}
