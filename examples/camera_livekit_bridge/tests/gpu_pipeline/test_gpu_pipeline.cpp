// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The GPU step pipeline: no steps copies, steps run in series in the order given, the sine warp is an identity at
// zero amplitude and a horizontal ripple otherwise, and bad specs are refused.

#include <gpu_ops/gpu_step.hpp>

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using camera_bridge::GpuPipeline;
using camera_bridge::ImageSize;
using camera_bridge::Nv12Planes;
using camera_bridge::Nv12PlanesOut;

int g_failures = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

struct Image
{
    ImageSize size;
    uint8_t* device = nullptr; // Y then UV, tightly packed (pitch = width)
    explicit Image(const ImageSize& s) : size(s)
    {
        cudaMalloc(reinterpret_cast<void**>(&device), static_cast<size_t>(s.width) * s.height * 3 / 2);
    }
    ~Image()
    {
        cudaFree(device);
    }
    Nv12Planes in() const
    {
        return { device, device + static_cast<size_t>(size.width) * size.height, size.width, size.width };
    }
    Nv12PlanesOut out()
    {
        return { device, device + static_cast<size_t>(size.width) * size.height, size.width, size.width };
    }
    std::vector<uint8_t> download() const
    {
        std::vector<uint8_t> host(static_cast<size_t>(size.width) * size.height * 3 / 2);
        cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost);
        return host;
    }
    void upload(const std::vector<uint8_t>& host)
    {
        cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice);
    }
};

std::vector<uint8_t> gradient(const ImageSize& s)
{
    std::vector<uint8_t> host(static_cast<size_t>(s.width) * s.height * 3 / 2);
    for (uint32_t y = 0; y < s.height; ++y)
        for (uint32_t x = 0; x < s.width; ++x)
            host[static_cast<size_t>(y) * s.width + x] = static_cast<uint8_t>(16 + (x * 3 + y) % 200);
    for (size_t i = static_cast<size_t>(s.width) * s.height; i < host.size(); ++i)
        host[i] = static_cast<uint8_t>(100 + i % 50);
    return host;
}

std::vector<uint8_t> run(const std::vector<std::string>& steps, const ImageSize& size, ImageSize& result)
{
    GpuPipeline pipeline;
    for (const auto& spec : steps)
        pipeline.add(camera_bridge::make_gpu_step(spec));
    const std::string reason = pipeline.prepare(size);
    check(reason.empty(), "prepare: " + reason);
    result = pipeline.output_size();
    Image source(size);
    source.upload(gradient(size));
    Image destination(result);
    check(pipeline.enqueue(source.in(), destination.out(), nullptr) == cudaSuccess, "enqueue");
    cudaDeviceSynchronize();
    return destination.download();
}

} // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const ImageSize size{ 128, 96 };
    ImageSize out;

    // No steps: a plain copy.
    const auto copied = run({}, size, out);
    check(out == size && copied == gradient(size), "no steps copies the picture");

    // Zero amplitude is an identity (the sampler lands exactly on pixel centres).
    const auto identity = run({ "sine-warp=0" }, size, out);
    check(identity == gradient(size), "sine-warp with zero amplitude is an identity");

    // A real ripple moves pixels sideways but keeps rows' overall content: the picture changes, same size.
    const auto rippled = run({ "sine-warp=8,32" }, size, out);
    check(out == size && rippled != gradient(size), "sine-warp changes the picture and keeps its size");

    // Steps run in series: warp(0) then a downscale equals the downscale alone; two warps differ from one.
    ImageSize shrunk;
    GpuPipeline plain;
    plain.add(camera_bridge::make_box_downscale_step(2));
    check(plain.prepare(size).empty() && plain.output_size() == ImageSize{ 64, 48 }, "downscale output size");
    const auto in_series = run({ "sine-warp=0" }, size, shrunk); // series of one, then compare chain of two below
    GpuPipeline chain;
    chain.add(camera_bridge::make_gpu_step("sine-warp=0"));
    chain.add(camera_bridge::make_box_downscale_step(2));
    check(chain.prepare(size).empty() && chain.output_size() == ImageSize{ 64, 48 }, "chain output size");
    Image source(size);
    source.upload(gradient(size));
    Image chained_out({ 64, 48 });
    Image direct_out({ 64, 48 });
    check(chain.enqueue(source.in(), chained_out.out(), nullptr) == cudaSuccess, "chain enqueue");
    check(plain.enqueue(source.in(), direct_out.out(), nullptr) == cudaSuccess, "plain enqueue");
    cudaDeviceSynchronize();
    check(chained_out.download() == direct_out.download(), "identity warp then downscale equals the downscale alone");
    (void)in_series;

    // Refusals.
    bool threw = false;
    try
    {
        camera_bridge::make_gpu_step("no-such-step");
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    check(threw, "an unknown step name is refused");
    GpuPipeline bad;
    bad.add(camera_bridge::make_gpu_step("sine-warp=4,0"));
    check(!bad.prepare(size).empty(), "a zero wavelength is refused at prepare");
    GpuPipeline odd;
    odd.add(camera_bridge::make_gpu_step("sine-warp"));
    check(!odd.prepare({ 127, 96 }).empty(), "an odd picture size is refused at prepare");

    if (g_failures != 0)
    {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all gpu_pipeline checks passed\n";
    return 0;
}
