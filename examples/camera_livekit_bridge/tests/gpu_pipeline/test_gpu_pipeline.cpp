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

// Host reference: 2x2 box average (rounded) of an NV12 picture stored Y then UV with pitch = width.
std::vector<uint8_t> box_downscale_by_two(const std::vector<uint8_t>& src, const ImageSize& s)
{
    const uint32_t w = s.width / 2;
    const uint32_t h = s.height / 2;
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * 3 / 2);
    auto average = [](uint32_t sum) { return static_cast<uint8_t>((sum + 2) / 4); };
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < w; ++c)
        {
            const size_t i = static_cast<size_t>(2 * r) * s.width + 2 * c;
            out[static_cast<size_t>(r) * w + c] = average(src[i] + src[i + 1] + src[i + s.width] + src[i + s.width + 1]);
        }
    const size_t src_uv = static_cast<size_t>(s.width) * s.height;
    const size_t dst_uv = static_cast<size_t>(w) * h;
    for (uint32_t r = 0; r < h / 2; ++r)
        for (uint32_t c = 0; c < w / 2; ++c)
            for (uint32_t comp = 0; comp < 2; ++comp)
            {
                const size_t i = src_uv + static_cast<size_t>(2 * r) * s.width + (2 * c) * 2 + comp;
                out[dst_uv + static_cast<size_t>(r) * w + c * 2 + comp] =
                    average(src[i] + src[i + 2] + src[i + s.width] + src[i + s.width + 2]);
            }
    return out;
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

    // Steps run in series: a non-identity warp then a downscale equals the host-side box downscale of the warped
    // picture, and differs from skipping the warp or applying the steps in the other order.
    const std::string warp = "sine-warp=8,32";
    ImageSize warped_size;
    const auto warped = run({ warp }, size, warped_size);
    const auto expected = box_downscale_by_two(warped, size);
    const auto skipped_warp = box_downscale_by_two(gradient(size), size);
    check(expected != skipped_warp, "the warp changes what the downscale sees");

    const ImageSize half{ 64, 48 };
    GpuPipeline chain;
    chain.add(camera_bridge::make_gpu_step(warp));
    chain.add(camera_bridge::make_box_downscale_step(2));
    check(chain.prepare(size).empty() && chain.output_size() == half, "warp then downscale output size");
    Image source(size);
    source.upload(gradient(size));
    Image chained_out(half);
    check(chain.enqueue(source.in(), chained_out.out(), nullptr) == cudaSuccess, "chain enqueue");

    GpuPipeline reversed;
    reversed.add(camera_bridge::make_box_downscale_step(2));
    reversed.add(camera_bridge::make_gpu_step(warp));
    check(reversed.prepare(size).empty() && reversed.output_size() == half, "downscale then warp output size");
    Image reversed_out(half);
    check(reversed.enqueue(source.in(), reversed_out.out(), nullptr) == cudaSuccess, "reversed enqueue");
    cudaDeviceSynchronize();
    const auto chained = chained_out.download();
    check(chained == expected, "warp then downscale equals the downscale of the warped picture");
    check(chained != skipped_warp, "the chain does not bypass the warp");
    check(chained != reversed_out.download(), "the chain is order dependent");

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
