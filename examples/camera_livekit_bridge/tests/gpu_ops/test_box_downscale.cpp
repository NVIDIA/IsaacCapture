// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gpu_ops/box_downscale.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace camera_bridge;

namespace
{

int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                       \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

struct HostNv12
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> y; // width * height, tightly packed
    std::vector<uint8_t> uv; // width * (height / 2), interleaved U,V
};

HostNv12 make_pattern(uint32_t width, uint32_t height, bool blocky, uint32_t block)
{
    HostNv12 img;
    img.width = width;
    img.height = height;
    img.y.resize(static_cast<size_t>(width) * height);
    img.uv.resize(static_cast<size_t>(width) * (height / 2));
    for (uint32_t r = 0; r < height; ++r)
        for (uint32_t c = 0; c < width; ++c)
        {
            uint8_t v;
            if (blocky)
            {
                // Within a block the values are base, base+1, base+2, base+3 so the average rounds up or down.
                const uint32_t base = ((r / block) * 37 + (c / block) * 11) % 240;
                v = static_cast<uint8_t>(base + ((r % block) + (c % block)) % 4);
            }
            else
                v = static_cast<uint8_t>((r * 3 + c * 5) % 256);
            img.y[static_cast<size_t>(r) * width + c] = v;
        }
    for (uint32_t r = 0; r < height / 2; ++r)
        for (uint32_t c = 0; c < width; ++c)
            img.uv[static_cast<size_t>(r) * width + c] =
                static_cast<uint8_t>(c % 2 == 0 ? (r * 7 + c * 3) % 256 : (r * 13 + c * 5 + 90) % 256);
    return img;
}

HostNv12 reference_downscale(const HostNv12& src, uint32_t d)
{
    HostNv12 out;
    out.width = src.width / d;
    out.height = src.height / d;
    out.y.resize(static_cast<size_t>(out.width) * out.height);
    out.uv.resize(static_cast<size_t>(out.width) * (out.height / 2));
    const uint32_t n = d * d;
    for (uint32_t r = 0; r < out.height; ++r)
        for (uint32_t c = 0; c < out.width; ++c)
        {
            uint32_t sum = 0;
            for (uint32_t j = 0; j < d; ++j)
                for (uint32_t i = 0; i < d; ++i)
                    sum += src.y[static_cast<size_t>(r * d + j) * src.width + c * d + i];
            out.y[static_cast<size_t>(r) * out.width + c] = static_cast<uint8_t>((sum + n / 2) / n);
        }
    for (uint32_t r = 0; r < out.height / 2; ++r)
        for (uint32_t c = 0; c < out.width / 2; ++c)
            for (uint32_t comp = 0; comp < 2; ++comp)
            {
                uint32_t sum = 0;
                for (uint32_t j = 0; j < d; ++j)
                    for (uint32_t i = 0; i < d; ++i)
                        sum += src.uv[static_cast<size_t>(r * d + j) * src.width + (c * d + i) * 2 + comp];
                out.uv[static_cast<size_t>(r) * out.width + c * 2 + comp] = static_cast<uint8_t>((sum + n / 2) / n);
            }
    return out;
}

bool cuda_ok(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e));
        ++g_failures;
        return false;
    }
    return true;
}

struct DevicePlane
{
    uint8_t* ptr = nullptr;
    size_t pitch = 0;
};

DevicePlane alloc_plane(uint32_t row_bytes, uint32_t rows)
{
    DevicePlane p;
    // Pad the row so the pitch is always larger than the data, whatever alignment the allocator picks.
    cuda_ok(cudaMallocPitch(reinterpret_cast<void**>(&p.ptr), &p.pitch, row_bytes + 64, rows), "cudaMallocPitch");
    cuda_ok(cudaMemset2D(p.ptr, p.pitch, 0xEE, p.pitch, rows), "cudaMemset2D");
    return p;
}

void run_case(const HostNv12& src, uint32_t divisor, const char* label)
{
    const uint32_t dw = src.width / divisor;
    const uint32_t dh = src.height / divisor;
    DevicePlane sy = alloc_plane(src.width, src.height);
    DevicePlane suv = alloc_plane(src.width, src.height / 2);
    DevicePlane dy = alloc_plane(dw, dh);
    DevicePlane duv = alloc_plane(dw, dh / 2);
    CHECK(sy.pitch > src.width && dy.pitch > dw);

    cuda_ok(
        cudaMemcpy2D(sy.ptr, sy.pitch, src.y.data(), src.width, src.width, src.height, cudaMemcpyHostToDevice), "up y");
    cuda_ok(cudaMemcpy2D(suv.ptr, suv.pitch, src.uv.data(), src.width, src.width, src.height / 2, cudaMemcpyHostToDevice),
            "up uv");

    const Nv12Planes in{ sy.ptr, suv.ptr, sy.pitch, suv.pitch };
    const Nv12PlanesOut out{ dy.ptr, duv.ptr, dy.pitch, duv.pitch };
    CHECK(box_downscale_nv12(in, src.width, src.height, out, divisor, nullptr) == cudaSuccess);
    cuda_ok(cudaDeviceSynchronize(), "sync");

    std::vector<uint8_t> got_y(static_cast<size_t>(dw) * dh), got_uv(static_cast<size_t>(dw) * (dh / 2));
    cuda_ok(cudaMemcpy2D(got_y.data(), dw, dy.ptr, dy.pitch, dw, dh, cudaMemcpyDeviceToHost), "down y");
    cuda_ok(cudaMemcpy2D(got_uv.data(), dw, duv.ptr, duv.pitch, dw, dh / 2, cudaMemcpyDeviceToHost), "down uv");

    const HostNv12 ref = reference_downscale(src, divisor);
    const bool y_match = got_y == ref.y;
    const bool uv_match = got_uv == ref.uv;
    if (!y_match || !uv_match)
        std::fprintf(stderr, "case %s divisor %u: y_match=%d uv_match=%d\n", label, divisor, y_match, uv_match);
    CHECK(y_match);
    CHECK(uv_match);

    // The padding bytes past each destination row must be left alone.
    std::vector<uint8_t> raw(dy.pitch * dh);
    cuda_ok(cudaMemcpy(raw.data(), dy.ptr, raw.size(), cudaMemcpyDeviceToHost), "raw");
    bool pad_intact = true;
    for (uint32_t r = 0; r < dh; ++r)
        for (size_t c = dw; c < dy.pitch; ++c)
            pad_intact = pad_intact && raw[r * dy.pitch + c] == 0xEE;
    CHECK(pad_intact);

    cudaFree(sy.ptr);
    cudaFree(suv.ptr);
    cudaFree(dy.ptr);
    cudaFree(duv.ptr);
}

void test_supported()
{
    CHECK(box_downscale_supported(640, 480, 1));
    CHECK(box_downscale_supported(640, 480, 2));
    CHECK(box_downscale_supported(640, 480, 4));
    CHECK(box_downscale_supported(1280, 720, 4));
    CHECK(box_downscale_supported(64, 64, 8));
    CHECK(!box_downscale_supported(640, 480, 0));
    CHECK(!box_downscale_supported(640, 480, 9));
    CHECK(!box_downscale_supported(640, 480, 16));
    CHECK(!box_downscale_supported(642, 480, 4)); // not divisible
    CHECK(!box_downscale_supported(640, 482, 4));
    CHECK(!box_downscale_supported(644, 480, 4)); // divisible, odd result width (161)
    CHECK(!box_downscale_supported(640, 484, 4)); // odd result height (121)
    CHECK(!box_downscale_supported(641, 480, 1)); // odd source
    CHECK(!box_downscale_supported(0, 480, 1));
    CHECK(!box_downscale_supported(640, 0, 1));
}

void test_invalid_launch()
{
    DevicePlane y = alloc_plane(64, 64);
    DevicePlane uv = alloc_plane(64, 32);
    const Nv12Planes in{ y.ptr, uv.ptr, y.pitch, uv.pitch };
    const Nv12PlanesOut out{ y.ptr, uv.ptr, y.pitch, uv.pitch };
    CHECK(box_downscale_nv12(in, 64, 64, out, 0, nullptr) == cudaErrorInvalidValue);
    CHECK(box_downscale_nv12(in, 64, 64, out, 9, nullptr) == cudaErrorInvalidValue);
    CHECK(box_downscale_nv12(in, 66, 64, out, 4, nullptr) == cudaErrorInvalidValue);
    CHECK(box_downscale_nv12(in, 68, 64, out, 4, nullptr) == cudaErrorInvalidValue);
    cudaFree(y.ptr);
    cudaFree(uv.ptr);
}

} // namespace

int main()
{
    test_supported();

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    {
        std::fprintf(stderr, "SKIP: no CUDA device; only the CPU-side checks ran\n");
        return g_failures == 0 ? 77 : 1;
    }

    test_invalid_launch();
    // 128x96 divides by 1, 2 and 4 into even sizes; 64x32 also exercises divisor 8 to the minimum even size.
    for (uint32_t d : { 1u, 2u, 4u })
    {
        run_case(make_pattern(128, 96, false, d), d, "gradient");
        run_case(make_pattern(128, 96, true, d), d, "blocky");
    }
    run_case(make_pattern(64, 32, false, 8), 8, "gradient");

    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("gpu_ops tests passed\n");
    return 0;
}
