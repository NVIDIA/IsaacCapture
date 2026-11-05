// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/sine_warp.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

using video_stage::ColorInfo;
using video_stage::Geometry;
using video_stage::Nv12View;
using video_stage::SineWarp;
using video_stage::SineWarpParams;

namespace
{

constexpr uint8_t kPadding = 0xAB;
constexpr float kTwoPi = 6.28318530717958647692F;

void cuda_ok(cudaError_t status)
{
    if (status != cudaSuccess)
        FAIL("CUDA error: " << cudaGetErrorString(status));
}

bool have_cuda_device()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Host-side NV12 image with tightly packed rows.
struct HostNv12
{
    uint32_t w = 0;
    uint32_t h = 0;
    std::vector<uint8_t> y; // w * h
    std::vector<uint8_t> uv; // w * (h / 2), interleaved U,V
};

HostNv12 make_image(uint32_t w, uint32_t h)
{
    HostNv12 img;
    img.w = w;
    img.h = h;
    img.y.resize(static_cast<size_t>(w) * h);
    img.uv.resize(static_cast<size_t>(w) * (h / 2));
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            img.y[y * w + x] = static_cast<uint8_t>(20 + (3 * x + 2 * y + ((x * y) % 7)) % 200);
    for (uint32_t cy = 0; cy < h / 2; ++cy)
        for (uint32_t cx = 0; cx < w / 2; ++cx)
        {
            img.uv[cy * w + 2 * cx] = static_cast<uint8_t>(30 + (5 * cx + cy) % 190); // U
            img.uv[cy * w + 2 * cx + 1] = static_cast<uint8_t>(220 - (2 * cx + 4 * cy) % 180); // V
        }
    return img;
}

// Device NV12 surface with pitch deliberately larger than the row size.
class DeviceSurface
{
public:
    DeviceSurface(uint32_t w, uint32_t h, bool full_range = false) : m_w(w), m_h(h)
    {
        cuda_ok(cudaMallocPitch(reinterpret_cast<void**>(&m_y), &m_y_pitch, w + 64, h));
        cuda_ok(cudaMallocPitch(reinterpret_cast<void**>(&m_uv), &m_uv_pitch, w + 64, h / 2));
        REQUIRE(m_y_pitch > w);
        REQUIRE(m_uv_pitch > w);
        cuda_ok(cudaMemset2D(m_y, m_y_pitch, kPadding, m_y_pitch, h));
        cuda_ok(cudaMemset2D(m_uv, m_uv_pitch, kPadding, m_uv_pitch, h / 2));
        m_color.full_range = full_range;
    }
    ~DeviceSurface()
    {
        cudaFree(m_y);
        cudaFree(m_uv);
    }
    DeviceSurface(const DeviceSurface&) = delete;
    DeviceSurface& operator=(const DeviceSurface&) = delete;

    void upload(const HostNv12& img)
    {
        cuda_ok(cudaMemcpy2D(m_y, m_y_pitch, img.y.data(), m_w, m_w, m_h, cudaMemcpyHostToDevice));
        cuda_ok(cudaMemcpy2D(m_uv, m_uv_pitch, img.uv.data(), m_w, m_w, m_h / 2, cudaMemcpyHostToDevice));
    }

    // Copies the visible area out and checks that the row padding was left untouched.
    HostNv12 download() const
    {
        HostNv12 img;
        img.w = m_w;
        img.h = m_h;
        img.y.resize(static_cast<size_t>(m_w) * m_h);
        img.uv.resize(static_cast<size_t>(m_w) * (m_h / 2));
        cuda_ok(cudaMemcpy2D(img.y.data(), m_w, m_y, m_y_pitch, m_w, m_h, cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy2D(img.uv.data(), m_w, m_uv, m_uv_pitch, m_w, m_h / 2, cudaMemcpyDeviceToHost));
        check_padding(m_y, m_y_pitch, m_h);
        check_padding(m_uv, m_uv_pitch, m_h / 2);
        return img;
    }

    Nv12View view() const
    {
        Nv12View v;
        v.y = m_y;
        v.uv = m_uv;
        v.y_pitch = m_y_pitch;
        v.uv_pitch = m_uv_pitch;
        v.geometry = { m_w, m_h };
        v.color = m_color;
        return v;
    }

private:
    void check_padding(const uint8_t* plane, size_t pitch, uint32_t rows) const
    {
        std::vector<uint8_t> pad(pitch - m_w);
        for (uint32_t r = 0; r < rows; ++r)
        {
            cuda_ok(cudaMemcpy(pad.data(), plane + r * pitch + m_w, pad.size(), cudaMemcpyDeviceToHost));
            for (uint8_t b : pad)
                REQUIRE(b == kPadding);
        }
    }

    uint32_t m_w, m_h;
    uint8_t* m_y = nullptr;
    uint8_t* m_uv = nullptr;
    size_t m_y_pitch = 0, m_uv_pitch = 0;
    ColorInfo m_color;
};

// CPU reference of the documented mapping.
float bilinear(const std::vector<uint8_t>& p, int stride, int ofs, int row_bytes, int w, int h, float sx, float sy)
{
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);
    auto at = [&](int x, int y) { return static_cast<float>(p[y * row_bytes + x * stride + ofs]); };
    const float top = (1.0F - fx) * at(x0, y0) + fx * at(x1, y0);
    const float bottom = (1.0F - fx) * at(x0, y1) + fx * at(x1, y1);
    return (1.0F - fy) * top + fy * bottom;
}

uint8_t to_byte(float v)
{
    return static_cast<uint8_t>(std::min(std::max(v + 0.5F, 0.0F), 255.0F));
}

float displaced_x(float x, float y, const SineWarpParams& p)
{
    return x + p.amplitude_px * std::sin(kTwoPi * y * (1.0F / p.wavelength_px) + p.phase_rad);
}

struct Reference
{
    HostNv12 img;
    std::vector<bool> luma_filled; // per luma pixel
    std::vector<bool> chroma_filled; // per chroma sample pair
};

Reference reference_warp(const HostNv12& in, const SineWarpParams& p, bool full_range)
{
    const int w = static_cast<int>(in.w);
    const int h = static_cast<int>(in.h);
    const int cw = w / 2;
    const int ch = h / 2;
    Reference ref;
    ref.img.w = in.w;
    ref.img.h = in.h;
    ref.img.y.resize(in.y.size());
    ref.img.uv.resize(in.uv.size());
    ref.luma_filled.assign(in.y.size(), false);
    ref.chroma_filled.assign(static_cast<size_t>(cw) * ch, false);
    const uint8_t fill = full_range ? 0 : 16;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const float sx = displaced_x(static_cast<float>(x), static_cast<float>(y), p);
            if (sx >= 0.0F && sx <= static_cast<float>(w - 1))
                ref.img.y[y * w + x] = to_byte(bilinear(in.y, 1, 0, w, w, h, sx, static_cast<float>(y)));
            else
            {
                ref.img.y[y * w + x] = fill;
                ref.luma_filled[y * w + x] = true;
            }
        }
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx)
        {
            const float luma_y = 2.0F * static_cast<float>(cy) + 0.5F;
            const float scx = 0.5F * displaced_x(2.0F * static_cast<float>(cx), luma_y, p);
            const float scy = 0.5F * (luma_y - 0.5F);
            if (scx < 0.0F || scx > static_cast<float>(cw - 1))
            {
                ref.img.uv[cy * w + 2 * cx] = 128;
                ref.img.uv[cy * w + 2 * cx + 1] = 128;
                ref.chroma_filled[cy * cw + cx] = true;
            }
            else
            {
                ref.img.uv[cy * w + 2 * cx] = to_byte(bilinear(in.uv, 2, 0, w, cw, ch, scx, scy));
                ref.img.uv[cy * w + 2 * cx + 1] = to_byte(bilinear(in.uv, 2, 1, w, cw, ch, scx, scy));
            }
        }
    return ref;
}

void run_once(SineWarp& op, const DeviceSurface& in, const DeviceSurface& out)
{
    cudaStream_t s = nullptr;
    cuda_ok(cudaStreamCreate(&s));
    op.enqueue(in.view(), out.view(), s);
    cuda_ok(cudaStreamSynchronize(s));
    cuda_ok(cudaStreamDestroy(s));
}

// Compares two images within `tol` levels; returns the worst difference.
int max_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    REQUIRE(a.size() == b.size());
    int worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    return worst;
}

void check_against_reference(uint32_t w, uint32_t h, const SineWarpParams& params, bool full_range = false)
{
    const HostNv12 src = make_image(w, h);
    DeviceSurface in(w, h, full_range), out(w, h, full_range);
    in.upload(src);
    SineWarp op(params);
    op.prepare({ w, h });
    run_once(op, in, out);
    const HostNv12 got = out.download();
    const Reference ref = reference_warp(src, params, full_range);
    CHECK(max_diff(got.y, ref.img.y) <= 1);
    CHECK(max_diff(got.uv, ref.img.uv) <= 1);
    // Border-filled pixels are exact.
    for (size_t i = 0; i < ref.luma_filled.size(); ++i)
        if (ref.luma_filled[i])
            REQUIRE(got.y[i] == (full_range ? 0 : 16));
    const size_t cw = w / 2;
    for (size_t i = 0; i < ref.chroma_filled.size(); ++i)
        if (ref.chroma_filled[i])
        {
            const size_t row = i / cw, col = i % cw;
            REQUIRE(got.uv[row * w + 2 * col] == 128);
            REQUIRE(got.uv[row * w + 2 * col + 1] == 128);
        }
}

struct Size
{
    uint32_t w, h;
};

} // namespace

#define REQUIRE_CUDA_OR_SKIP()                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!have_cuda_device())                                                                                       \
            SKIP("No CUDA device available");                                                                          \
    } while (false)

TEST_CASE("SineWarp zero amplitude is a bit-exact identity", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    for (const Size sz : { Size{ 64, 48 }, Size{ 130, 66 } })
    {
        SECTION(std::to_string(sz.w) + "x" + std::to_string(sz.h))
        {
            const HostNv12 src = make_image(sz.w, sz.h);
            DeviceSurface in(sz.w, sz.h), out(sz.w, sz.h);
            in.upload(src);
            SineWarp op(SineWarpParams{ 0.0F, 37.0F, 0.7F });
            op.prepare({ sz.w, sz.h });
            run_once(op, in, out);
            const HostNv12 got = out.download();
            CHECK(got.y == src.y);
            CHECK(got.uv == src.uv);
        }
    }
}

TEST_CASE("SineWarp matches the CPU reference", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    for (const Size sz : { Size{ 64, 48 }, Size{ 130, 66 } })
    {
        SECTION("default-ish parameters " + std::to_string(sz.w) + "x" + std::to_string(sz.h))
        {
            check_against_reference(sz.w, sz.h, SineWarpParams{ 5.0F, 32.0F, 0.0F });
        }
        SECTION("phase and wavelength " + std::to_string(sz.w) + "x" + std::to_string(sz.h))
        {
            check_against_reference(sz.w, sz.h, SineWarpParams{ 3.5F, 21.0F, 1.1F });
        }
        SECTION("large amplitude leaves the image " + std::to_string(sz.w) + "x" + std::to_string(sz.h))
        {
            check_against_reference(sz.w, sz.h, SineWarpParams{ 40.0F, 24.0F, 0.4F });
        }
        SECTION("full-range border fill " + std::to_string(sz.w) + "x" + std::to_string(sz.h))
        {
            check_against_reference(sz.w, sz.h, SineWarpParams{ 40.0F, 24.0F, 0.4F }, true);
        }
    }
}

TEST_CASE("SineWarp large amplitude actually produces border fill", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    const SineWarpParams params{ 40.0F, 24.0F, 0.4F };
    const HostNv12 src = make_image(64, 48);
    const Reference ref = reference_warp(src, params, false);
    CHECK(std::count(ref.luma_filled.begin(), ref.luma_filled.end(), true) > 0);
    CHECK(std::count(ref.chroma_filled.begin(), ref.chroma_filled.end(), true) > 0);
}

TEST_CASE("SineWarp keeps constant chroma constant in the interior", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    const uint32_t w = 130, h = 66;
    HostNv12 src = make_image(w, h);
    for (uint32_t cy = 0; cy < h / 2; ++cy)
        for (uint32_t cx = 0; cx < w / 2; ++cx)
        {
            src.uv[cy * w + 2 * cx] = 90;
            src.uv[cy * w + 2 * cx + 1] = 200;
        }
    const SineWarpParams params{ 4.0F, 30.0F, 0.3F };
    DeviceSurface in(w, h), out(w, h);
    in.upload(src);
    SineWarp op(params);
    op.prepare({ w, h });
    run_once(op, in, out);
    const HostNv12 got = out.download();
    const Reference ref = reference_warp(src, params, false);
    size_t interior = 0;
    for (uint32_t cy = 0; cy < h / 2; ++cy)
        for (uint32_t cx = 0; cx < w / 2; ++cx)
        {
            if (ref.chroma_filled[cy * (w / 2) + cx])
                continue;
            ++interior;
            REQUIRE(got.uv[cy * w + 2 * cx] == 90);
            REQUIRE(got.uv[cy * w + 2 * cx + 1] == 200);
        }
    CHECK(interior > 0);
}

TEST_CASE("SineWarp chroma follows the siting, not luma/2", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    // Vertical chroma edge: U steps from 40 to 220 at chroma column 16. Compared with the reference the output
    // pins the half-luma-row siting of the displacement.
    const uint32_t w = 64, h = 48;
    HostNv12 src = make_image(w, h);
    for (uint32_t cy = 0; cy < h / 2; ++cy)
        for (uint32_t cx = 0; cx < w / 2; ++cx)
        {
            src.uv[cy * w + 2 * cx] = cx < 16 ? 40 : 220;
            src.uv[cy * w + 2 * cx + 1] = 128;
        }
    const SineWarpParams params{ 6.0F, 16.0F, 0.0F };
    DeviceSurface in(w, h), out(w, h);
    in.upload(src);
    SineWarp op(params);
    op.prepare({ w, h });
    run_once(op, in, out);
    const HostNv12 got = out.download();
    const Reference ref = reference_warp(src, params, false);
    CHECK(max_diff(got.uv, ref.img.uv) <= 1);

    // The naive mapping (sample at luma_y = 2*cy, no half-row offset) must differ somewhere on this edge.
    SineWarpParams p = params;
    int naive_worst = 0;
    for (uint32_t cy = 0; cy < h / 2; ++cy)
        for (uint32_t cx = 0; cx < w / 2; ++cx)
        {
            const float scx = 0.5F * displaced_x(2.0F * static_cast<float>(cx), 2.0F * static_cast<float>(cy), p);
            if (scx < 0.0F || scx > static_cast<float>(w / 2 - 1))
                continue;
            const uint8_t naive = to_byte(bilinear(src.uv, 2, 0, w, w / 2, h / 2, scx, static_cast<float>(cy)));
            naive_worst =
                std::max(naive_worst, std::abs(static_cast<int>(naive) - static_cast<int>(got.uv[cy * w + 2 * cx])));
        }
    CHECK(naive_worst > 1);
}

TEST_CASE("SineWarp contract violations throw", "[video_stage][gpu]")
{
    SECTION("non-positive wavelength throws in the constructor")
    {
        CHECK_THROWS_AS(SineWarp(SineWarpParams{ 1.0F, 0.0F, 0.0F }), std::invalid_argument);
        CHECK_THROWS_AS(SineWarp(SineWarpParams{ 1.0F, -5.0F, 0.0F }), std::invalid_argument);
    }
    SECTION("prepare rejects odd or zero dimensions")
    {
        SineWarp op;
        CHECK_THROWS_AS(op.prepare({ 0, 48 }), std::invalid_argument);
        CHECK_THROWS_AS(op.prepare({ 64, 0 }), std::invalid_argument);
        CHECK_THROWS_AS(op.prepare({ 63, 48 }), std::invalid_argument);
        CHECK_THROWS_AS(op.prepare({ 64, 47 }), std::invalid_argument);
    }

    REQUIRE_CUDA_OR_SKIP();
    DeviceSurface in(64, 48), out(64, 48);
    SineWarp op;

    SECTION("enqueue before prepare throws std::logic_error")
    {
        CHECK_THROWS_AS(op.enqueue(in.view(), out.view(), nullptr), std::logic_error);
    }
    SECTION("enqueue with a different input geometry throws std::logic_error")
    {
        op.prepare({ 128, 96 });
        CHECK_THROWS_AS(op.enqueue(in.view(), out.view(), nullptr), std::logic_error);
    }

    op.prepare({ 64, 48 });

    SECTION("mismatched output geometry")
    {
        DeviceSurface other(32, 48);
        CHECK_THROWS_AS(op.enqueue(in.view(), other.view(), nullptr), std::invalid_argument);
    }
    SECTION("null planes")
    {
        Nv12View bad = out.view();
        bad.y = nullptr;
        CHECK_THROWS_AS(op.enqueue(in.view(), bad, nullptr), std::invalid_argument);
        bad = out.view();
        bad.uv = nullptr;
        CHECK_THROWS_AS(op.enqueue(in.view(), bad, nullptr), std::invalid_argument);
        bad = in.view();
        bad.y = nullptr;
        CHECK_THROWS_AS(op.enqueue(bad, out.view(), nullptr), std::invalid_argument);
    }
    SECTION("pitch below the row size")
    {
        Nv12View bad = out.view();
        bad.y_pitch = 63;
        CHECK_THROWS_AS(op.enqueue(in.view(), bad, nullptr), std::invalid_argument);
        bad = out.view();
        bad.uv_pitch = 63;
        CHECK_THROWS_AS(op.enqueue(in.view(), bad, nullptr), std::invalid_argument);
        bad = in.view();
        bad.y_pitch = 10;
        CHECK_THROWS_AS(op.enqueue(bad, out.view(), nullptr), std::invalid_argument);
    }
}

TEST_CASE("SineWarp enqueue is asynchronous and correct after the caller syncs", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    const SineWarpParams params{ 5.0F, 32.0F, 0.2F };
    const HostNv12 src = make_image(130, 66);
    DeviceSurface in(130, 66), out(130, 66);
    in.upload(src);
    SineWarp op(params);
    op.prepare({ 130, 66 });
    cudaStream_t s = nullptr;
    cuda_ok(cudaStreamCreate(&s));
    cudaEvent_t done = nullptr;
    cuda_ok(cudaEventCreate(&done));
    op.enqueue(in.view(), out.view(), s);
    cuda_ok(cudaEventRecord(done, s));
    cuda_ok(cudaEventSynchronize(done));
    const HostNv12 got = out.download();
    const Reference ref = reference_warp(src, params, false);
    CHECK(max_diff(got.y, ref.img.y) <= 1);
    CHECK(max_diff(got.uv, ref.img.uv) <= 1);
    cuda_ok(cudaEventDestroy(done));
    cuda_ok(cudaStreamDestroy(s));
}

TEST_CASE("SineWarp prepared operation is reusable across surfaces and streams", "[video_stage][gpu]")
{
    REQUIRE_CUDA_OR_SKIP();
    const SineWarpParams params{ 5.0F, 32.0F, 0.2F };
    const uint32_t w = 130, h = 66;
    const HostNv12 a = make_image(w, h);
    HostNv12 b = make_image(w, h);
    for (auto& v : b.y)
        v = static_cast<uint8_t>(255 - v);
    for (auto& v : b.uv)
        v = static_cast<uint8_t>(255 - v);

    DeviceSurface in_a(w, h), in_b(w, h), out_a(w, h), out_b(w, h);
    in_a.upload(a);
    in_b.upload(b);
    SineWarp op(params);
    op.prepare({ w, h });
    const Reference ref_a = reference_warp(a, params, false);
    const Reference ref_b = reference_warp(b, params, false);

    SECTION("two surfaces back-to-back on one stream")
    {
        cudaStream_t s = nullptr;
        cuda_ok(cudaStreamCreate(&s));
        op.enqueue(in_a.view(), out_a.view(), s);
        op.enqueue(in_b.view(), out_b.view(), s);
        cuda_ok(cudaStreamSynchronize(s));
        cuda_ok(cudaStreamDestroy(s));
    }
    SECTION("two surfaces on two streams")
    {
        cudaStream_t s1 = nullptr, s2 = nullptr;
        cuda_ok(cudaStreamCreate(&s1));
        cuda_ok(cudaStreamCreate(&s2));
        op.enqueue(in_a.view(), out_a.view(), s1);
        op.enqueue(in_b.view(), out_b.view(), s2);
        cuda_ok(cudaStreamSynchronize(s1));
        cuda_ok(cudaStreamSynchronize(s2));
        cuda_ok(cudaStreamDestroy(s1));
        cuda_ok(cudaStreamDestroy(s2));
    }
    const HostNv12 got_a = out_a.download();
    const HostNv12 got_b = out_b.download();
    CHECK(max_diff(got_a.y, ref_a.img.y) <= 1);
    CHECK(max_diff(got_a.uv, ref_a.img.uv) <= 1);
    CHECK(max_diff(got_b.y, ref_b.img.y) <= 1);
    CHECK(max_diff(got_b.uv, ref_b.img.uv) <= 1);
}
