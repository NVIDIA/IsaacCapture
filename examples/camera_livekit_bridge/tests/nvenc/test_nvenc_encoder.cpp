// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nvenc/nvenc_encoder.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

// NAL unit types found after Annex-B start codes (00 00 01).
std::vector<int> nal_types(const std::vector<uint8_t>& d)
{
    std::vector<int> types;
    for (size_t i = 0; i + 3 < d.size(); ++i)
        if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1)
            types.push_back(d[i + 3] & 0x1F);
    return types;
}

bool has_type(const std::vector<int>& t, int type)
{
    for (int v : t)
        if (v == type)
            return true;
    return false;
}

bool starts_with_start_code(const std::vector<uint8_t>& d)
{
    return (d.size() >= 3 && d[0] == 0 && d[1] == 0 && d[2] == 1) ||
           (d.size() >= 4 && d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 1);
}

void fill_surface(const NvencEncoder::Surface& s, uint8_t luma)
{
    const uint32_t w = s.view.geometry.width;
    const uint32_t h = s.view.geometry.height;
    CHECK(cudaMemset2D(s.view.y, s.view.y_pitch, luma, w, h) == cudaSuccess);
    CHECK(cudaMemset2D(s.view.uv, s.view.uv_pitch, 128, w, h / 2) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

void test_invalid_geometry(CudaContext& ctx)
{
    for (Geometry g : { Geometry{ 641, 480 }, Geometry{ 640, 479 }, Geometry{ 0, 480 }, Geometry{ 640, 0 } })
    {
        EncoderConfig cfg;
        cfg.geometry = g;
        bool threw = false;
        try
        {
            NvencEncoder bad(ctx, cfg);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        catch (const std::exception& e)
        {
            std::fprintf(stderr, "unexpected exception type for %ux%u: %s\n", g.width, g.height, e.what());
        }
        CHECK(threw);
    }
}

void test_pool(NvencEncoder& enc, uint32_t pool_size)
{
    // The pool hands out exactly input_surfaces distinct slots, then reports false until one is released.
    std::vector<NvencEncoder::Surface> held(pool_size);
    for (uint32_t i = 0; i < pool_size; ++i)
    {
        CHECK(enc.acquire(held[i]));
        for (uint32_t j = 0; j < i; ++j)
            CHECK(held[i].slot != held[j].slot);
    }
    NvencEncoder::Surface extra;
    CHECK(!enc.acquire(extra));
    enc.release(held[0]);
    enc.release(held[0]); // idempotent
    CHECK(enc.acquire(extra));
    CHECK(extra.slot == held[0].slot);
    enc.release(extra);
    for (uint32_t i = 1; i < pool_size; ++i)
        enc.release(held[i]);
}

bool ffmpeg_available()
{
    return std::system("command -v ffmpeg >/dev/null 2>&1") == 0;
}

void test_decodable(const std::vector<uint8_t>& idr)
{
    if (!ffmpeg_available())
        return;
    const char* tmp = std::getenv("TMPDIR");
    const std::string path = std::string(tmp ? tmp : "/tmp") + "/test_nvenc_idr_" + std::to_string(::getpid()) + ".h264";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(idr.data()), static_cast<std::streamsize>(idr.size()));
    }
    const int rc = std::system(("ffmpeg -v error -i " + path + " -f null - </dev/null").c_str());
    CHECK(rc == 0);
    std::remove(path.c_str());
}

} // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }

    constexpr uint32_t kPool = 3;
    constexpr uint32_t kIdrPeriod = 30;
    EncoderConfig cfg;
    cfg.geometry = { 640, 480 };
    cfg.fps_numerator = 30;
    cfg.fps_denominator = 1;
    cfg.idr_period = kIdrPeriod;
    cfg.input_surfaces = kPool;

    std::unique_ptr<CudaContext> ctx;
    std::unique_ptr<NvencEncoder> enc;
    try
    {
        ctx = std::make_unique<CudaContext>(0);
        enc = std::make_unique<NvencEncoder>(*ctx, cfg);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "SKIP: cannot create an NVENC encoder here: %s\n", e.what());
        return 77;
    }

    test_invalid_geometry(*ctx);
    test_pool(*enc, kPool);

    std::vector<uint8_t> first_idr;
    constexpr int kFrames = 8;
    for (int i = 0; i < kFrames; ++i)
    {
        NvencEncoder::Surface s;
        CHECK(enc->acquire(s));
        CHECK(s.view.geometry == cfg.geometry);
        CHECK(s.view.y_pitch >= 640);
        fill_surface(s, 100);
        const uint64_t key = 1000 + static_cast<uint64_t>(i);
        EncodedPacket p = enc->encode(s, nullptr, key, i == 0);

        CHECK(!p.data.empty());
        CHECK(starts_with_start_code(p.data));
        CHECK(p.key == key);
        const std::vector<int> types = nal_types(p.data);
        if (i == 0)
        {
            CHECK(p.keyframe);
            CHECK(has_type(types, 7));
            CHECK(has_type(types, 8));
            CHECK(has_type(types, 5));
            first_idr = p.data;
        }
        else
        {
            CHECK(!p.keyframe);
            CHECK(!has_type(types, 5));
            CHECK(has_type(types, 1));
        }
    }

    // encode() released its surface: the whole pool is free again.
    {
        std::vector<NvencEncoder::Surface> all(kPool);
        for (auto& s : all)
            CHECK(enc->acquire(s));
        for (auto& s : all)
            enc->release(s);
    }

    // A later forced IDR carries parameter sets again.
    {
        NvencEncoder::Surface s;
        CHECK(enc->acquire(s));
        fill_surface(s, 100);
        EncodedPacket p = enc->encode(s, nullptr, 77, true);
        const std::vector<int> types = nal_types(p.data);
        CHECK(p.keyframe);
        CHECK(p.key == 77);
        CHECK(has_type(types, 7) && has_type(types, 8) && has_type(types, 5));
    }

    test_decodable(first_idr);

    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("nvenc tests passed\n");
    return 0;
}
