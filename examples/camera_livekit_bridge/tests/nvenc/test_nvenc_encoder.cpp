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

void fill_noise(const NvencEncoder::Surface& s, uint32_t seed)
{
    const uint32_t w = s.view.geometry.width;
    const uint32_t h = s.view.geometry.height;
    std::vector<uint8_t> luma(static_cast<size_t>(w) * h);
    uint32_t x = seed * 2654435761u + 1u;
    for (uint8_t& v : luma)
    {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v = static_cast<uint8_t>(x >> 24);
    }
    CHECK(cudaMemcpy2D(s.view.y, s.view.y_pitch, luma.data(), w, w, h, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemset2D(s.view.uv, s.view.uv_pitch, 128, w, h / 2) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

EncodedPacket encode_noise(NvencEncoder& enc, uint32_t seed, uint64_t key, bool force_idr)
{
    NvencEncoder::Surface s;
    CHECK(enc.acquire(s));
    fill_noise(s, seed);
    return enc.encode(s, nullptr, key, force_idr);
}

EncodedPacket encode_flat(NvencEncoder& enc, uint64_t key, bool force_idr)
{
    NvencEncoder::Surface s;
    CHECK(enc.acquire(s));
    fill_surface(s, 100);
    return enc.encode(s, nullptr, key, force_idr);
}

void test_bitrate_getter(NvencEncoder& enc, uint32_t configured)
{
    CHECK(enc.bitrate() == configured);
    enc.set_bitrate(1'000'000);
    CHECK(enc.bitrate() == 1'000'000);
    enc.set_bitrate(6'000'000);
    CHECK(enc.bitrate() == 6'000'000);
}

void test_zero_bitrate_rejected(NvencEncoder& enc)
{
    const uint32_t before = enc.bitrate();
    bool threw = false;
    try
    {
        enc.set_bitrate(0);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    CHECK(threw);
    CHECK(enc.bitrate() == before);
}

void test_output_after_reconfigure(NvencEncoder& enc)
{
    // Start from an IDR so the following pictures are P-frames.
    CHECK(encode_flat(enc, 500, true).keyframe);
    enc.set_bitrate(2'000'000);
    for (int i = 0; i < 4; ++i)
    {
        const uint64_t key = 501 + static_cast<uint64_t>(i);
        EncodedPacket p = encode_flat(enc, key, false);
        CHECK(!p.data.empty());
        CHECK(starts_with_start_code(p.data));
        CHECK(p.key == key);
        CHECK(!p.keyframe);
        const std::vector<int> types = nal_types(p.data);
        CHECK(has_type(types, 1));
        CHECK(!has_type(types, 5));
    }
    EncodedPacket idr = encode_flat(enc, 600, true);
    const std::vector<int> types = nal_types(idr.data);
    CHECK(idr.keyframe);
    CHECK(idr.key == 600);
    CHECK(has_type(types, 7) && has_type(types, 8) && has_type(types, 5));
}

double mean_noise_size(NvencEncoder& enc, int frames, int skip, uint32_t& seed)
{
    double total = 0;
    int counted = 0;
    for (int i = 0; i < frames; ++i)
    {
        EncodedPacket p = encode_noise(enc, seed++, 700 + static_cast<uint64_t>(seed), false);
        CHECK(!p.data.empty());
        if (i >= skip)
        {
            total += static_cast<double>(p.data.size());
            ++counted;
        }
    }
    return counted ? total / counted : 0.0;
}

void test_rate_changes_output_size(NvencEncoder& enc)
{
    uint32_t seed = 1;
    CHECK(encode_noise(enc, seed++, 650, true).keyframe);
    enc.set_bitrate(6'000'000);
    const double high = mean_noise_size(enc, 30, 5, seed);
    enc.set_bitrate(500'000);
    const double low = mean_noise_size(enc, 30, 5, seed);
    std::printf("noise mean packet size: high=%.0f B low=%.0f B ratio=%.3f\n", high, low, high > 0 ? low / high : 0.0);
    CHECK(high > 0);
    CHECK(low < 0.3 * high);
}

void test_decodable_across_reconfigure(NvencEncoder& enc)
{
    if (!ffmpeg_available())
        return;
    std::vector<uint8_t> stream;
    auto append = [&stream](const EncodedPacket& p) { stream.insert(stream.end(), p.data.begin(), p.data.end()); };
    uint32_t seed = 5000;
    enc.set_bitrate(4'000'000);
    append(encode_noise(enc, seed++, 800, true));
    for (int i = 0; i < 5; ++i)
        append(encode_noise(enc, seed++, 801 + static_cast<uint64_t>(i), false));
    enc.set_bitrate(800'000);
    for (int i = 0; i < 6; ++i)
        append(encode_noise(enc, seed++, 810 + static_cast<uint64_t>(i), false));
    enc.set_bitrate(3'000'000);
    for (int i = 0; i < 6; ++i)
        append(encode_noise(enc, seed++, 820 + static_cast<uint64_t>(i), false));

    const char* tmp = std::getenv("TMPDIR");
    const std::string path =
        std::string(tmp ? tmp : "/tmp") + "/test_nvenc_reconfig_" + std::to_string(::getpid()) + ".h264";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(stream.data()), static_cast<std::streamsize>(stream.size()));
    }
    const int rc = std::system(("ffmpeg -v error -i " + path + " -f null - </dev/null").c_str());
    CHECK(rc == 0);
    std::remove(path.c_str());
}

// MSB-first reader over an SPS RBSP (emulation-prevention bytes already removed).
struct BitReader
{
    const std::vector<uint8_t>& d;
    size_t pos = 0; // bit position
    bool ok = true;

    uint32_t u(int n)
    {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i, ++pos)
        {
            if ((pos >> 3) >= d.size())
            {
                ok = false;
                return 0;
            }
            v = (v << 1) | ((d[pos >> 3] >> (7 - (pos & 7))) & 1u);
        }
        return v;
    }

    uint32_t ue()
    {
        int zeros = 0;
        while (ok && u(1) == 0 && zeros < 32)
            ++zeros;
        return ok ? (1u << zeros) - 1u + u(zeros) : 0;
    }
};

// Returns video_full_range_flag from the first SPS in an Annex-B access unit; -1 if absent, unparsable, or the
// VUI carries no video_signal_type.
int sps_video_full_range(const std::vector<uint8_t>& au)
{
    for (size_t i = 0; i + 4 < au.size(); ++i)
    {
        if (!(au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1 && (au[i + 3] & 0x1F) == 7))
            continue;
        // RBSP after the NAL header, up to the next start code, with 00 00 03 emulation bytes dropped.
        std::vector<uint8_t> rbsp;
        for (size_t j = i + 4; j < au.size(); ++j)
        {
            if (j + 2 < au.size() && au[j] == 0 && au[j + 1] == 0 && au[j + 2] <= 1)
                break;
            if (rbsp.size() >= 2 && rbsp[rbsp.size() - 1] == 0 && rbsp[rbsp.size() - 2] == 0 && au[j] == 3)
                continue;
            rbsp.push_back(au[j]);
        }
        BitReader r{ rbsp };
        const uint32_t profile = r.u(8);
        r.u(16); // constraint flags + level_idc
        r.ue(); // seq_parameter_set_id
        constexpr uint32_t kHighProfiles[] = { 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135 };
        for (uint32_t hp : kHighProfiles)
        {
            if (profile != hp)
                continue;
            if (r.ue() == 3) // chroma_format_idc
                r.u(1);
            r.ue();
            r.ue();
            r.u(1);
            if (r.u(1)) // seq_scaling_matrix_present_flag: not produced by NVENC, not parsed here
                return -1;
        }
        r.ue(); // log2_max_frame_num_minus4
        const uint32_t poc_type = r.ue();
        if (poc_type == 0)
            r.ue();
        else if (poc_type == 1)
            return -1;
        r.ue(); // max_num_ref_frames
        r.u(1);
        r.ue();
        r.ue();
        if (!r.u(1)) // frame_mbs_only_flag
            r.u(1);
        r.u(1); // direct_8x8_inference_flag
        if (r.u(1)) // frame_cropping_flag
            for (int k = 0; k < 4; ++k)
                r.ue();
        if (!r.u(1)) // vui_parameters_present_flag
            return -1;
        if (r.u(1)) // aspect_ratio_info_present_flag
            if (r.u(8) == 255)
                r.u(32);
        if (r.u(1)) // overscan_info_present_flag
            r.u(1);
        if (!r.u(1)) // video_signal_type_present_flag
            return -1;
        r.u(3); // video_format
        const uint32_t full = r.u(1);
        return r.ok ? static_cast<int>(full) : -1;
    }
    return -1;
}

// A grayscale ramp (Y 0..255 across the width, UV 128) must be signalled as full or limited range as configured.
void test_range_signalling(CudaContext& ctx, bool full_range)
{
    EncoderConfig cfg;
    cfg.geometry = { 640, 480 };
    cfg.full_range = full_range;
    cfg.input_surfaces = 2;
    NvencEncoder enc(ctx, cfg);

    std::vector<uint8_t> luma(static_cast<size_t>(640) * 480);
    for (size_t i = 0; i < luma.size(); ++i)
        luma[i] = static_cast<uint8_t>((i % 640) * 255 / 639);

    int flag = -2;
    for (int i = 0; i < 3; ++i)
    {
        NvencEncoder::Surface s;
        CHECK(enc.acquire(s));
        CHECK(cudaMemcpy2D(s.view.y, s.view.y_pitch, luma.data(), 640, 640, 480, cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemset2D(s.view.uv, s.view.uv_pitch, 128, 640, 240) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        EncodedPacket p = enc.encode(s, nullptr, 2000 + static_cast<uint64_t>(i), i == 0);
        if (i == 0)
            flag = sps_video_full_range(p.data);
    }
    if (flag != (full_range ? 1 : 0))
        std::fprintf(stderr, "video_full_range_flag=%d for full_range=%d\n", flag, full_range ? 1 : 0);
    CHECK(flag == (full_range ? 1 : 0));
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
    test_bitrate_getter(*enc, cfg.bitrate_bps);
    test_zero_bitrate_rejected(*enc);

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
    test_output_after_reconfigure(*enc);
    test_rate_changes_output_size(*enc);
    test_decodable_across_reconfigure(*enc);
    test_range_signalling(*ctx, true);
    test_range_signalling(*ctx, false);

    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("nvenc tests passed\n");
    return 0;
}
