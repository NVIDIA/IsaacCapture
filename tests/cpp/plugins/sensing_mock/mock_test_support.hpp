// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gif_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <core/mock_test_hooks.hpp>
#include <core/sipl_camera.hpp>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace sensing_test
{

/// Sets an environment variable for one scope; the mock reads its knobs at construction.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value) : m_name(name)
    {
        set(value);
    }
    /// Names the variable now; the value is set later with set().
    explicit ScopedEnv(const char* name) : m_name(name)
    {
    }
    void set(const char* value)
    {
        ::setenv(m_name.c_str(), value, 1);
    }
    ~ScopedEnv()
    {
        ::unsetenv(m_name.c_str());
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string m_name;
};

/// Every sensor shows this GIF as it is for one scope.
class ScopedPicture
{
public:
    explicit ScopedPicture(const std::string& path)
    {
        plugins::sensing::mock_set_test_picture(path);
    }
    ~ScopedPicture()
    {
        plugins::sensing::mock_set_test_picture("");
    }
    ScopedPicture(const ScopedPicture&) = delete;
    ScopedPicture& operator=(const ScopedPicture&) = delete;
};

template <typename Pred>
bool wait_for(Pred done, std::chrono::milliseconds limit = std::chrono::seconds(5))
{
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!done())
    {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
}

inline plugins::sensing::SiplConfig make_sipl_config()
{
    plugins::sensing::SiplConfig config;
    config.platform_config_json = SENSING_PLATFORM_CONFIG;
    config.platform_config_name = "SHW5G_2";
    config.link_masks = { 0x0000, 0x1100 };
    return config;
}

inline std::string unique_path(const std::string& stem, const std::string& extension)
{
    static int counter = 0;
    return "/tmp/" + stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + extension;
}

/// A path that is removed when the scope ends, however the test exits.
class TempFile
{
public:
    explicit TempFile(const std::string& extension) : m_path(unique_path("sensing_mock", extension))
    {
    }
    ~TempFile()
    {
        std::remove(m_path.c_str());
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::string& path() const
    {
        return m_path;
    }

private:
    std::string m_path;
};

/// The mock at test size, showing a short GIF that cycles black, red, blue and white every 40 ms.
/// Small frames keep tests fast, and solid frames of known colours show whether a frame is whole.
class FastMock
{
public:
    FastMock() : m_gif(".gif"), m_scale("SENSING_MOCK_SCALE_DIV", "8")
    {
        write_gif(m_gif.path(), 16, 16,
                  { solid_frame(16, 16, kGifBlack, 4), solid_frame(16, 16, kGifRed, 4),
                    solid_frame(16, 16, kGifBlue, 4), solid_frame(16, 16, kGifWhite, 4) });
        m_source = std::make_unique<ScopedPicture>(m_gif.path());
    }

private:
    TempFile m_gif;
    ScopedEnv m_scale;
    std::unique_ptr<ScopedPicture> m_source;
};

struct Rgb
{
    int r;
    int g;
    int b;
};

/// One frame copied to the host.
struct Shot
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t sequence = 0;
    std::vector<uint8_t> rgba;

    Rgb at(uint32_t x, uint32_t y) const
    {
        const uint8_t* px = &rgba[(static_cast<size_t>(y) * width + x) * 4];
        return { px[0], px[1], px[2] };
    }
};

inline bool near(const Rgb& c, int r, int g, int b, int tolerance = 12)
{
    return std::abs(c.r - r) <= tolerance && std::abs(c.g - g) <= tolerance && std::abs(c.b - b) <= tolerance;
}

/// Black, red, blue or white: the colours of the frames FastMock plays.
inline bool is_cycle_colour(const Rgb& c)
{
    return near(c, 0, 0, 0) || near(c, 255, 0, 0) || near(c, 0, 0, 255) || near(c, 255, 255, 255, 14);
}

/// True when the corners and centre agree, as in a frame of one solid colour. A frame torn between two
/// captures would show two.
inline bool is_whole(const Shot& shot)
{
    const Rgb centre = shot.at(shot.width / 2, shot.height / 2);
    for (const Rgb& corner : { shot.at(0, 0), shot.at(shot.width - 1, 0), shot.at(0, shot.height - 1),
                               shot.at(shot.width - 1, shot.height - 1) })
    {
        if (!near(corner, centre.r, centre.g, centre.b, 14))
            return false;
    }
    return is_cycle_colour(centre);
}

/// Copies a device RGBA frame to the host as tightly packed rows.
inline std::vector<uint8_t> download_rgba(uintptr_t device_ptr, size_t pitch, uint32_t width, uint32_t height)
{
    std::vector<uint8_t> host(static_cast<size_t>(width) * height * 4);
    const cudaError_t err =
        cudaMemcpy2D(host.data(), static_cast<size_t>(width) * 4, reinterpret_cast<const void*>(device_ptr), pitch,
                     static_cast<size_t>(width) * 4, height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpy2D: ") + cudaGetErrorString(err));
    return host;
}

/// Waits for one frame from `sensor` and copies it to the host.
inline Shot grab(plugins::sensing::SiplCamera& camera, uint32_t sensor)
{
    Shot shot;
    REQUIRE(wait_for(
        [&]
        {
            auto view = camera.latest(sensor);
            if (!view)
                return false;
            shot.width = view->width;
            shot.height = view->height;
            shot.sequence = view->sequence;
            shot.rgba = download_rgba(view->ptr, view->pitch, view->width, view->height);
            return true;
        }));
    return shot;
}

/// One NAL unit of an Annex-B stream. `type` is the nal_unit_type (7 SPS, 8 PPS, 5 IDR slice, 1 other
/// slice); `payload` follows the one-byte NAL header.
struct Nal
{
    uint8_t type = 0;
    std::vector<uint8_t> payload;
};

constexpr uint8_t kNalSlice = 1;
constexpr uint8_t kNalIdr = 5;
constexpr uint8_t kNalSei = 6;
constexpr uint8_t kNalSps = 7;
constexpr uint8_t kNalPps = 8;

inline std::vector<Nal> split_annex_b(const std::vector<uint8_t>& data)
{
    // Offsets just past each 00 00 01 start code.
    std::vector<size_t> starts;
    for (size_t i = 0; i + 3 <= data.size(); ++i)
    {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1)
            starts.push_back(i + 3);
    }
    std::vector<Nal> nals;
    for (size_t n = 0; n < starts.size(); ++n)
    {
        size_t end = n + 1 < starts.size() ? starts[n + 1] - 3 : data.size();
        while (end > starts[n] && data[end - 1] == 0 && n + 1 < starts.size())
            --end; // the zero of a four-byte start code
        if (starts[n] >= end)
            continue;
        Nal nal;
        nal.type = data[starts[n]] & 0x1F;
        nal.payload.assign(data.begin() + starts[n] + 1, data.begin() + end);
        nals.push_back(std::move(nal));
    }
    return nals;
}

inline bool is_slice(const Nal& nal)
{
    return nal.type == kNalSlice || nal.type == kNalIdr;
}

/// One decoded picture: its size and the mean of its luma plane.
struct Decoded
{
    int width = 0;
    int height = 0;
    double luma = 0;
};

/// Decodes an Annex-B stream with libavcodec's software H.264 decoder, in output order.
inline std::vector<Decoded> decode_annex_b(const std::vector<uint8_t>& data)
{
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    REQUIRE(codec != nullptr);
    AVCodecContext* context = avcodec_alloc_context3(codec);
    AVCodecParserContext* parser = av_parser_init(AV_CODEC_ID_H264);
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    REQUIRE((context && parser && packet && frame));
    REQUIRE(avcodec_open2(context, codec, nullptr) == 0);

    std::vector<Decoded> out;
    auto drain = [&]
    {
        while (avcodec_receive_frame(context, frame) == 0)
        {
            Decoded picture;
            picture.width = frame->width;
            picture.height = frame->height;
            double sum = 0;
            for (int y = 0; y < frame->height; ++y)
            {
                const uint8_t* row = frame->data[0] + static_cast<size_t>(y) * frame->linesize[0];
                for (int x = 0; x < frame->width; ++x)
                    sum += row[x];
            }
            picture.luma = sum / (static_cast<double>(frame->width) * frame->height);
            out.push_back(picture);
        }
    };
    size_t offset = 0;
    while (offset < data.size())
    {
        const int used = av_parser_parse2(parser, context, &packet->data, &packet->size, data.data() + offset,
                                          static_cast<int>(data.size() - offset), AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        REQUIRE(used >= 0);
        offset += static_cast<size_t>(used);
        if (packet->size > 0)
        {
            REQUIRE(avcodec_send_packet(context, packet) == 0);
            drain();
        }
    }
    // The parser holds the last NAL until it sees what follows; flush it.
    av_parser_parse2(parser, context, &packet->data, &packet->size, nullptr, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
    if (packet->size > 0)
    {
        REQUIRE(avcodec_send_packet(context, packet) == 0);
        drain();
    }
    avcodec_send_packet(context, nullptr);
    drain();

    av_frame_free(&frame);
    av_packet_free(&packet);
    av_parser_close(parser);
    avcodec_free_context(&context);
    return out;
}

} // namespace sensing_test
