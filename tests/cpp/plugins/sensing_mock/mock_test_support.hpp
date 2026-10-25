// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gif_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <core/mock_test_hooks.hpp>
#include <core/sipl_camera.hpp>

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

/// One NAL unit of the mock encoder's Annex-B output.
struct Nal
{
    uint8_t type = 0;
    std::vector<uint8_t> payload;
};

inline std::vector<Nal> split_annex_b(const std::vector<uint8_t>& data)
{
    std::vector<Nal> nals;
    for (size_t i = 0; i + 4 < data.size();)
    {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1)
        {
            Nal nal;
            nal.type = data[i + 4];
            size_t end = i + 5;
            while (end + 3 < data.size() &&
                   !(data[end] == 0 && data[end + 1] == 0 && data[end + 2] == 0 && data[end + 3] == 1))
                ++end;
            if (end + 3 >= data.size())
                end = data.size();
            nal.payload.assign(data.begin() + i + 5, data.begin() + end);
            nals.push_back(std::move(nal));
            i = end;
        }
        else
        {
            ++i;
        }
    }
    return nals;
}

/// The frame number the mock records after the "MOCK" magic, as 8 hex-like letters.
inline uint32_t frame_of(const Nal& slice)
{
    REQUIRE(slice.payload.size() >= 12);
    REQUIRE(std::string(slice.payload.begin(), slice.payload.begin() + 4) == "MOCK");
    uint32_t value = 0;
    for (int i = 0; i < 8; ++i)
        value = (value << 4) | static_cast<uint32_t>(slice.payload[4 + i] - 'A');
    return value;
}

inline bool is_slice(const Nal& nal)
{
    return nal.type == 0x65 || nal.type == 0x41;
}

} // namespace sensing_test
