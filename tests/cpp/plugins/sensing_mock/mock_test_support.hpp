// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <core/sipl_camera.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
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
        ::setenv(name, value, 1);
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

/// Reads the 16-bit frame counter the mock paints top-left: 16 px cells, most significant bit first.
inline uint32_t read_counter(const std::vector<uint8_t>& rgba, uint32_t width)
{
    uint32_t value = 0;
    for (int bit = 0; bit < 16; ++bit)
    {
        const size_t x = 8 + 16 * static_cast<size_t>(bit);
        const size_t y = 8;
        const uint8_t* px = &rgba[(y * width + x) * 4];
        const bool on = px[0] > 127 && px[1] > 127 && px[2] > 127;
        value = (value << 1) | (on ? 1u : 0u);
    }
    return value;
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
