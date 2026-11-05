// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_gif_source.hpp"

#include <cuda_runtime.h>

#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stb_image.h>
#include <stdexcept>

namespace plugins
{
namespace sensing
{

namespace
{

constexpr int kMinDelayMs = 20;

} // namespace

std::vector<double> gif_frame_ends(const std::vector<int>& delays_ms)
{
    std::vector<double> ends;
    ends.reserve(delays_ms.size());
    double total = 0.0;
    for (int delay : delays_ms)
    {
        total += std::max(delay, kMinDelayMs) / 1000.0;
        ends.push_back(total);
    }
    return ends;
}

size_t gif_frame_index(const std::vector<double>& frame_ends, double seconds)
{
    if (frame_ends.empty())
        return 0;
    const double t = std::fmod(std::max(seconds, 0.0), frame_ends.back());
    const auto it = std::upper_bound(frame_ends.begin(), frame_ends.end(), t);
    return std::min<size_t>(static_cast<size_t>(it - frame_ends.begin()), frame_ends.size() - 1);
}

MockGifSource::MockGifSource(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("mock GIF: cannot open " + path);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    int width = 0;
    int height = 0;
    int frames = 0;
    int channels = 0;
    int* delays = nullptr;
    unsigned char* pixels = stbi_load_gif_from_memory(
        bytes.data(), static_cast<int>(bytes.size()), &delays, &width, &height, &frames, &channels, 4);
    if (!pixels)
        throw std::runtime_error("mock GIF: cannot decode " + path + ": " + stbi_failure_reason());

    // Composite onto the background, so the kernel only ever sees opaque pixels.
    const size_t pixel_count = static_cast<size_t>(width) * height * frames;
    for (size_t i = 0; i < pixel_count; ++i)
    {
        unsigned char* px = pixels + i * 4;
        const int alpha = px[3];
        for (int c = 0; c < 3; ++c)
            px[c] = static_cast<unsigned char>((px[c] * alpha + kBackground * (255 - alpha) + 127) / 255);
        px[3] = 255;
    }

    m_width = static_cast<uint32_t>(width);
    m_height = static_cast<uint32_t>(height);
    m_frame_ends = gif_frame_ends(std::vector<int>(delays, delays + frames));

    const size_t bytes_per_frame = static_cast<size_t>(width) * height * 4;
    const cudaError_t alloc = cudaMalloc(reinterpret_cast<void**>(&m_device_frames), bytes_per_frame * frames);
    const cudaError_t copy = alloc == cudaSuccess ?
                                 cudaMemcpy(m_device_frames, pixels, bytes_per_frame * frames, cudaMemcpyHostToDevice) :
                                 alloc;
    stbi_image_free(pixels);
    stbi_image_free(delays);
    if (copy != cudaSuccess)
    {
        cudaFree(m_device_frames);
        throw std::runtime_error(std::string("mock GIF: cannot upload to the GPU: ") + cudaGetErrorString(copy));
    }
}

MockGifSource::~MockGifSource()
{
    cudaFree(m_device_frames);
}

const uint8_t* MockGifSource::frame(size_t index) const
{
    return m_device_frames + index * static_cast<size_t>(m_width) * m_height * 4;
}

} // namespace sensing
} // namespace plugins
