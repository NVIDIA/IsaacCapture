// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A minimal GIF89a writer, so tests can make animated GIFs of known content.

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace sensing_test
{

/// Palette indices a fixture frame can use.
enum GifColor : uint8_t
{
    kGifBlack = 0,
    kGifRed = 1,
    kGifBlue = 2,
    kGifWhite = 3,
};

struct GifFrame
{
    /// width * height palette indices, row by row.
    std::vector<uint8_t> pixels;
    int delay_centiseconds = 10;
};

namespace detail
{

class LzwBits
{
public:
    void put(uint32_t code, int bits)
    {
        m_buffer |= code << m_count;
        m_count += bits;
        while (m_count >= 8)
        {
            m_bytes.push_back(static_cast<uint8_t>(m_buffer & 0xFF));
            m_buffer >>= 8;
            m_count -= 8;
        }
    }
    std::vector<uint8_t> finish()
    {
        if (m_count > 0)
            m_bytes.push_back(static_cast<uint8_t>(m_buffer & 0xFF));
        return m_bytes;
    }

private:
    std::vector<uint8_t> m_bytes;
    uint32_t m_buffer = 0;
    int m_count = 0;
};

inline void put16(std::string& out, int value)
{
    out.push_back(static_cast<char>(value & 0xFF));
    out.push_back(static_cast<char>((value >> 8) & 0xFF));
}

/// Valid LZW with no compression: each pixel is a literal 9-bit code, and the table is cleared often
/// enough that the code size never has to grow.
inline std::vector<uint8_t> lzw_literal(const std::vector<uint8_t>& pixels)
{
    constexpr uint32_t kClear = 256;
    constexpr uint32_t kEnd = 257;
    constexpr size_t kRun = 250;
    LzwBits bits;
    bits.put(kClear, 9);
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        bits.put(pixels[i], 9);
        if ((i + 1) % kRun == 0)
            bits.put(kClear, 9);
    }
    bits.put(kEnd, 9);
    return bits.finish();
}

} // namespace detail

inline void write_gif(const std::string& path, int width, int height, const std::vector<GifFrame>& frames)
{
    std::string out = "GIF89a";
    detail::put16(out, width);
    detail::put16(out, height);
    out.push_back(static_cast<char>(0xF7)); // global colour table, 256 entries
    out.push_back(0);
    out.push_back(0);

    std::string palette(256 * 3, '\0');
    auto set = [&](int index, int r, int g, int b)
    {
        palette[index * 3] = static_cast<char>(r);
        palette[index * 3 + 1] = static_cast<char>(g);
        palette[index * 3 + 2] = static_cast<char>(b);
    };
    set(kGifBlack, 0, 0, 0);
    set(kGifRed, 255, 0, 0);
    set(kGifBlue, 0, 0, 255);
    set(kGifWhite, 255, 255, 255);
    out += palette;

    for (const auto& frame : frames)
    {
        out += std::string("\x21\xF9\x04\x00", 4);
        detail::put16(out, frame.delay_centiseconds);
        out += std::string("\x00\x00", 2);

        out.push_back(0x2C);
        detail::put16(out, 0);
        detail::put16(out, 0);
        detail::put16(out, width);
        detail::put16(out, height);
        out.push_back(0);

        out.push_back(8); // minimum code size
        const auto data = detail::lzw_literal(frame.pixels);
        for (size_t offset = 0; offset < data.size(); offset += 255)
        {
            const size_t n = std::min<size_t>(255, data.size() - offset);
            out.push_back(static_cast<char>(n));
            out.append(reinterpret_cast<const char*>(data.data()) + offset, n);
        }
        out.push_back(0);
    }
    out.push_back(0x3B);

    std::ofstream file(path, std::ios::binary);
    file.write(out.data(), static_cast<std::streamsize>(out.size()));
}

/// A frame of one colour.
inline GifFrame solid_frame(int width, int height, GifColor color, int delay_centiseconds)
{
    return { std::vector<uint8_t>(static_cast<size_t>(width) * height, color), delay_centiseconds };
}

/// Left half `left`, right half `right`.
inline GifFrame split_frame(int width, int height, GifColor left, GifColor right, int delay_centiseconds)
{
    GifFrame frame{ std::vector<uint8_t>(static_cast<size_t>(width) * height), delay_centiseconds };
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
            frame.pixels[static_cast<size_t>(y) * width + x] = x < width / 2 ? left : right;
    }
    return frame;
}

/// A white square of `side` pixels centred on black.
inline GifFrame square_frame(int width, int height, int side, int delay_centiseconds)
{
    GifFrame frame{ std::vector<uint8_t>(static_cast<size_t>(width) * height, kGifBlack), delay_centiseconds };
    for (int y = (height - side) / 2; y < (height + side) / 2; ++y)
    {
        for (int x = (width - side) / 2; x < (width + side) / 2; ++x)
            frame.pixels[static_cast<size_t>(y) * width + x] = kGifWhite;
    }
    return frame;
}

} // namespace sensing_test
