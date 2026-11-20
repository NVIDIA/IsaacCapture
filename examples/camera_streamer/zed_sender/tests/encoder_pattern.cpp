// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Opt-in VIC/NVENC test: deterministic BT.601 limited-range stereo color bars.
#include "HardwareEncoder.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: encoder_pattern OUTPUT_PREFIX\n";
        return 2;
    }
    try
    {
        // Y,U,V: black, gray, white, red, green, blue, yellow, cyan.
        const std::array<std::array<uint8_t, 3>, 8> colors{ { { { 16, 128, 128 } },
                                                              { { 126, 128, 128 } },
                                                              { { 235, 128, 128 } },
                                                              { { 81, 90, 240 } },
                                                              { { 145, 54, 34 } },
                                                              { { 41, 240, 110 } },
                                                              { { 210, 16, 146 } },
                                                              { { 170, 166, 16 } } } };
        const uint32_t width = 1280, height = 720, outputWidth = 640, outputHeight = 360;
        std::vector<uint8_t> pixels(width * 2 * height * 2);
        std::ofstream left(std::string(argv[1]) + "-left.h265", std::ios::binary);
        std::ofstream right(std::string(argv[1]) + "-right.h265", std::ios::binary);
        if (!left || !right)
            throw std::runtime_error("cannot create output files");
        zed_sender::HardwareEncoder encoder;
        encoder.open(width * 2, height, outputWidth, outputHeight, 30, 6, 30);
        for (unsigned frame = 0; frame < 8; ++frame)
        {
            for (uint32_t y = 0; y < height; ++y)
                for (uint32_t x = 0; x < width * 2; x += 2)
                {
                    const unsigned eye = x / width;
                    const unsigned bar = (x % width) / (width / 8);
                    const auto& c = colors[((eye ? 7 - bar : bar) + frame) % 8];
                    const size_t offset = (size_t(y) * width * 2 + x) * 2;
                    pixels[offset] = pixels[offset + 2] = c[0];
                    pixels[offset + 1] = c[1];
                    pixels[offset + 3] = c[2];
                }
            auto encoded = encoder.encode(pixels.data(), pixels.size(), width * 4);
            left.write(reinterpret_cast<const char*>(encoded.left.data()), encoded.left.size());
            right.write(reinterpret_cast<const char*>(encoded.right.data()), encoded.right.size());
        }
        if (!left || !right)
            throw std::runtime_error("output write failed");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
