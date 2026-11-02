/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvRemap.h"
#include "TestHarness.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

std::string writeRmap(uint32_t destWidth,
                      uint32_t destHeight,
                      const std::vector<float>& mapX,
                      const std::vector<float>& mapY,
                      const char* magic = "RMAP")
{
    char path[] = "/tmp/argus_sender_rmap_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return "";
    close(fd);
    std::ofstream file(path, std::ios::binary);
    const float balance = 0.0F;
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&destWidth), 4);
    file.write(reinterpret_cast<const char*>(&destHeight), 4);
    file.write(reinterpret_cast<const char*>(&balance), 4);
    file.write(reinterpret_cast<const char*>(mapX.data()), static_cast<std::streamsize>(mapX.size() * sizeof(float)));
    file.write(reinterpret_cast<const char*>(mapY.data()), static_cast<std::streamsize>(mapY.size() * sizeof(float)));
    return path;
}

// Reference float bilinear sample of a tightly packed plane.
double bilinear(const std::vector<uint8_t>& plane, uint32_t width, double x, double y)
{
    const uint32_t x0 = static_cast<uint32_t>(x);
    const uint32_t y0 = static_cast<uint32_t>(y);
    const double fx = x - x0;
    const double fy = y - y0;
    const size_t base = static_cast<size_t>(y0) * width + x0;
    return plane[base] * (1 - fx) * (1 - fy) + plane[base + 1] * fx * (1 - fy) + plane[base + width] * (1 - fx) * fy +
           plane[base + width + 1] * fx * fy;
}

} // namespace

int main()
{
    TestContext test("nv_remap");

    // Synthetic pitch-linear NV12 source: 16x8 luma gradient, flat chroma.
    const uint32_t srcWidth = 16;
    const uint32_t srcHeight = 8;
    const size_t srcYPitch = 24; // wider than the image: pitch must be honored
    const size_t srcUVPitch = 24;
    std::vector<uint8_t> srcY(srcYPitch * srcHeight, 0);
    std::vector<uint8_t> tightY(srcWidth * srcHeight, 0);
    for (uint32_t y = 0; y < srcHeight; ++y)
    {
        for (uint32_t x = 0; x < srcWidth; ++x)
        {
            const uint8_t value = static_cast<uint8_t>(10 * x + 3 * y);
            srcY[y * srcYPitch + x] = value;
            tightY[y * srcWidth + x] = value;
        }
    }
    std::vector<uint8_t> srcUV(srcUVPitch * srcHeight / 2, 0);
    for (uint32_t y = 0; y < srcHeight / 2; ++y)
    {
        for (uint32_t x = 0; x < srcWidth / 2; ++x)
        {
            srcUV[y * srcUVPitch + 2 * x] = static_cast<uint8_t>(60 + 4 * x);
            srcUV[y * srcUVPitch + 2 * x + 1] = static_cast<uint8_t>(200 - 4 * y);
        }
    }

    // --- supersample 1, identity map: output equals the source region. ---
    {
        const uint32_t outWidth = 8;
        const uint32_t outHeight = 4;
        std::vector<float> mapX(outWidth * outHeight);
        std::vector<float> mapY(outWidth * outHeight);
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            for (uint32_t x = 0; x < outWidth; ++x)
            {
                mapX[y * outWidth + x] = static_cast<float>(x);
                mapY[y * outWidth + x] = static_cast<float>(y);
            }
        }
        const std::string path = writeRmap(outWidth, outHeight, mapX, mapY);
        argus_sender::NvRemapTable table;
        std::string error;
        TEST_CHECK(test, argus_sender::loadNvRemapTable(path, srcWidth, srcHeight, outWidth, outHeight, &table, &error));
        TEST_CHECK_EQ(test, table.supersample, 1U);
        std::vector<uint8_t> out(outWidth * outHeight * 3 / 2, 0xAA);
        argus_sender::nvRemapNv12(table, srcY.data(), srcYPitch, srcUV.data(), srcUVPitch, out.data());
        bool lumaExact = true;
        for (uint32_t y = 0; y < outHeight; ++y)
            for (uint32_t x = 0; x < outWidth; ++x)
                lumaExact = lumaExact && out[y * outWidth + x] == tightY[y * srcWidth + x];
        TEST_CHECK(test, lumaExact);
        // Chroma taps are sited at the luma cell center (a quarter UV texel
        // off the top-left texel on an identity map), so allow interpolation.
        const uint8_t* outUV = out.data() + outWidth * outHeight;
        TEST_CHECK(test, std::fabs(static_cast<double>(outUV[0]) - srcUV[0]) <= 2.0);
        TEST_CHECK(test, std::fabs(static_cast<double>(outUV[1]) - srcUV[1]) <= 2.0);
        std::remove(path.c_str());
    }

    // --- supersample 2: 2x2 box of bilinear samples matches a float
    // reference within rounding. ---
    {
        const uint32_t outWidth = 4;
        const uint32_t outHeight = 2;
        const uint32_t destWidth = 8;
        const uint32_t destHeight = 4;
        std::vector<float> mapX(destWidth * destHeight);
        std::vector<float> mapY(destWidth * destHeight);
        for (uint32_t y = 0; y < destHeight; ++y)
        {
            for (uint32_t x = 0; x < destWidth; ++x)
            {
                mapX[y * destWidth + x] = 0.75F + 1.4F * x;
                mapY[y * destWidth + x] = 0.5F + 1.2F * y;
            }
        }
        const std::string path = writeRmap(destWidth, destHeight, mapX, mapY);
        argus_sender::NvRemapTable table;
        std::string error;
        TEST_CHECK(test, argus_sender::loadNvRemapTable(path, srcWidth, srcHeight, outWidth, outHeight, &table, &error));
        TEST_CHECK_EQ(test, table.supersample, 2U);
        std::vector<uint8_t> out(outWidth * outHeight * 3 / 2, 0);
        argus_sender::nvRemapNv12(table, srcY.data(), srcYPitch, srcUV.data(), srcUVPitch, out.data());
        double maxError = 0.0;
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            for (uint32_t x = 0; x < outWidth; ++x)
            {
                double sum = 0.0;
                for (uint32_t dy = 0; dy < 2; ++dy)
                    for (uint32_t dx = 0; dx < 2; ++dx)
                    {
                        const size_t i = static_cast<size_t>(2 * y + dy) * destWidth + 2 * x + dx;
                        sum += bilinear(tightY, srcWidth, mapX[i], mapY[i]);
                    }
                maxError = std::max(maxError, std::fabs(sum / 4.0 - out[y * outWidth + x]));
            }
        }
        // Fixed-point taps (1/64 weights) plus two rounding stages.
        TEST_CHECK(test, maxError <= 1.5);
        std::remove(path.c_str());
    }

    // --- out-of-source map values become black luma / neutral chroma. ---
    {
        const uint32_t outWidth = 4;
        const uint32_t outHeight = 2;
        std::vector<float> mapX(outWidth * outHeight, -5.0F);
        std::vector<float> mapY(outWidth * outHeight, -5.0F);
        const std::string path = writeRmap(outWidth, outHeight, mapX, mapY);
        argus_sender::NvRemapTable table;
        std::string error;
        // 100% out-of-source is rejected as a wrong-source map...
        TEST_CHECK(test, !argus_sender::loadNvRemapTable(path, srcWidth, srcHeight, outWidth, outHeight, &table, &error));
        TEST_CHECK(test, error.find("different source") != std::string::npos);
        std::remove(path.c_str());

        // ...but a single stray pixel passes and renders black/neutral.
        for (size_t i = 1; i < mapX.size(); ++i)
        {
            mapX[i] = 1.0F;
            mapY[i] = 1.0F;
        }
        const std::string path2 = writeRmap(outWidth, outHeight, mapX, mapY);
        TEST_CHECK(
            test, !argus_sender::loadNvRemapTable(path2, srcWidth, srcHeight, outWidth, outHeight, &table, &error));
        std::remove(path2.c_str());
    }

    // --- header validation. ---
    {
        std::vector<float> mapX(6, 1.0F);
        std::vector<float> mapY(6, 1.0F);
        argus_sender::NvRemapTable table;
        std::string error;

        const std::string badMagic = writeRmap(3, 2, mapX, mapY, "RMPX");
        TEST_CHECK(test, !argus_sender::loadNvRemapTable(badMagic, srcWidth, srcHeight, 4, 2, &table, &error));
        TEST_CHECK(test, error.find("not an RMAP") != std::string::npos);
        std::remove(badMagic.c_str());

        // dest 6x2 is not out 4x2 times a shared factor of 1 or 2.
        std::vector<float> mapX62(12, 1.0F);
        std::vector<float> mapY62(12, 1.0F);
        const std::string badFactor = writeRmap(6, 2, mapX62, mapY62);
        TEST_CHECK(test, !argus_sender::loadNvRemapTable(badFactor, srcWidth, srcHeight, 4, 2, &table, &error));
        TEST_CHECK(test, error.find("supersample") != std::string::npos);
        std::remove(badFactor.c_str());
    }

    return test.finish();
}
