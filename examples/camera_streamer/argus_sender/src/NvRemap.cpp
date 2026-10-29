/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvRemap.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

namespace argus_sender
{
namespace
{

const char kRmapMagic[4] = { 'R', 'M', 'A', 'P' };
const uint32_t kWeightOne = 64;

bool fail(std::string* error, const std::string& message)
{
    if (error)
        *error = message;
    return false;
}

// One fixed-point bilinear tap from a float source coordinate; returns false
// (zero-weight tap) when the 2x2 footprint falls outside the source.
NvRemapTap buildTap(float mx, float my, uint32_t srcWidth, uint32_t srcHeight, bool* outOfSource)
{
    NvRemapTap tap;
    if (!(mx >= 0.0F) || !(my >= 0.0F) || mx > static_cast<float>(srcWidth - 1) || my > static_cast<float>(srcHeight - 1))
    {
        *outOfSource = true;
        return tap;
    }
    uint32_t x0 = static_cast<uint32_t>(mx);
    uint32_t y0 = static_cast<uint32_t>(my);
    // Keep the 2x2 footprint inside the source on the last row/column.
    if (x0 >= srcWidth - 1)
        x0 = srcWidth - 2;
    if (y0 >= srcHeight - 1)
        y0 = srcHeight - 2;
    const float fx = mx - static_cast<float>(x0);
    const float fy = my - static_cast<float>(y0);
    uint32_t w01 = static_cast<uint32_t>(fx * (1.0F - fy) * kWeightOne + 0.5F);
    uint32_t w10 = static_cast<uint32_t>((1.0F - fx) * fy * kWeightOne + 0.5F);
    uint32_t w11 = static_cast<uint32_t>(fx * fy * kWeightOne + 0.5F);
    if (w01 + w10 + w11 > kWeightOne)
        w11 = kWeightOne - w01 - w10;
    const uint32_t w00 = kWeightOne - w01 - w10 - w11;
    tap.x0 = static_cast<uint16_t>(x0);
    tap.y0 = static_cast<uint16_t>(y0);
    tap.w00 = static_cast<uint8_t>(w00);
    tap.w01 = static_cast<uint8_t>(w01);
    tap.w10 = static_cast<uint8_t>(w10);
    tap.w11 = static_cast<uint8_t>(w11);
    return tap;
}

inline uint32_t sampleTap(const NvRemapTap& tap,
                          const uint8_t* plane,
                          size_t pitch,
                          size_t pixelStride,
                          size_t channelOffset,
                          uint32_t rowOffset)
{
    const uint8_t* base = plane + static_cast<size_t>(tap.y0 - rowOffset) * pitch +
                          static_cast<size_t>(tap.x0) * pixelStride + channelOffset;
    return (static_cast<uint32_t>(tap.w00) * base[0] + static_cast<uint32_t>(tap.w01) * base[pixelStride] +
            static_cast<uint32_t>(tap.w10) * base[pitch] + static_cast<uint32_t>(tap.w11) * base[pitch + pixelStride] +
            kWeightOne / 2) /
           kWeightOne;
}

inline bool tapIsVoid(const NvRemapTap& tap)
{
    return (tap.w00 | tap.w01 | tap.w10 | tap.w11) == 0;
}

} // namespace

bool loadNvRemapTable(const std::string& path,
                      uint32_t srcWidth,
                      uint32_t srcHeight,
                      uint32_t outWidth,
                      uint32_t outHeight,
                      NvRemapTable* table,
                      std::string* error)
{
    if (!table)
        return fail(error, "remap table output is null");
    *table = NvRemapTable();
    if (srcWidth < 2 || srcHeight < 2 || outWidth == 0 || outHeight == 0 || outWidth % 2 != 0 || outHeight % 2 != 0)
    {
        return fail(error, "remap source/output dimensions are invalid");
    }

    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
        return fail(error, "cannot open remap file '" + path + "'");

    char magic[4] = { 0, 0, 0, 0 };
    uint32_t destWidth = 0;
    uint32_t destHeight = 0;
    float balance = 0.0F;
    bool ok = std::fread(magic, 1, 4, file) == 4 && std::fread(&destWidth, 4, 1, file) == 1 &&
              std::fread(&destHeight, 4, 1, file) == 1 && std::fread(&balance, 4, 1, file) == 1;
    if (!ok || std::memcmp(magic, kRmapMagic, 4) != 0)
    {
        std::fclose(file);
        return fail(error, "'" + path + "' is not an RMAP file");
    }
    if (destWidth == 0 || destHeight == 0 || destWidth % outWidth != 0 || destHeight % outHeight != 0 ||
        destWidth / outWidth != destHeight / outHeight || destWidth / outWidth > 2)
    {
        std::ostringstream message;
        message << "'" << path << "' destination " << destWidth << "x" << destHeight << " is not the wire output "
                << outWidth << "x" << outHeight << " times a shared supersample factor of 1 or 2";
        std::fclose(file);
        return fail(error, message.str());
    }
    const uint32_t supersample = destWidth / outWidth;

    const size_t count = static_cast<size_t>(destWidth) * destHeight;
    std::vector<float> mapX(count);
    std::vector<float> mapY(count);
    ok = std::fread(mapX.data(), sizeof(float), count, file) == count &&
         std::fread(mapY.data(), sizeof(float), count, file) == count;
    std::fclose(file);
    if (!ok)
        return fail(error, "'" + path + "' is truncated");

    table->srcWidth = srcWidth;
    table->srcHeight = srcHeight;
    table->destWidth = destWidth;
    table->destHeight = destHeight;
    table->outWidth = outWidth;
    table->outHeight = outHeight;
    table->supersample = supersample;
    table->balance = balance;

    size_t outOfSourceCount = 0;
    table->lumaTaps.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        bool outOfSource = false;
        table->lumaTaps[i] = buildTap(mapX[i], mapY[i], srcWidth, srcHeight, &outOfSource);
        if (outOfSource)
            ++outOfSourceCount;
    }
    if (outOfSourceCount * 100 > count)
    {
        std::ostringstream message;
        message << "'" << path << "' addresses pixels outside the " << srcWidth << "x" << srcHeight << " source for "
                << outOfSourceCount << "/" << count << " destinations; the map was built for a different source";
        *table = NvRemapTable();
        return fail(error, message.str());
    }
    uint32_t lumaRowFirst = srcHeight - 1;
    uint32_t lumaRowLast = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (tapIsVoid(table->lumaTaps[i]))
            continue;
        const uint32_t y0 = table->lumaTaps[i].y0;
        if (y0 < lumaRowFirst)
            lumaRowFirst = y0;
        if (y0 + 1 > lumaRowLast)
            lumaRowLast = y0 + 1;
    }
    table->lumaRowFirst = lumaRowFirst;
    table->lumaRowLast = lumaRowLast;

    // Chroma taps: one per output UV cell, sampling the half-resolution
    // interleaved UV plane at the average of the cell's central luma map
    // positions (chroma is already 2x subsampled; a single filtered tap per
    // cell is enough).
    const uint32_t chromaWidth = outWidth / 2;
    const uint32_t chromaHeight = outHeight / 2;
    const uint32_t block = 2 * supersample;
    table->chromaTaps.resize(static_cast<size_t>(chromaWidth) * chromaHeight);
    for (uint32_t cy = 0; cy < chromaHeight; ++cy)
    {
        for (uint32_t cx = 0; cx < chromaWidth; ++cx)
        {
            const uint32_t x = cx * block + block / 2 - 1;
            const uint32_t y = cy * block + block / 2 - 1;
            float sumX = 0.0F;
            float sumY = 0.0F;
            for (uint32_t dy = 0; dy < 2; ++dy)
            {
                const size_t row = static_cast<size_t>(y + dy) * destWidth + x;
                sumX += mapX[row] + mapX[row + 1];
                sumY += mapY[row] + mapY[row + 1];
            }
            bool outOfSource = false;
            table->chromaTaps[static_cast<size_t>(cy) * chromaWidth + cx] =
                buildTap(sumX * 0.25F * 0.5F, sumY * 0.25F * 0.5F, srcWidth / 2, srcHeight / 2, &outOfSource);
        }
    }
    uint32_t chromaRowFirst = srcHeight / 2 - 1;
    uint32_t chromaRowLast = 0;
    for (size_t i = 0; i < table->chromaTaps.size(); ++i)
    {
        if (tapIsVoid(table->chromaTaps[i]))
            continue;
        const uint32_t y0 = table->chromaTaps[i].y0;
        if (y0 < chromaRowFirst)
            chromaRowFirst = y0;
        if (y0 + 1 > chromaRowLast)
            chromaRowLast = y0 + 1;
    }
    table->chromaRowFirst = chromaRowFirst;
    table->chromaRowLast = chromaRowLast;
    return true;
}

namespace
{

// Luma resample for output rows [rowBegin, rowEnd) — the unit of the two-way
// split below.
void remapLumaRows(const NvRemapTable& table,
                   const uint8_t* srcY,
                   size_t srcYPitch,
                   uint32_t srcYRowOffset,
                   uint8_t* dst,
                   uint32_t rowBegin,
                   uint32_t rowEnd)
{
    const uint32_t outWidth = table.outWidth;
    uint8_t* dstY = dst + static_cast<size_t>(rowBegin) * outWidth;

    if (table.supersample == 1)
    {
        const NvRemapTap* tap = table.lumaTaps.data() + static_cast<size_t>(rowBegin) * outWidth;
        for (uint32_t y = rowBegin; y < rowEnd; ++y)
        {
            for (uint32_t x = 0; x < outWidth; ++x, ++tap)
            {
                dstY[x] =
                    tapIsVoid(*tap) ? 0 : static_cast<uint8_t>(sampleTap(*tap, srcY, srcYPitch, 1, 0, srcYRowOffset));
            }
            dstY += outWidth;
        }
        return;
    }

    // 2x supersample: each output pixel is the box average of a 2x2 block
    // of bilinear samples, so minified regions are prefiltered.
    const uint32_t destWidth = table.destWidth;
    for (uint32_t y = rowBegin; y < rowEnd; ++y)
    {
        const NvRemapTap* row0 = table.lumaTaps.data() + static_cast<size_t>(2 * y) * destWidth;
        const NvRemapTap* row1 = row0 + destWidth;
        for (uint32_t x = 0; x < outWidth; ++x)
        {
            const NvRemapTap& t00 = row0[2 * x];
            const NvRemapTap& t01 = row0[2 * x + 1];
            const NvRemapTap& t10 = row1[2 * x];
            const NvRemapTap& t11 = row1[2 * x + 1];
            if (tapIsVoid(t00) || tapIsVoid(t01) || tapIsVoid(t10) || tapIsVoid(t11))
            {
                dstY[x] = 0;
                continue;
            }
            const uint32_t sum = sampleTap(t00, srcY, srcYPitch, 1, 0, srcYRowOffset) +
                                 sampleTap(t01, srcY, srcYPitch, 1, 0, srcYRowOffset) +
                                 sampleTap(t10, srcY, srcYPitch, 1, 0, srcYRowOffset) +
                                 sampleTap(t11, srcY, srcYPitch, 1, 0, srcYRowOffset);
            dstY[x] = static_cast<uint8_t>((sum + 2) / 4);
        }
        dstY += outWidth;
    }
}

} // namespace

void nvRemapNv12(const NvRemapTable& table,
                 const uint8_t* srcY,
                 size_t srcYPitch,
                 const uint8_t* srcUV,
                 size_t srcUVPitch,
                 uint8_t* dst,
                 uint32_t srcYRowOffset,
                 uint32_t srcUVRowOffset)
{
    const uint32_t outWidth = table.outWidth;
    const uint32_t outHeight = table.outHeight;
    uint8_t* dstY = dst;

    // The luma gather dominates the frame cost; split it across a helper
    // thread so the per-frame latency stays inside a 60 fps tick.
    const uint32_t split = outHeight / 2;
    std::thread lower(remapLumaRows, std::cref(table), srcY, srcYPitch, srcYRowOffset, dstY, split, outHeight);
    remapLumaRows(table, srcY, srcYPitch, srcYRowOffset, dstY, 0, split);
    lower.join();

    uint8_t* dstUV = dst + static_cast<size_t>(outWidth) * outHeight;
    const uint32_t chromaWidth = outWidth / 2;
    const uint32_t chromaHeight = outHeight / 2;
    const NvRemapTap* tap = table.chromaTaps.data();
    for (uint32_t y = 0; y < chromaHeight; ++y)
    {
        for (uint32_t x = 0; x < chromaWidth; ++x, ++tap)
        {
            if (tapIsVoid(*tap))
            {
                dstUV[2 * x] = 128;
                dstUV[2 * x + 1] = 128;
                continue;
            }
            dstUV[2 * x] = static_cast<uint8_t>(sampleTap(*tap, srcUV, srcUVPitch, 2, 0, srcUVRowOffset));
            dstUV[2 * x + 1] = static_cast<uint8_t>(sampleTap(*tap, srcUV, srcUVPitch, 2, 1, srcUVRowOffset));
        }
        dstUV += static_cast<size_t>(chromaWidth) * 2;
    }
}

} // namespace argus_sender
