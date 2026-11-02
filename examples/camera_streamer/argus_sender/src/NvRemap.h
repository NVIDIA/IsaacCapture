/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_NV_REMAP_H
#define ARGUS_SENDER_NV_REMAP_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace argus_sender
{

// Precomputed NV12 lookup-table remap.
//
// Loads the RMAP binaries emitted by
// robotics/hardware_sdk/camera/calib/gen_remap_maps.py (little-endian
// header struct(<4sIIf) = [magic "RMAP", dest width u32, dest height u32,
// balance f32], then map_x[H*W] and map_y[H*W] float32 row-major: the SOURCE
// pixel each destination pixel samples, cv2.remap CV_32FC1 semantics).
//
// The float maps are converted once at load into fixed-point bilinear taps
// (top-left source coordinate + four weights summing to 64), so the per-frame
// work is integer gathers only. The RMAP destination grid may be an integer
// multiple (the supersample factor, 1 or 2 per axis, identical on both) of the
// wire output size; remapNv12() then box-averages each SxS block so the
// minified output is properly prefiltered instead of aliased.
//
// This module is deliberately free of Jetson/Argus dependencies so the LUT
// build and resample math stay unit-testable on x86.

struct NvRemapTap
{
    uint16_t x0 = 0;
    uint16_t y0 = 0;
    // Bilinear weights in 1/64 units; all zero marks an out-of-source tap
    // (black for luma, neutral 128 for chroma).
    uint8_t w00 = 0;
    uint8_t w01 = 0;
    uint8_t w10 = 0;
    uint8_t w11 = 0;
};

struct NvRemapTable
{
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;
    // RMAP destination grid (= outWidth/outHeight * supersample).
    uint32_t destWidth = 0;
    uint32_t destHeight = 0;
    uint32_t outWidth = 0;
    uint32_t outHeight = 0;
    uint32_t supersample = 1;
    float balance = 0.0F;
    // Source rows the luma taps actually read ([first, last] inclusive of the
    // bilinear +1 row). Callers gathering from slow (uncached/write-combined)
    // mappings can copy just these rows into cached memory and pass the copy
    // with the matching row offset.
    uint32_t lumaRowFirst = 0;
    uint32_t lumaRowLast = 0;
    // Same for the chroma taps, in UV-plane rows.
    uint32_t chromaRowFirst = 0;
    uint32_t chromaRowLast = 0;
    // destWidth x destHeight luma taps, row-major.
    std::vector<NvRemapTap> lumaTaps;
    // (outWidth/2) x (outHeight/2) chroma taps addressing the half-resolution
    // interleaved UV plane, row-major.
    std::vector<NvRemapTap> chromaTaps;

    bool loaded() const
    {
        return !lumaTaps.empty();
    }
};

// Reads an RMAP file and builds the fixed-point taps. The RMAP header must
// carry destination dimensions equal to outWidth/outHeight times a shared
// supersample factor of 1 or 2, and every in-range map value must address the
// srcWidth x srcHeight source frame. Map values outside the source become
// zero-weight taps; more than 1% of them is treated as a wrong-source error.
bool loadNvRemapTable(const std::string& path,
                      uint32_t srcWidth,
                      uint32_t srcHeight,
                      uint32_t outWidth,
                      uint32_t outHeight,
                      NvRemapTable* table,
                      std::string* error);

// Resamples one pitch-linear NV12 frame (srcY / interleaved srcUV) into a
// tightly packed outWidth x outHeight NV12 payload (Y plane then interleaved
// UV plane, no padding). `dst` must hold outWidth * outHeight * 3 / 2 bytes.
// srcYRowOffset/srcUVRowOffset declare how many leading source rows the given
// plane pointers are missing (0 for a full frame): row r of the original
// source lives at plane + (r - offset) * pitch. Use with the table's
// lumaRowFirst/chromaRowFirst when gathering from a partial cached copy.
void nvRemapNv12(const NvRemapTable& table,
                 const uint8_t* srcY,
                 size_t srcYPitch,
                 const uint8_t* srcUV,
                 size_t srcUVPitch,
                 uint8_t* dst,
                 uint32_t srcYRowOffset = 0,
                 uint32_t srcUVRowOffset = 0);

} // namespace argus_sender

#endif // ARGUS_SENDER_NV_REMAP_H
