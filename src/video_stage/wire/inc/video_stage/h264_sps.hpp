// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace video_stage
{
namespace h264
{

/// The fields of an H.264 sequence parameter set that decide how a decoder buffers pictures, and where the VUI's
/// bitstream_restriction sits so it can be rewritten.
struct SpsInfo
{
    uint32_t profile_idc = 0;
    uint32_t level_idc = 0;
    uint32_t chroma_format_idc = 1;
    uint32_t pic_order_cnt_type = 0;
    uint32_t max_num_ref_frames = 0;
    uint32_t width = 0; ///< coded width in luma pixels
    uint32_t height = 0; ///< coded height in luma pixels
    bool frame_mbs_only = true;
    bool vui_present = false;
    bool full_range = false;
    /// The stream states where its chroma samples sit (vertical offsets, 0 = the default left/co-sited siting that
    /// the example operation assumes). Absent means the default.
    bool chroma_loc_present = false;
    uint32_t chroma_loc_top = 0;
    uint32_t chroma_loc_bottom = 0;
    bool bitstream_restriction = false;
    /// Valid when bitstream_restriction is set: how many pictures may precede one in decode order and follow it in
    /// display order, and how many a decoder must hold.
    uint32_t max_num_reorder_frames = 0;
    uint32_t max_dec_frame_buffering = 0;

    /// One line for a log.
    std::string describe() const;
};

/// The first SPS NAL (type 7) in an Annex-B access unit, parsed; nullopt if there is none or it is not understood.
std::optional<SpsInfo> parse_sps(const uint8_t* annexb, size_t size);

/// A copy of the access unit whose SPS declares that the stream never reorders pictures: bitstream_restriction with
/// max_num_reorder_frames 0 and max_dec_frame_buffering equal to max_num_ref_frames (at least 1). Without that, a
/// decoder must assume the worst-case DPB for the level and holds back many pictures before showing the first.
/// Only call it for a stream that really has no B-frames. Returns nullopt if there is no SPS or it cannot be rewritten.
std::optional<std::vector<uint8_t>> declare_no_reordering(const uint8_t* annexb, size_t size);

} // namespace h264
} // namespace video_stage
