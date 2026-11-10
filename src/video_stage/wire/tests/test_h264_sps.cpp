// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/h264_sps.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

using video_stage::h264::declare_no_reordering;
using video_stage::h264::parse_sps;
using video_stage::h264::SpsInfo;

namespace
{
// First SPS, PPS and the start of the first IDR slice of short ffmpeg streams (x264 unless noted), 3 frames each.
// Every SPS contains emulation-prevention bytes in its VUI timing fields. The IDR bytes are truncated: only their
// preservation matters here.
// a_high_720
const std::vector<uint8_t> a_high_720_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, 0xac, 0xb2,
                                              0x00, 0xa0, 0x0b, 0x76, 0x02, 0x20, 0x00, 0x00, 0x03, 0x00,
                                              0x20, 0x00, 0x00, 0x07, 0x81, 0xe3, 0x06, 0x49 };
const std::vector<uint8_t> a_high_720_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xc3, 0xcb, 0x22, 0xc0 };
const std::vector<uint8_t> a_high_720_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x08, 0x7f, 0xd3, 0xd6, 0x67, 0x0b,
                                              0x58, 0xc9, 0x09, 0x0f, 0xd0, 0x9f, 0x5b, 0xe3, 0x26, 0xe9, 0xc3, 0xd4,
                                              0xb8, 0x70, 0x34, 0xe3, 0x55, 0x12, 0xc0, 0x2f, 0xf6, 0xfb, 0x3c, 0x4c,
                                              0x99, 0xf9, 0xe1, 0x86, 0x0b, 0xe5, 0x90, 0x5d, 0x3f, 0x68, 0xf2, 0xb6 };
// b_high_2560
const std::vector<uint8_t> b_high_2560_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x33, 0xac, 0xb2,
                                               0x00, 0x50, 0x01, 0xf3, 0x60, 0x22, 0x00, 0x00, 0x03, 0x00,
                                               0x02, 0x00, 0x00, 0x03, 0x00, 0x78, 0x1e, 0x30, 0x64, 0x90 };
const std::vector<uint8_t> b_high_2560_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xc3, 0xcb, 0x22, 0xc0 };
const std::vector<uint8_t> b_high_2560_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x08, 0x7f, 0xd3, 0xd6, 0x67, 0x0b,
                                               0x58, 0xc9, 0x09, 0x0f, 0xd0, 0x9f, 0x5b, 0xe3, 0x26, 0xe9, 0xc3, 0xd4,
                                               0xb8, 0x70, 0x34, 0xe3, 0x55, 0x12, 0xce, 0x35, 0x37, 0xb3, 0xca, 0x63,
                                               0x32, 0x3d, 0xb7, 0x41, 0xc7, 0x2a, 0x82, 0xea, 0x04, 0xb9, 0x64, 0x3e };
// c_base_720
const std::vector<uint8_t> c_base_720_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1f, 0xd9, 0x00,
                                              0x50, 0x05, 0xbb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10,
                                              0x00, 0x00, 0x03, 0x03, 0xc0, 0xf1, 0x83, 0x24, 0x80 };
const std::vector<uint8_t> c_base_720_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xcb, 0x83, 0xcb, 0x20 };
const std::vector<uint8_t> c_base_720_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x08, 0x7c, 0x42, 0x22, 0xb0, 0x60,
                                              0x42, 0x1f, 0x80, 0xea, 0x40, 0x03, 0xec, 0xcc, 0xb3, 0x20, 0x00, 0x2d,
                                              0x66, 0x69, 0x14, 0x68, 0x3e, 0x4e, 0xff, 0x70, 0x2a, 0x0a, 0x4c, 0xb3,
                                              0x6e, 0x2d, 0xfe, 0x03, 0x16, 0x67, 0x22, 0x8e, 0x1f, 0x37, 0x7f, 0xb0 };
// d_cbr_2560
const std::vector<uint8_t> d_cbr_2560_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x33, 0xac, 0xb2, 0x00,
                                              0x50, 0x01, 0xf3, 0x60, 0x22, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00,
                                              0x00, 0x03, 0x00, 0x78, 0xcc, 0x00, 0x01, 0xe8, 0x48, 0x00, 0x10,
                                              0x4b, 0xed, 0x61, 0x80, 0x3c, 0x60, 0xc9, 0x20 };
const std::vector<uint8_t> d_cbr_2560_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xcc, 0xb2, 0x2c };
const std::vector<uint8_t> d_cbr_2560_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x0b, 0x7f, 0xdb, 0xe5, 0x95, 0xbd,
                                              0xc6, 0x03, 0x72, 0x93, 0x37, 0x67, 0xad, 0x5c, 0xfd, 0xf5, 0x21, 0xdc,
                                              0xcb, 0x9f, 0xfd, 0xce, 0xc4, 0xe2, 0x96, 0x1a, 0x45, 0xb2, 0x75, 0x05,
                                              0x88, 0xac, 0xfb, 0x5f, 0xf2, 0x45, 0x4c, 0x6d, 0x93, 0x7f, 0xac, 0x86 };
// e_bframes
const std::vector<uint8_t> e_bframes_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9,
                                             0x40, 0x50, 0x05, 0xbb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00,
                                             0x10, 0x00, 0x00, 0x03, 0x03, 0xc0, 0xf1, 0x83, 0x19, 0x60 };
const std::vector<uint8_t> e_bframes_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xe9, 0x38, 0xf2, 0xc8, 0xb0 };
const std::vector<uint8_t> e_bframes_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x21, 0xff, 0xd3, 0xd6, 0x67,
                                             0x0b, 0x58, 0xc9, 0x09, 0x0f, 0xd0, 0x9f, 0x5b, 0xe3, 0x26, 0xe9, 0xc3,
                                             0xd4, 0xb8, 0x70, 0x34, 0xe3, 0x55, 0x12, 0xc0, 0x2f, 0xf6, 0xfb, 0x3c,
                                             0x4c, 0x99, 0xf9, 0xe1, 0x86, 0x0b, 0xe5, 0x90, 0x5d, 0x3f, 0x68, 0xf2 };
// f_full
const std::vector<uint8_t> f_full_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, 0xac, 0xb2,
                                          0x00, 0xa0, 0x0b, 0x76, 0x02, 0xd9, 0x00, 0x00, 0x03, 0x00,
                                          0x01, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x0f, 0x18, 0x32, 0x48 };
const std::vector<uint8_t> f_full_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xc3, 0xcb, 0x22, 0xc0 };
const std::vector<uint8_t> f_full_idr = { 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x08, 0x7f, 0xd3, 0xd6, 0x67, 0x0b,
                                          0x58, 0xc9, 0x09, 0x0f, 0xd0, 0x9f, 0x5b, 0xe3, 0x26, 0xe9, 0xc3, 0xd4,
                                          0xb8, 0x70, 0x34, 0xe3, 0x55, 0x12, 0xc0, 0x2f, 0xf6, 0xfb, 0x3c, 0x4c,
                                          0x99, 0xf9, 0xe1, 0x86, 0x0b, 0xe5, 0x90, 0x5d, 0x3f, 0x68, 0xf2, 0xb6 };
// g_nvenc
const std::vector<uint8_t> g_nvenc_sps = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x4d, 0x40, 0x1f, 0x95, 0x90, 0x05, 0x00, 0x5b,
                                           0xb0, 0x11, 0x00, 0x00, 0x03, 0x03, 0xe8, 0x00, 0x00, 0xea, 0x60, 0xe0, 0x00,
                                           0x00, 0x16, 0xe3, 0x60, 0x00, 0x07, 0xa1, 0x20, 0xbb, 0xcb, 0x82, 0x80 };
const std::vector<uint8_t> g_nvenc_pps = { 0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0x8f, 0x20 };
const std::vector<uint8_t> g_nvenc_idr = { 0x00, 0x00, 0x00, 0x01, 0x65, 0xb8, 0x04, 0x3f, 0xad, 0x6e, 0xda, 0xc4,
                                           0xf1, 0x5a, 0x80, 0x2e, 0xee, 0x6f, 0x27, 0x6e, 0x50, 0x1f, 0xdd, 0x44,
                                           0xcf, 0xce, 0x97, 0x18, 0xd2, 0x37, 0x20, 0xd0, 0x6c, 0x94, 0xee, 0xe0,
                                           0xe1, 0xc6, 0x50, 0x59, 0x74, 0xda, 0x8d, 0x62, 0xfa, 0x36, 0x69, 0xbc };

std::vector<uint8_t> concat(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const std::vector<uint8_t>& c)
{
    std::vector<uint8_t> out(a);
    out.insert(out.end(), b.begin(), b.end());
    out.insert(out.end(), c.begin(), c.end());
    return out;
}

struct Case
{
    const char* name;
    const std::vector<uint8_t>* sps;
    const std::vector<uint8_t>* pps;
    const std::vector<uint8_t>* idr;
    uint32_t profile, level, width, height, refs, poc;
    bool vui, full_range, restriction;
    uint32_t reorder, dpb;
};

const std::vector<Case>& cases()
{
    static const std::vector<Case> c = {
        { "x264 high 1280x720", &a_high_720_sps, &a_high_720_pps, &a_high_720_idr, 100, 31, 1280, 720, 3, 2, true,
          false, true, 0, 3 },
        { "x264 high 2560x1984", &b_high_2560_sps, &b_high_2560_pps, &b_high_2560_idr, 100, 51, 2560, 1984, 3, 2, true,
          false, true, 0, 3 },
        { "x264 baseline 1280x720", &c_base_720_sps, &c_base_720_pps, &c_base_720_idr, 66, 31, 1280, 720, 3, 2, true,
          false, true, 0, 3 },
        { "x264 high CBR (nal-hrd) 2560x1984", &d_cbr_2560_sps, &d_cbr_2560_pps, &d_cbr_2560_idr, 100, 51, 2560, 1984,
          3, 2, true, false, true, 0, 3 },
        { "x264 high with B-frames", &e_bframes_sps, &e_bframes_pps, &e_bframes_idr, 100, 31, 1280, 720, 4, 0, true,
          false, true, 2, 4 },
        { "x264 full range", &f_full_sps, &f_full_pps, &f_full_idr, 100, 31, 1280, 720, 3, 2, true, true, true, 0, 3 },
        // NVENC emits a VUI without bitstream_restriction: the case the rewrite exists for.
        { "h264_nvenc main", &g_nvenc_sps, &g_nvenc_pps, &g_nvenc_idr, 77, 31, 1280, 720, 3, 2, true, false, false, 0, 0 },
    };
    return c;
}
} // namespace

TEST_CASE("parse_sps reads the fields of real SPS NAL units", "[Sps]")
{
    for (const Case& c : cases())
    {
        INFO(c.name);
        const auto au = concat(*c.sps, *c.pps, *c.idr);
        const auto info = parse_sps(au.data(), au.size());
        REQUIRE(info);
        CHECK(info->profile_idc == c.profile);
        CHECK(info->level_idc == c.level);
        CHECK(info->width == c.width);
        CHECK(info->height == c.height);
        CHECK(info->max_num_ref_frames == c.refs);
        CHECK(info->pic_order_cnt_type == c.poc);
        CHECK(info->frame_mbs_only);
        CHECK(info->vui_present == c.vui);
        CHECK(info->full_range == c.full_range);
        CHECK(info->bitstream_restriction == c.restriction);
        if (c.restriction)
        {
            CHECK(info->max_num_reorder_frames == c.reorder);
            CHECK(info->max_dec_frame_buffering == c.dpb);
        }
        CHECK_FALSE(info->describe().empty());
    }
}

TEST_CASE("declare_no_reordering sets the restriction and changes nothing else", "[Sps]")
{
    for (const Case& c : cases())
    {
        INFO(c.name);
        const auto au = concat(*c.sps, *c.pps, *c.idr);
        const auto before = parse_sps(au.data(), au.size());
        REQUIRE(before);
        const auto out = declare_no_reordering(au.data(), au.size());
        REQUIRE(out);
        const auto after = parse_sps(out->data(), out->size());
        REQUIRE(after);

        CHECK(after->bitstream_restriction);
        CHECK(after->max_num_reorder_frames == 0);
        CHECK(after->max_dec_frame_buffering == std::max<uint32_t>(before->max_num_ref_frames, 1));
        CHECK(after->profile_idc == before->profile_idc);
        CHECK(after->level_idc == before->level_idc);
        CHECK(after->chroma_format_idc == before->chroma_format_idc);
        CHECK(after->pic_order_cnt_type == before->pic_order_cnt_type);
        CHECK(after->max_num_ref_frames == before->max_num_ref_frames);
        CHECK(after->width == before->width);
        CHECK(after->height == before->height);
        CHECK(after->frame_mbs_only == before->frame_mbs_only);
        CHECK(after->vui_present);
        CHECK(after->full_range == before->full_range);

        // Everything after the SPS (PPS, IDR) is byte-identical.
        const std::size_t rest = c.pps->size() + c.idr->size();
        REQUIRE(out->size() > rest);
        CHECK(std::equal(au.end() - rest, au.end(), out->end() - rest));
        // The SPS keeps its start code and header byte.
        CHECK(std::equal(c.sps->begin(), c.sps->begin() + 5, out->begin()));

        // Rewriting is idempotent.
        const auto twice = declare_no_reordering(out->data(), out->size());
        REQUIRE(twice);
        CHECK(*twice == *out);
    }
}

TEST_CASE("an access unit with a different NAL ahead of the SPS is handled", "[Sps]")
{
    const Case& c = cases()[6];
    const std::vector<uint8_t> aud = { 0, 0, 0, 1, 0x09, 0xf0 };
    const auto body = concat(*c.sps, *c.pps, *c.idr);
    const auto au = concat(aud, body, {});
    const auto info = parse_sps(au.data(), au.size());
    REQUIRE(info);
    CHECK(info->width == 1280);
    const auto out = declare_no_reordering(au.data(), au.size());
    REQUIRE(out);
    CHECK(std::equal(aud.begin(), aud.end(), out->begin()));
    const auto after = parse_sps(out->data(), out->size());
    REQUIRE(after);
    CHECK(after->bitstream_restriction);
    CHECK(std::equal(au.end() - (c.pps->size() + c.idr->size()), au.end(), out->end() - (c.pps->size() + c.idr->size())));
}

TEST_CASE("a three byte start code is accepted", "[Sps]")
{
    const Case& c = cases()[0];
    std::vector<uint8_t> au(c.sps->begin() + 1, c.sps->end()); // 00 00 01 + NAL
    const auto info = parse_sps(au.data(), au.size());
    REQUIRE(info);
    CHECK(info->width == 1280);
}

TEST_CASE("no SPS or a damaged SPS yields nullopt", "[Sps]")
{
    const std::vector<uint8_t> none = { 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80 };
    CHECK_FALSE(parse_sps(none.data(), none.size()));
    CHECK_FALSE(declare_no_reordering(none.data(), none.size()));
    CHECK_FALSE(parse_sps(none.data(), 0));

    const Case& c = cases()[0];
    std::vector<uint8_t> truncated(c.sps->begin(), c.sps->begin() + 8);
    CHECK_FALSE(declare_no_reordering(truncated.data(), truncated.size()));
}

namespace
{
struct BitWriter
{
    std::vector<uint8_t> bytes;
    int used = 8;
    void bit(unsigned v)
    {
        if (used == 8)
        {
            bytes.push_back(0);
            used = 0;
        }
        bytes.back() = static_cast<uint8_t>(bytes.back() | ((v & 1u) << (7 - used)));
        ++used;
    }
    void bits(unsigned v, int n)
    {
        for (int i = n - 1; i >= 0; --i)
            bit((v >> i) & 1u);
    }
    void ue(unsigned v)
    {
        const unsigned x = v + 1;
        int len = 0;
        while ((x >> len) > 1)
            ++len;
        bits(0, len);
        bits(x, len + 1);
    }
};

/// A baseline-profile 1280x720 SPS whose VUI carries only the chroma location field. `chroma_loc` is null for none.
std::vector<uint8_t> synthetic_sps(const std::pair<unsigned, unsigned>* chroma_loc)
{
    BitWriter w;
    w.bits(66, 8); // profile_idc
    w.bits(0, 8); // constraint flags
    w.bits(31, 8); // level_idc
    w.ue(0); // seq_parameter_set_id
    w.ue(0); // log2_max_frame_num_minus4
    w.ue(2); // pic_order_cnt_type
    w.ue(1); // max_num_ref_frames
    w.bit(0); // gaps_in_frame_num_value_allowed_flag
    w.ue(79); // pic_width_in_mbs_minus1
    w.ue(44); // pic_height_in_map_units_minus1
    w.bit(1); // frame_mbs_only_flag
    w.bit(1); // direct_8x8_inference_flag
    w.bit(0); // frame_cropping_flag
    w.bit(1); // vui_parameters_present_flag
    w.bit(0); // aspect_ratio_info_present_flag
    w.bit(0); // overscan_info_present_flag
    w.bit(0); // video_signal_type_present_flag
    if (chroma_loc)
    {
        w.bit(1);
        w.ue(chroma_loc->first);
        w.ue(chroma_loc->second);
    }
    else
    {
        w.bit(0);
    }
    w.bit(0); // timing_info_present_flag
    w.bit(0); // nal_hrd_parameters_present_flag
    w.bit(0); // vcl_hrd_parameters_present_flag
    w.bit(0); // pic_struct_present_flag
    w.bit(0); // bitstream_restriction_flag
    w.bit(1); // rbsp_stop_one_bit
    std::vector<uint8_t> out = { 0, 0, 0, 1, 0x67 };
    out.insert(out.end(), w.bytes.begin(), w.bytes.end());
    return out;
}
} // namespace

TEST_CASE("chroma sample locations parse into chroma_loc_*", "[Sps]")
{
    const std::pair<unsigned, unsigned> zero{ 0, 0 };
    const std::pair<unsigned, unsigned> one{ 1, 1 };
    const std::pair<unsigned, unsigned> mixed{ 2, 5 };

    SECTION("absent")
    {
        const auto au = synthetic_sps(nullptr);
        const auto info = parse_sps(au.data(), au.size());
        REQUIRE(info);
        CHECK(info->width == 1280);
        CHECK(info->height == 720);
        CHECK(info->vui_present);
        CHECK_FALSE(info->chroma_loc_present);
        CHECK(info->chroma_loc_top == 0);
        CHECK(info->chroma_loc_bottom == 0);
    }
    SECTION("present with (0, 0)")
    {
        const auto au = synthetic_sps(&zero);
        const auto info = parse_sps(au.data(), au.size());
        REQUIRE(info);
        CHECK(info->chroma_loc_present);
        CHECK(info->chroma_loc_top == 0);
        CHECK(info->chroma_loc_bottom == 0);
    }
    SECTION("present with (1, 1)")
    {
        const auto au = synthetic_sps(&one);
        const auto info = parse_sps(au.data(), au.size());
        REQUIRE(info);
        CHECK(info->chroma_loc_present);
        CHECK(info->chroma_loc_top == 1);
        CHECK(info->chroma_loc_bottom == 1);
        CHECK(info->width == 1280);
    }
    SECTION("distinct top and bottom values")
    {
        const auto au = synthetic_sps(&mixed);
        const auto info = parse_sps(au.data(), au.size());
        REQUIRE(info);
        CHECK(info->chroma_loc_top == 2);
        CHECK(info->chroma_loc_bottom == 5);
    }
}
