// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/h264_sps.hpp"

#include <algorithm>
#include <cstring>

namespace video_stage
{
namespace h264
{

namespace
{

// RBSP bytes of a NAL: the payload with emulation-prevention bytes (00 00 03) removed.
std::vector<uint8_t> unescape(const uint8_t* p, size_t n)
{
    std::vector<uint8_t> out;
    out.reserve(n);
    int zeros = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (zeros >= 2 && p[i] == 3)
        {
            zeros = 0;
            continue;
        }
        out.push_back(p[i]);
        zeros = p[i] == 0 ? zeros + 1 : 0;
    }
    return out;
}

std::vector<uint8_t> escape(const std::vector<uint8_t>& rbsp)
{
    std::vector<uint8_t> out;
    out.reserve(rbsp.size() + 8);
    int zeros = 0;
    for (uint8_t b : rbsp)
    {
        if (zeros >= 2 && b <= 3)
        {
            out.push_back(3);
            zeros = 0;
        }
        out.push_back(b);
        zeros = b == 0 ? zeros + 1 : 0;
    }
    return out;
}

class BitReader
{
public:
    explicit BitReader(const std::vector<uint8_t>& d) : m_data(d)
    {
    }
    bool ok() const
    {
        return !m_error;
    }
    size_t position() const
    {
        return m_pos;
    }
    uint32_t bits(int n)
    {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i)
            v = (v << 1) | bit();
        return v;
    }
    uint32_t bit()
    {
        if (m_pos >= m_data.size() * 8)
        {
            m_error = true;
            return 0;
        }
        const uint32_t b = (m_data[m_pos >> 3] >> (7 - (m_pos & 7))) & 1;
        ++m_pos;
        return b;
    }
    uint32_t ue()
    {
        int zeros = 0;
        while (!bit() && ok() && zeros < 32)
            ++zeros;
        if (zeros >= 32)
        {
            m_error = true;
            return 0;
        }
        return (1u << zeros) - 1 + bits(zeros);
    }
    int32_t se()
    {
        const uint32_t k = ue();
        return (k & 1) ? static_cast<int32_t>((k + 1) / 2) : -static_cast<int32_t>(k / 2);
    }

private:
    const std::vector<uint8_t>& m_data;
    size_t m_pos = 0;
    bool m_error = false;
};

class BitWriter
{
public:
    void bit(uint32_t b)
    {
        if ((m_bits & 7) == 0)
            m_data.push_back(0);
        if (b)
            m_data.back() |= static_cast<uint8_t>(1u << (7 - (m_bits & 7)));
        ++m_bits;
    }
    void bits(uint32_t v, int n)
    {
        for (int i = n - 1; i >= 0; --i)
            bit((v >> i) & 1);
    }
    void ue(uint32_t v)
    {
        const uint32_t x = v + 1;
        int len = 0;
        while ((x >> len) > 1)
            ++len;
        bits(0, len);
        bits(x, len + 1);
    }
    void copy(const std::vector<uint8_t>& src, size_t from, size_t to)
    {
        for (size_t i = from; i < to; ++i)
            bit((src[i >> 3] >> (7 - (i & 7))) & 1);
    }
    std::vector<uint8_t>& finish()
    {
        bit(1); // rbsp_stop_one_bit, then zero bits to a byte boundary
        return m_data;
    }

private:
    std::vector<uint8_t> m_data;
    size_t m_bits = 0;
};

void skip_scaling_list(BitReader& r, int size)
{
    int last = 8, next = 8;
    for (int j = 0; j < size; ++j)
    {
        if (next != 0)
        {
            next = (last + r.se() + 256) % 256;
        }
        last = next == 0 ? last : next;
    }
}

void skip_hrd(BitReader& r)
{
    const uint32_t cpb_cnt = r.ue() + 1;
    r.bits(4);
    r.bits(4);
    for (uint32_t i = 0; i < cpb_cnt && r.ok(); ++i)
    {
        r.ue();
        r.ue();
        r.bit();
    }
    r.bits(5);
    r.bits(5);
    r.bits(5);
    r.bits(5);
}

struct Parsed
{
    SpsInfo info;
    std::vector<uint8_t> rbsp;      // the SPS RBSP, header byte first
    size_t vui_flag_pos = 0;        // bit position of vui_parameters_present_flag
    size_t restriction_flag_pos = 0; // bit position of bitstream_restriction_flag (valid when vui_present)
};

std::optional<Parsed> parse_rbsp(std::vector<uint8_t> rbsp)
{
    if (rbsp.size() < 4)
        return std::nullopt;
    Parsed out;
    out.rbsp = std::move(rbsp);
    BitReader r(out.rbsp);
    r.bits(8); // NAL header
    SpsInfo& s = out.info;
    s.profile_idc = r.bits(8);
    r.bits(8);
    s.level_idc = r.bits(8);
    r.ue();
    const uint32_t high[] = { 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135 };
    if (std::find(std::begin(high), std::end(high), s.profile_idc) != std::end(high))
    {
        s.chroma_format_idc = r.ue();
        if (s.chroma_format_idc == 3)
            r.bit();
        r.ue();
        r.ue();
        r.bit();
        if (r.bit())
        {
            for (int i = 0; i < (s.chroma_format_idc != 3 ? 8 : 12) && r.ok(); ++i)
            {
                if (r.bit())
                    skip_scaling_list(r, i < 6 ? 16 : 64);
            }
        }
    }
    r.ue();
    s.pic_order_cnt_type = r.ue();
    if (s.pic_order_cnt_type == 0)
    {
        r.ue();
    }
    else if (s.pic_order_cnt_type == 1)
    {
        r.bit();
        r.se();
        r.se();
        const uint32_t cycle = r.ue();
        for (uint32_t i = 0; i < cycle && r.ok(); ++i)
            r.se();
    }
    s.max_num_ref_frames = r.ue();
    r.bit();
    const uint32_t w_mbs = r.ue() + 1;
    const uint32_t h_map_units = r.ue() + 1;
    s.frame_mbs_only = r.bit() != 0;
    if (!s.frame_mbs_only)
        r.bit();
    r.bit();
    uint32_t crop[4] = { 0, 0, 0, 0 };
    if (r.bit())
    {
        for (uint32_t& c : crop)
            c = r.ue();
    }
    s.width = w_mbs * 16;
    s.height = h_map_units * (s.frame_mbs_only ? 1 : 2) * 16;
    const uint32_t cx = (s.chroma_format_idc == 1 || s.chroma_format_idc == 2) ? 2 : 1;
    const uint32_t cy = (s.chroma_format_idc == 1 ? 2 : 1) * (s.frame_mbs_only ? 1 : 2);
    s.width -= cx * (crop[0] + crop[1]);
    s.height -= cy * (crop[2] + crop[3]);
    out.vui_flag_pos = r.position();
    s.vui_present = r.bit() != 0;
    if (s.vui_present)
    {
        if (r.bit())
        {
            if (r.bits(8) == 255)
                r.bits(32);
        }
        if (r.bit())
            r.bit();
        if (r.bit())
        {
            r.bits(3);
            s.full_range = r.bit() != 0;
            if (r.bit())
                r.bits(24);
        }
        if (r.bit())
        {
            r.ue();
            r.ue();
        }
        if (r.bit())
        {
            r.bits(32);
            r.bits(32);
            r.bit();
        }
        const bool nal_hrd = r.bit() != 0;
        if (nal_hrd)
            skip_hrd(r);
        const bool vcl_hrd = r.bit() != 0;
        if (vcl_hrd)
            skip_hrd(r);
        if (nal_hrd || vcl_hrd)
            r.bit();
        r.bit();
        out.restriction_flag_pos = r.position();
        s.bitstream_restriction = r.bit() != 0;
        if (s.bitstream_restriction)
        {
            r.bit();
            r.ue();
            r.ue();
            r.ue();
            r.ue();
            s.max_num_reorder_frames = r.ue();
            s.max_dec_frame_buffering = r.ue();
        }
    }
    if (!r.ok())
        return std::nullopt;
    return out;
}

// Finds the first NAL of `type`; returns the offsets of its header byte and its end (start of the next start code).
bool find_nal(const uint8_t* d, size_t n, int type, size_t& begin, size_t& end, size_t& start_code)
{
    for (size_t i = 0; i + 3 < n; ++i)
    {
        if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1 && (d[i + 3] & 0x1F) == type)
        {
            begin = i + 3;
            start_code = (i > 0 && d[i - 1] == 0) ? i - 1 : i;
            size_t j = begin;
            while (j + 2 < n && !(d[j] == 0 && d[j + 1] == 0 && (d[j + 2] == 1 || d[j + 2] == 0)))
                ++j;
            end = (j + 2 < n) ? j : n;
            return true;
        }
    }
    return false;
}

} // namespace

std::string SpsInfo::describe() const
{
    std::string s = "H.264 profile " + std::to_string(profile_idc) + " level " + std::to_string(level_idc) + ", " +
                    std::to_string(width) + "x" + std::to_string(height) + ", refs " + std::to_string(max_num_ref_frames) +
                    ", poc type " + std::to_string(pic_order_cnt_type);
    if (bitstream_restriction)
        s += ", bitstream_restriction: reorder " + std::to_string(max_num_reorder_frames) + ", dpb " +
             std::to_string(max_dec_frame_buffering);
    else
        s += vui_present ? ", VUI without bitstream_restriction" : ", no VUI";
    return s;
}

std::optional<SpsInfo> parse_sps(const uint8_t* annexb, size_t size)
{
    size_t begin = 0, end = 0, sc = 0;
    if (!find_nal(annexb, size, 7, begin, end, sc))
        return std::nullopt;
    auto parsed = parse_rbsp(unescape(annexb + begin, end - begin));
    if (!parsed)
        return std::nullopt;
    return parsed->info;
}

std::optional<std::vector<uint8_t>> declare_no_reordering(const uint8_t* annexb, size_t size)
{
    size_t begin = 0, end = 0, sc = 0;
    if (!find_nal(annexb, size, 7, begin, end, sc))
        return std::nullopt;
    auto parsed = parse_rbsp(unescape(annexb + begin, end - begin));
    if (!parsed)
        return std::nullopt;
    const SpsInfo& info = parsed->info;
    const std::vector<uint8_t>& rbsp = parsed->rbsp;
    // Drop the trailing stop bit and padding: copy the original bits up to the point the VUI tail is replaced.
    BitWriter w;
    if (!info.vui_present)
    {
        w.copy(rbsp, 0, parsed->vui_flag_pos);
        w.bit(1); // vui_parameters_present_flag
        // Minimal VUI: nothing present except bitstream_restriction.
        for (int i = 0; i < 8; ++i)
            w.bit(0); // aspect ratio, overscan, video signal, chroma loc, timing, nal hrd, vcl hrd, pic_struct
        w.bit(1);     // bitstream_restriction_flag
    }
    else
    {
        w.copy(rbsp, 0, parsed->restriction_flag_pos);
        w.bit(1); // bitstream_restriction_flag
    }
    w.bit(1);   // motion_vectors_over_pic_boundaries_flag
    w.ue(0);    // max_bytes_per_pic_denom
    w.ue(1);    // max_bits_per_mb_denom
    w.ue(15);   // log2_max_mv_length_horizontal
    w.ue(15);   // log2_max_mv_length_vertical
    w.ue(0);    // max_num_reorder_frames
    w.ue(std::max<uint32_t>(info.max_num_ref_frames, 1)); // max_dec_frame_buffering
    const std::vector<uint8_t> new_rbsp = w.finish();
    const std::vector<uint8_t> payload = escape(new_rbsp);
    std::vector<uint8_t> out(annexb, annexb + sc);
    out.insert(out.end(), { 0, 0, 0, 1 });
    out.insert(out.end(), payload.begin(), payload.end());
    out.insert(out.end(), annexb + end, annexb + size);
    return out;
}

} // namespace h264
} // namespace video_stage
