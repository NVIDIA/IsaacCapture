// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/h264_access_units.hpp"

namespace video_stage
{

namespace
{

struct Nal
{
    std::size_t start; ///< offset of the start code (including a leading zero byte when it is a 4-byte code)
    int type;
    std::size_t header; ///< offset of the NAL header byte
};

std::vector<Nal> find_nals(const std::uint8_t* data, std::size_t size)
{
    std::vector<Nal> nals;
    std::size_t i = 0;
    while (i + 3 < size)
    {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1)
        {
            const std::size_t start = (i > 0 && data[i - 1] == 0) ? i - 1 : i;
            nals.push_back({ start, data[i + 3] & 0x1F, i + 3 });
            i += 3;
        }
        else
        {
            ++i;
        }
    }
    return nals;
}

/// first_mb_in_slice, the Exp-Golomb code right after the NAL header byte; large values read as nonzero.
std::size_t first_mb_in_slice(const std::uint8_t* data, std::size_t size, std::size_t header)
{
    std::uint32_t bits = 0;
    for (std::size_t k = 0; k < 4; ++k)
        bits = (bits << 8) | (header + 1 + k < size ? data[header + 1 + k] : 0);
    int zeros = 0;
    while (zeros < 15 && !((bits >> (31 - zeros)) & 1))
        ++zeros;
    const std::uint32_t code = (bits >> (32 - (2 * zeros + 1))) & ((1u << (2 * zeros + 1)) - 1);
    return code - 1;
}

bool is_trailing(int type)
{
    return type == 10 || type == 11 || type == 12;
}

} // namespace

std::vector<AccessUnitRange> split_access_units(const std::uint8_t* data, std::size_t size)
{
    const std::vector<Nal> nals = find_nals(data, size);
    std::vector<std::size_t> starts;
    bool previous_vcl = false;
    for (std::size_t n = 0; n < nals.size(); ++n)
    {
        const bool vcl = nals[n].type == 1 || nals[n].type == 5;
        bool begin = false;
        if (n == 0)
            begin = true;
        else if (vcl)
            begin = previous_vcl && first_mb_in_slice(data, size, nals[n].header) == 0;
        else
            begin = previous_vcl && !is_trailing(nals[n].type);
        if (begin)
            starts.push_back(nals[n].start);
        previous_vcl = vcl || (previous_vcl && is_trailing(nals[n].type));
    }
    starts.push_back(size);

    std::vector<AccessUnitRange> units;
    for (std::size_t u = 0; u + 1 < starts.size(); ++u)
    {
        AccessUnitRange unit{ starts[u], starts[u + 1] - starts[u], false };
        for (const auto& nal : nals)
            if (nal.start >= unit.offset && nal.start < unit.offset + unit.size && nal.type == 5)
                unit.idr = true;
        units.push_back(unit);
    }
    return units;
}

} // namespace video_stage
