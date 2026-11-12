// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/timesync/chrony_tracking.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <vector>

namespace isaaccapture::timesync
{

namespace
{

constexpr size_t kFieldCount = 14;

std::string_view trim(std::string_view s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    {
        s.remove_suffix(1);
    }
    return s;
}

bool parse_double(std::string_view s, double& out)
{
    s = trim(s);
    if (s.empty())
    {
        return false;
    }
    if (s.front() == '+')
    {
        s.remove_prefix(1);
    }
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc() && end == s.data() + s.size() && std::isfinite(out);
}

bool parse_int(std::string_view s, long long& out)
{
    s = trim(s);
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return !s.empty() && ec == std::errc() && end == s.data() + s.size();
}

std::string upper(std::string_view s)
{
    std::string r(trim(s));
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::toupper(c); });
    return r;
}

} // namespace

ChronyParseResult parse_tracking_csv(std::string_view line)
{
    line = trim(line);
    if (line.empty())
    {
        return { std::nullopt, ChronyParseError::Empty };
    }

    std::vector<std::string_view> f;
    size_t start = 0;
    for (;;)
    {
        const size_t comma = line.find(',', start);
        if (comma == std::string_view::npos)
        {
            f.push_back(line.substr(start));
            break;
        }
        f.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    if (f.size() != kFieldCount)
    {
        return { std::nullopt, ChronyParseError::FieldCount };
    }

    ChronyTracking t;
    t.reference_id = upper(f[0]);
    t.reference_name = std::string(trim(f[1]));
    t.leap_status = std::string(trim(f[13]));

    long long stratum = 0;
    if (!parse_int(f[2], stratum))
    {
        return { std::nullopt, ChronyParseError::BadNumber };
    }
    if (stratum < 0 || stratum > 255)
    {
        return { std::nullopt, ChronyParseError::BadStratum };
    }
    t.stratum = static_cast<int>(stratum);

    double* const numeric[] = {
        &t.ref_time_unix_s,   &t.system_time_s, &t.last_offset_s, &t.rms_offset_s,      &t.frequency_ppm,
        &t.residual_freq_ppm, &t.skew_ppm,      &t.root_delay_s,  &t.root_dispersion_s, &t.update_interval_s,
    };
    for (size_t i = 0; i < std::size(numeric); ++i)
    {
        if (!parse_double(f[3 + i], *numeric[i]))
        {
            return { std::nullopt, ChronyParseError::BadNumber };
        }
    }
    if (t.rms_offset_s < 0.0 || t.skew_ppm < 0.0 || t.root_delay_s < 0.0 || t.root_dispersion_s < 0.0 ||
        t.update_interval_s < 0.0)
    {
        return { std::nullopt, ChronyParseError::BadNumber };
    }
    return { std::move(t), ChronyParseError::None };
}

bool is_synchronised(const ChronyTracking& t, bool allow_local_reference)
{
    const bool leap_ok =
        t.leap_status == "Normal" || t.leap_status == "Insert second" || t.leap_status == "Delete second";
    if (!leap_ok || t.stratum < 1 || t.stratum > 15 || t.reference_id == "00000000" || t.reference_id.empty())
    {
        return false;
    }
    return allow_local_reference || t.reference_id != kChronyLocalReferenceId;
}

} // namespace isaaccapture::timesync
