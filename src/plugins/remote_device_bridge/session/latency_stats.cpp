// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/latency_stats.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace remote_plugin_session
{

namespace
{

std::FILE* open_or_throw(const std::string& path, const char* mode)
{
    std::FILE* file = std::fopen(path.c_str(), mode);
    if (file == nullptr)
    {
        throw std::runtime_error("--stats-csv: cannot open '" + path + "': " + std::strerror(errno));
    }
    return file;
}

std::string ms(int64_t ns)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", static_cast<double>(ns) / 1e6);
    return text;
}

//! Nearest-rank percentile; reorders `values`.
int64_t percentile(std::vector<int64_t>& values, double p)
{
    const size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(values.size())));
    const auto nth = values.begin() + static_cast<std::ptrdiff_t>(std::max<size_t>(rank, 1) - 1);
    std::nth_element(values.begin(), nth, values.end());
    return *nth;
}

} // namespace

float index_curl_m(const XrHandJointLocationEXT* joints)
{
    const XrVector3f& tip = joints[XR_HAND_JOINT_INDEX_TIP_EXT].pose.position;
    const XrVector3f& wrist = joints[XR_HAND_JOINT_WRIST_EXT].pose.position;
    const float dx = tip.x - wrist.x;
    const float dy = tip.y - wrist.y;
    const float dz = tip.z - wrist.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::string format_clock_stats(std::vector<int64_t>& ages_ns, const ClockOffsetEstimator& clock)
{
    const bool two_way = std::strcmp(clock.offset_source(), "two-way") == 0;
    std::string out = " | age_ms";
    if (ages_ns.empty())
    {
        out += " p50=- p99=- max=-";
    }
    else
    {
        out += " p50=" + ms(percentile(ages_ns, 0.50)) + " p99=" + ms(percentile(ages_ns, 0.99)) +
               " max=" + ms(*std::max_element(ages_ns.begin(), ages_ns.end()));
    }
    out += two_way ? " | rtt_ms min=" + ms(clock.rtt_min_ns()) + " last=" + ms(clock.rtt_last_ns()) :
                     std::string(" | rtt_ms min=- last=-");
    char offset[48];
    std::snprintf(offset, sizeof(offset), "%.3f", static_cast<double>(clock.offset_ns()) / 1e6);
    return out + " | offset_ms=" + offset + " clock=" + clock.offset_source();
}

void SampleCsv::create(const std::string& path)
{
    std::FILE* file = open_or_throw(path, "w");
    std::fputs(kHeader, file);
    std::fclose(file);
}

int SampleCsv::format(char* out, size_t size, const SampleCsvRow& row)
{
    return std::snprintf(out, size, "%lld,%lld,%lld,%lld,%.3f,%s,%u,%.5f\n", static_cast<long long>(row.push_mono_ns),
                         static_cast<long long>(row.recv_mono_ns), static_cast<long long>(row.capture_sender_ns),
                         static_cast<long long>(row.capture_mapped_ns), static_cast<double>(row.age_ns) / 1e6, row.side,
                         static_cast<unsigned>(row.seq), static_cast<double>(row.curl_m));
}

SampleCsv::SampleCsv(const std::string& path) : m_file(open_or_throw(path, "a"))
{
}

SampleCsv::~SampleCsv()
{
    std::fclose(m_file);
}

void SampleCsv::write(const SampleCsvRow& row)
{
    char line[256];
    const int size = format(line, sizeof(line), row);
    if (size > 0)
    {
        std::fwrite(line, 1, std::min(static_cast<size_t>(size), sizeof(line) - 1), m_file);
    }
}

} // namespace remote_plugin_session
