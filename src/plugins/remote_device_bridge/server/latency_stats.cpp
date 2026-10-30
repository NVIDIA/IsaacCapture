// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_device_bridge/latency_stats.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace remote_device_bridge
{

namespace
{

std::runtime_error file_error(const char* what, const std::string& path)
{
    return std::runtime_error(std::string(what) + " stats CSV '" + path + "': " + std::strerror(errno));
}

//! Formats one row (newline included) without allocating; returns the snprintf result.
int format_row(char* out, size_t size, const SampleCsvRow& row)
{
    return std::snprintf(out, size, "%lld,%lld,%lld,%lld,%.3f,%s,%u,%.5f\n", static_cast<long long>(row.push_mono_ns),
                         static_cast<long long>(row.recv_mono_ns), static_cast<long long>(row.capture_sender_ns),
                         static_cast<long long>(row.capture_mapped_ns), static_cast<double>(row.age_ns) / 1e6, row.side,
                         static_cast<unsigned>(row.seq), static_cast<double>(row.curl_m));
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

std::string format_clock_stats(std::vector<int64_t> ages_ns, const remote_plugin_session::ClockOffsetEstimator& clock)
{
    const bool two_way = clock.offset_source() == remote_plugin_session::ClockOffsetEstimator::Source::TwoWay;
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
    return out + " | offset_ms=" + offset + " clock=" + (two_way ? "two-way" : "one-way");
}

SampleCsv::SampleCsv(const std::string& path)
    : m_fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644))
{
    if (m_fd < 0)
    {
        throw file_error("cannot open", path);
    }
    const size_t size = std::strlen(kHeader);
    if (::write(m_fd, kHeader, size) != static_cast<ssize_t>(size))
    {
        const std::runtime_error error = file_error("cannot write", path);
        ::close(m_fd);
        throw error;
    }
}

SampleCsv::~SampleCsv()
{
    ::close(m_fd);
}

bool SampleCsv::write(const SampleCsvRow& row) const
{
    char line[256];
    const int size = format_row(line, sizeof(line), row);
    if (size <= 0 || static_cast<size_t>(size) >= sizeof(line))
    {
        return false;
    }
    // One write(2) per row, so rows from concurrent clients never interleave.
    return ::write(m_fd, line, static_cast<size_t>(size)) == size;
}

} // namespace remote_device_bridge
