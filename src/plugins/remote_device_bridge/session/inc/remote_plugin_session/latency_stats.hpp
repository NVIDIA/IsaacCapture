// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "clock_offset_estimator.hpp"

#include <openxr/openxr.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace remote_plugin_session
{

//! Index-tip to wrist distance in metres: the step-test signal shared with tools/latency_probe_xrclient.py.
float index_curl_m(const XrHandJointLocationEXT* joints);

//! " | age_ms p50=.. p99=.. max=.. | rtt_ms min=.. last=.. | offset_ms=.. clock=.." for the bridge's 5 s line.
//! Reorders `ages_ns`; "-" marks a value with no data.
std::string format_clock_stats(std::vector<int64_t>& ages_ns, const ClockOffsetEstimator& clock);

//! One pushed hand sample; all times are bridge CLOCK_MONOTONIC ns except capture_sender_ns.
struct SampleCsvRow
{
    int64_t push_mono_ns = 0; //!< after the OpenXR push returned
    int64_t recv_mono_ns = 0; //!< frame read off the transport
    int64_t capture_sender_ns = 0;
    int64_t capture_mapped_ns = 0;
    int64_t age_ns = 0; //!< staleness check time - capture_mapped_ns
    const char* side = "";
    uint32_t seq = 0;
    float curl_m = 0.0f;
};

/*!
 * @brief --stats-csv writer: buffered FILE*, flushed on destruction.
 *
 * The bridge parent calls create() once (truncate + header); every forked client appends rows, so concurrent
 * clients may interleave partial rows when a buffer flush splits one.
 */
class SampleCsv
{
public:
    static constexpr const char* kHeader =
        "push_mono_ns,recv_mono_ns,capture_sender_ns,capture_mapped_ns,age_ms,side,seq,curl\n";

    //! Truncates `path` and writes the header; throws std::runtime_error.
    static void create(const std::string& path);
    //! Formats one row (newline included) without allocating; returns the snprintf result.
    static int format(char* out, size_t size, const SampleCsvRow& row);

    //! Opens `path` for appending; throws std::runtime_error.
    explicit SampleCsv(const std::string& path);
    ~SampleCsv();

    SampleCsv(const SampleCsv&) = delete;
    SampleCsv& operator=(const SampleCsv&) = delete;

    void write(const SampleCsvRow& row);

private:
    std::FILE* m_file;
};

} // namespace remote_plugin_session
