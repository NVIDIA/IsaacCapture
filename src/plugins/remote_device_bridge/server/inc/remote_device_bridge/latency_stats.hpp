// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>
#include <remote_plugin_session/clock_offset_estimator.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace remote_device_bridge
{

//! Index-tip to wrist distance in metres: the step-test signal of the latency tools.
float index_curl_m(const XrHandJointLocationEXT* joints);

//! " | age_ms p50=.. p99=.. max=.. | rtt_ms min=.. last=.. | offset_ms=.. clock=.." for the bridge's status line;
//! "-" marks a value with no data.
std::string format_clock_stats(std::vector<int64_t> ages_ns, const remote_plugin_session::ClockOffsetEstimator& clock);

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
 * @brief --stats-csv writer.
 *
 * The bridge parent opens it once; forked clients inherit the descriptor and append each row with one write(2)
 * (O_APPEND), so rows from concurrent clients never split.
 */
class SampleCsv
{
public:
    static constexpr const char* kHeader =
        "push_mono_ns,recv_mono_ns,capture_sender_ns,capture_mapped_ns,age_ms,side,seq,curl\n";

    //! Truncates `path`, writes the header and keeps it open for appending; throws std::runtime_error.
    explicit SampleCsv(const std::string& path);

    void write(const SampleCsvRow& row) const;

private:
    struct FileCloser
    {
        void operator()(std::FILE* file) const noexcept
        {
            std::fclose(file);
        }
    };

    std::unique_ptr<std::FILE, FileCloser> m_file;
};

} // namespace remote_device_bridge
