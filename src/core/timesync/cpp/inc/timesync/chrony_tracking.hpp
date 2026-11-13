// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace isaaccapture::timesync
{

/// One `chronyc -c tracking` report (chrony 4.x, 14 CSV fields). Times are seconds, rates are ppm.
struct ChronyTracking
{
    std::string reference_id; ///< Upper-case hex; an NTP source is its IPv4 address, 7F7F0101 is a local reference.
    std::string reference_name;
    int stratum = 0;
    double ref_time_unix_s = 0.0;
    double system_time_s = 0.0; ///< Signed; only the magnitude is used, so the sign convention does not matter.
    double last_offset_s = 0.0;
    double rms_offset_s = 0.0;
    double frequency_ppm = 0.0;
    double residual_freq_ppm = 0.0;
    double skew_ppm = 0.0;
    double root_delay_s = 0.0;
    double root_dispersion_s = 0.0;
    double update_interval_s = 0.0;
    std::string leap_status; ///< "Normal", "Insert second", "Delete second" or "Not synchronised".
};

enum class ChronyParseError
{
    None,
    Empty,
    FieldCount,
    BadNumber, ///< Not a finite number, or a field that cannot be negative is.
    BadStratum,
};

struct ChronyParseResult
{
    std::optional<ChronyTracking> tracking;
    ChronyParseError error = ChronyParseError::None;
};

/// Parses one CSV line. Anything unexpected is an error; nothing is guessed.
ChronyParseResult parse_tracking_csv(std::string_view line);

/// Reference id chrony reports when the host serves its own clock (`local` directive).
inline constexpr std::string_view kChronyLocalReferenceId = "7F7F0101";

/// True when chrony reports a usable reference: a normal or leap-pending status and a stratum of 1 to 15.
/// A local reference passes only when `allow_local_reference` is set, because it says nothing about UTC.
bool is_synchronised(const ChronyTracking& t, bool allow_local_reference);

} // namespace isaaccapture::timesync
