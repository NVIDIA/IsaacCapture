// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <video_types/timestamp_basis.hpp>

#include <iostream>

namespace
{

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

} // namespace

int main()
{
    using camera_bridge::TimestampBasis;

    check(camera_bridge::parse_timestamp_basis("monotonic_us") == TimestampBasis::MonotonicUs, "parse monotonic_us");
    check(camera_bridge::parse_timestamp_basis("capture_monotonic_us") == TimestampBasis::CaptureMonotonicUs,
          "parse capture_monotonic_us");
    check(!camera_bridge::parse_timestamp_basis("capture_unix_us").has_value(), "reject a Unix-time basis");
    check(!camera_bridge::parse_timestamp_basis("processing_unix_us").has_value(), "reject an unsupported basis");
    check(!camera_bridge::parse_timestamp_basis("").has_value(), "reject an empty basis");

    if (g_failures == 0)
        std::cout << "timestamp_basis: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
