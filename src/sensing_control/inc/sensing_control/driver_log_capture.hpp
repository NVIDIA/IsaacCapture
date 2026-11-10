// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace plugins
{
namespace sensing
{

/**
 * @brief Collects what native code prints to stdout and stderr while it runs, for the length of one scope.
 *
 * The camera driver explains why a camera failed only in its own log lines ("lock timeout", "OPEN_CIRCUIT", an I2C
 * NACK); the status code it returns says nothing. This redirects descriptors 1 and 2 into a temporary file for the
 * scope, and on finish() restores them, writes what was captured back to them so the service journal still has it,
 * and returns the lines that matter for a failure. The descriptors are restored on every path, including
 * destruction without finish(). Use it only around a start-up call, never for the process lifetime.
 */
class DriverLogCapture
{
public:
    DriverLogCapture();
    ~DriverLogCapture();

    DriverLogCapture(const DriverLogCapture&) = delete;
    DriverLogCapture& operator=(const DriverLogCapture&) = delete;

    /// Restore the descriptors, replay the captured text, and return the captured lines that look like failures.
    std::string finish();

    /// The failure-looking lines of `text` (an error, failure, timeout, NACK, open circuit), newest last, joined with
    /// "; " and limited in length.
    static std::string interesting_lines(const std::string& text);

private:
    int m_saved_out = -1;
    int m_saved_err = -1;
    int m_file = -1;
    bool m_active = false;
};

} // namespace sensing
} // namespace plugins
