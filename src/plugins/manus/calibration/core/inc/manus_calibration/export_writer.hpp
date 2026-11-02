// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plugins::manus::calibration
{

// Where exported MCAL bytes land. An interface so tests can inject write failures.
class ExportWriter
{
public:
    virtual ~ExportWriter() = default;

    // True when `dir` is an existing directory the worker may create files in.
    virtual bool directory_usable(const std::string& dir) = 0;

    // Writes `bytes` as `dir/basename`, durably and atomically. Returns an error message on failure.
    virtual std::optional<std::string> write(const std::string& dir,
                                             const std::string& basename,
                                             const std::vector<uint8_t>& bytes) = 0;
};

// Writes `<basename>.part`, fsyncs it, renames it into place and fsyncs the directory, so a crash can
// never leave a truncated file under the final name.
class PosixExportWriter final : public ExportWriter
{
public:
    bool directory_usable(const std::string& dir) override;
    std::optional<std::string> write(const std::string& dir,
                                     const std::string& basename,
                                     const std::vector<uint8_t>& bytes) override;
};

} // namespace plugins::manus::calibration
