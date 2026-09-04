// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/**
 * @brief Raw H.264 file writer
 *
 * Writes H.264 NAL units directly to a file without container.
 * File opens in constructor and closes in destructor (RAII).
 */
class RawDataWriter
{
public:
    /**
     * @brief Construct the writer and open the file.
     * @param path Output file path. Must not be empty.
     * @throws std::runtime_error if the file cannot be opened.
     */
    explicit RawDataWriter(const std::string& path);
    ~RawDataWriter();

    // Non-copyable, non-movable
    RawDataWriter(const RawDataWriter&) = delete;
    RawDataWriter& operator=(const RawDataWriter&) = delete;

    void write(const std::vector<uint8_t>& data);

private:
    std::ofstream m_file;
};

} // namespace sensing
} // namespace plugins
