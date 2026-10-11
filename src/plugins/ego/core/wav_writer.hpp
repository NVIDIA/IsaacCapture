// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "diagnostics.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins::ego
{

class WavWriter
{
public:
    WavWriter() = default;
    ~WavWriter()
    {
        try
        {
            close();
        }
        catch (const std::exception& error)
        {
            detail::log_error() << "Ego WAV shutdown failed: " << error.what() << std::endl;
        }
    }
    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;

    void open(const std::string& path, uint32_t rate, uint16_t channels, uint16_t bits)
    {
        const std::filesystem::path output(path);
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        file_.open(path, std::ios::binary | std::ios::trunc);
        if (!file_)
            throw std::runtime_error("Unable to open WAV output: " + path);
        data_bytes_ = 0;
        rate_ = rate;
        channels_ = channels;
        bits_ = bits;
        write_header(0);
        if (!file_)
            throw std::runtime_error("Failed while writing Ego WAV header");
    }

    uint64_t write(const std::vector<uint8_t>& bytes)
    {
        check_capacity(bytes.size());
        const uint64_t offset = 44 + data_bytes_;
        file_.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!file_)
            throw std::runtime_error("Failed while writing Ego WAV data");
        data_bytes_ += bytes.size();
        return offset;
    }

    bool is_open() const
    {
        return file_.is_open();
    }

    void close()
    {
        if (!file_.is_open())
            return;
        try
        {
            check_capacity(0);
        }
        catch (...)
        {
            file_.close();
            throw;
        }
        file_.seekp(0);
        write_header(static_cast<uint32_t>(data_bytes_));
        file_.flush();
        bool failed = !file_;
        file_.close();
        failed = failed || file_.fail();
        if (failed)
            throw std::runtime_error("Failed while finalizing Ego WAV output");
    }

private:
    void check_capacity(uint64_t additional_bytes) const
    {
        // RIFF's size includes 36 header bytes; reject before writing so archive cleanup can abort.
        constexpr uint64_t max_data_bytes = std::numeric_limits<uint32_t>::max() - 36U;
        if (data_bytes_ > max_data_bytes || additional_bytes > max_data_bytes - data_bytes_)
            throw std::runtime_error("Ego WAV exceeds RIFF 32-bit size limit");
    }

    void le16(uint16_t value)
    {
        const uint8_t bytes[] = { static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8) };
        file_.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    void le32(uint32_t value)
    {
        const uint8_t bytes[] = { static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                                  static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24) };
        file_.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    void write_header(uint32_t data_bytes)
    {
        file_.write("RIFF", 4);
        le32(36 + data_bytes);
        file_.write("WAVEfmt ", 8);
        le32(16);
        le16(1);
        le16(channels_);
        le32(rate_);
        const auto block_align = static_cast<uint16_t>(channels_ * bits_ / 8);
        le32(rate_ * block_align);
        le16(block_align);
        le16(bits_);
        file_.write("data", 4);
        le32(data_bytes);
    }

    std::ofstream file_;
    uint32_t rate_ = 0;
    uint16_t channels_ = 0;
    uint16_t bits_ = 0;
    uint64_t data_bytes_ = 0;
};

} // namespace plugins::ego
