// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace plugins::ego::detail
{

class RawCalibrationTransfer
{
public:
    void append(size_t total_size, size_t offset, const uint8_t* data, size_t size) noexcept;
    void complete() noexcept;
    void fail() noexcept;
    std::string finish() const;

private:
    std::string data_;
    size_t total_size_ = 0;
    bool completed_ = false;
    bool failed_ = false;
    std::exception_ptr exception_;
};

} // namespace plugins::ego::detail
