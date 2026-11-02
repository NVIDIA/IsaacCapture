// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace plugins::manus::calibration
{

// Lowercase hex SHA-256 of `data`, used to let the supervisor verify an exported file.
std::string sha256_hex(const std::vector<uint8_t>& data);

} // namespace plugins::manus::calibration
