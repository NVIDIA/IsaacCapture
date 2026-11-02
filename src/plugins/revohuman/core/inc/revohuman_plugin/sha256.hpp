// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace plugins
{
namespace revohuman
{

//! FIPS 180-4 SHA-256 of a byte range, as 64 lowercase hex digits.
std::string sha256_hex(const std::uint8_t* data, std::size_t size);

} // namespace revohuman
} // namespace plugins
