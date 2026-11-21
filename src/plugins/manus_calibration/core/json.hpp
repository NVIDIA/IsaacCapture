// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace plugins::manus::calibration::json
{

// The only value shapes the command protocol uses: a string, an integer, or an array of strings.
using Value = std::variant<std::string, int64_t, std::vector<std::string>>;
using Members = std::vector<std::pair<std::string, Value>>;

struct ParseResult
{
    std::optional<Members> members;
    std::string error;
};

// Parses one flat JSON object. Rejects duplicate keys, nested objects, floats, booleans, null and any
// trailing bytes, so a malformed command can never be half-accepted.
ParseResult parse_object(std::string_view text);

// Appends `text` as a quoted JSON string. Bytes that are not valid UTF-8 become U+FFFD, so vendor
// strings can never produce an invalid protocol line.
void append_string(std::string& out, std::string_view text);

} // namespace plugins::manus::calibration::json
