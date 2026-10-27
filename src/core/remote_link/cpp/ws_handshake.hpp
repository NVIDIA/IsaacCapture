// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace core::remote_link_detail
{

std::array<uint8_t, 20> sha1(const std::string& text);
std::string base64(const uint8_t* data, size_t size);
//! Sec-WebSocket-Accept for a Sec-WebSocket-Key (RFC 6455 section 4.2.2).
std::string websocket_accept_key(const std::string& key);

} // namespace core::remote_link_detail
