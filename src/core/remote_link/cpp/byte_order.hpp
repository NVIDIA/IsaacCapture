// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace core::remote_link_detail
{

template <typename T>
void write_le(uint8_t* out, T value)
{
    using U = std::make_unsigned_t<T>;
    const U bits = static_cast<U>(value);
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        out[i] = static_cast<uint8_t>(bits >> (8 * i));
    }
}

template <typename T>
T read_le(const uint8_t* in)
{
    using U = std::make_unsigned_t<T>;
    U bits = 0;
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        bits |= static_cast<U>(static_cast<U>(in[i]) << (8 * i));
    }
    return static_cast<T>(bits);
}

} // namespace core::remote_link_detail
