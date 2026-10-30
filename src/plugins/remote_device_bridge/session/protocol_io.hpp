// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/remote_plugin_session/protocol_frame.hpp"

#include <openxr/openxr.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace remote_plugin_protocol
{
namespace detail
{

//! Stores the low `size` bytes of `value` little-endian at `out` (in-place patching).
inline void store_le(uint8_t* out, uint64_t value, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

//! Writes the header into the first kHeaderSize bytes of `out`.
void encode_header(const FrameHeader& header, uint8_t* out);

//! Appends little-endian fields to a caller-owned buffer.
class Writer
{
public:
    explicit Writer(std::vector<uint8_t>& out) : m_out(out)
    {
    }

    void u8(uint8_t value)
    {
        m_out.push_back(value);
    }

    void boolean(bool value)
    {
        u8(value ? 1 : 0);
    }

    void u16(uint16_t value)
    {
        for (int i = 0; i < 2; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void u32(uint32_t value)
    {
        for (int i = 0; i < 4; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void u64(uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void i64(int64_t value)
    {
        u64(static_cast<uint64_t>(value));
    }

    void f32(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    void str(const std::string& value)
    {
        if (value.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("protocol: string too long");
        }
        u32(static_cast<uint32_t>(value.size()));
        m_out.insert(m_out.end(), value.begin(), value.end());
    }

    void bytes(const uint8_t* data, size_t size)
    {
        m_out.insert(m_out.end(), data, data + size);
    }

    void pose(const XrPosef& value)
    {
        f32(value.orientation.x);
        f32(value.orientation.y);
        f32(value.orientation.z);
        f32(value.orientation.w);
        f32(value.position.x);
        f32(value.position.y);
        f32(value.position.z);
    }

private:
    std::vector<uint8_t>& m_out;
};

class Reader
{
public:
    //! `what` names the frame or header being read in DecodeError messages.
    Reader(const uint8_t* data, size_t size, const char* what) : m_data(data), m_size(size), m_what(what)
    {
    }

    Reader(const std::vector<uint8_t>& payload, const char* what) : Reader(payload.data(), payload.size(), what)
    {
    }

    uint8_t u8()
    {
        need(1);
        return m_data[m_pos++];
    }

    bool boolean()
    {
        const uint8_t value = u8();
        if (value > 1)
        {
            throw DecodeError(std::string("protocol: ") + m_what + ": invalid boolean value " + std::to_string(value));
        }
        return value == 1;
    }

    uint16_t u16()
    {
        need(2);
        uint16_t value = 0;
        for (int i = 0; i < 2; ++i)
        {
            value = static_cast<uint16_t>(value | (static_cast<uint16_t>(m_data[m_pos++]) << (8 * i)));
        }
        return value;
    }

    uint32_t u32()
    {
        need(4);
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            value |= static_cast<uint32_t>(m_data[m_pos++]) << (8 * i);
        }
        return value;
    }

    uint64_t u64()
    {
        need(8);
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
        {
            value |= static_cast<uint64_t>(m_data[m_pos++]) << (8 * i);
        }
        return value;
    }

    int64_t i64()
    {
        return static_cast<int64_t>(u64());
    }

    float f32()
    {
        const uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string str()
    {
        const uint32_t size = u32();
        need(size);
        std::string value(reinterpret_cast<const char*>(m_data + m_pos), size);
        m_pos += size;
        return value;
    }

    std::vector<uint8_t> bytes(size_t size)
    {
        need(size);
        std::vector<uint8_t> value(m_data + m_pos, m_data + m_pos + size);
        m_pos += size;
        return value;
    }

    XrPosef pose()
    {
        XrPosef value;
        value.orientation.x = f32();
        value.orientation.y = f32();
        value.orientation.z = f32();
        value.orientation.w = f32();
        value.position.x = f32();
        value.position.y = f32();
        value.position.z = f32();
        return value;
    }

    void finish() const
    {
        if (m_pos != m_size)
        {
            throw DecodeError(std::string("protocol: ") + m_what + ": " + std::to_string(m_size - m_pos) +
                              " trailing bytes");
        }
    }

private:
    void need(size_t count) const
    {
        if (m_size - m_pos < count)
        {
            throw DecodeError(std::string("protocol: ") + m_what + ": truncated");
        }
    }

    const uint8_t* m_data;
    size_t m_size;
    const char* m_what;
    size_t m_pos = 0;
};

//! Reserves the header, writes the payload behind it, then fills the header in with the resulting size.
template <typename WritePayload>
inline std::vector<uint8_t> build_frame(FrameType type, uint16_t channel_id, size_t payload_hint, WritePayload write_payload)
{
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + payload_hint);
    out.resize(kHeaderSize);
    Writer writer(out);
    write_payload(writer);
    if (out.size() - kHeaderSize > kMaxPayloadSize)
    {
        throw std::invalid_argument("protocol: payload too large");
    }
    FrameHeader header;
    header.type = type;
    header.channel_id = channel_id;
    header.payload_size = static_cast<uint32_t>(out.size() - kHeaderSize);
    encode_header(header, out.data());
    return out;
}

} // namespace detail
} // namespace remote_plugin_protocol
