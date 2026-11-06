// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/avs1.hpp"

#include <cstring>

namespace video_stage
{
namespace avs1
{

namespace
{

constexpr uint16_t kStreamPresent = 0x0001;
constexpr uint16_t kSamplePresent = 0x0001;
constexpr uint16_t kSampleFrameTimestampValid = 0x0004;
constexpr uint16_t kSampleKnownFlags = kSamplePresent | 0x0002 | kSampleFrameTimestampValid | 0x0008;

void put16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

void put32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int shift = 24; shift >= 0; shift -= 8)
        out.push_back(static_cast<uint8_t>(v >> shift));
}

void put64(std::vector<uint8_t>& out, uint64_t v)
{
    for (int shift = 56; shift >= 0; shift -= 8)
        out.push_back(static_cast<uint8_t>(v >> shift));
}

uint16_t get16(const uint8_t* p)
{
    return static_cast<uint16_t>((p[0] << 8) | p[1]);
}

uint32_t get32(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

uint64_t get64(const uint8_t* p)
{
    return (static_cast<uint64_t>(get32(p)) << 32) | get32(p + 4);
}

std::optional<Record> fail(std::string& error, const char* why)
{
    error = why;
    return std::nullopt;
}

} // namespace

void encode(const Sample& s, const uint8_t* payload, size_t payload_size, std::vector<uint8_t>& out)
{
    const uint32_t descriptor_bytes = kStreamBytes + kSampleBytes;
    const uint32_t record_length = kHeaderBytes + descriptor_bytes + static_cast<uint32_t>(payload_size);
    out.reserve(out.size() + 4 + record_length);
    put32(out, record_length);

    // Bundle header (64 bytes).
    out.insert(out.end(), { 'A', 'V', 'S', '1' });
    put16(out, kVersion);
    put16(out, 0); // bundle flags: no sender wall-clock fields
    put32(out, kHeaderBytes);
    put16(out, 1); // stream_count
    put16(out, 1); // sample_count
    put64(out, s.bundle_sequence);
    put64(out, s.session_id);
    put64(out, 0); // sender_capture_ready_ns
    put64(out, 0); // sender_tcp_enqueue_ns
    put32(out, descriptor_bytes);
    put32(out, static_cast<uint32_t>(payload_size));
    put64(out, 0); // sender_capture_ready_tsc_ns

    // Stream descriptor (40 bytes).
    put16(out, 0); // stream_index
    put16(out, kStreamPresent);
    out.push_back(static_cast<uint8_t>(s.codec));
    out.push_back(1); // pixel_format NV12
    put16(out, 0); // rotation
    put32(out, s.width);
    put32(out, s.height);
    put32(out, s.fps_numerator);
    put32(out, 1); // fps_denominator
    put64(out, s.bundle_sequence); // stream_sequence
    put16(out, 0); // first_sample
    put16(out, 1); // sample_count
    put32(out, 0); // reserved

    // Sample descriptor (72 bytes).
    put16(out, 0); // stream_index
    out.push_back(0); // camera_index
    out.push_back(0); // role: mono
    put16(out, kSamplePresent | kSampleFrameTimestampValid);
    put16(out, 0); // reserved
    put32(out, 0); // device_index
    put32(out, 0); // sensor_mode_index
    put64(out, s.frame_number);
    put64(out, 0); // sof_tsc_ns
    put64(out, s.frame_timestamp_ns);
    put64(out, 0); // sender_capture_ready_ns
    put64(out, 0); // sender_process_submit_ns
    put64(out, 0); // sender_process_done_ns
    put32(out, 0); // payload_offset
    put32(out, static_cast<uint32_t>(payload_size));

    out.insert(out.end(), payload, payload + payload_size);
}

uint32_t read_length(const uint8_t prefix[4])
{
    return get32(prefix);
}

std::optional<Record> parse(const uint8_t* data, size_t size, std::string& error)
{
    constexpr size_t kFixed = kHeaderBytes + kStreamBytes + kSampleBytes;
    if (size < kFixed)
        return fail(error, "record shorter than its descriptors");
    if (std::memcmp(data, "AVS1", 4) != 0)
        return fail(error, "bad magic");
    if (get16(data + 4) != kVersion)
        return fail(error, "unsupported version");
    if (get16(data + 6) != 0)
        return fail(error, "unknown bundle flags");
    if (get32(data + 8) != kHeaderBytes)
        return fail(error, "bad header size");
    if (get16(data + 12) != 1 || get16(data + 14) != 1)
        return fail(error, "this stage takes one stream with one sample");
    const uint32_t descriptor_bytes = get32(data + 48);
    const uint32_t payload_bytes = get32(data + 52);
    if (descriptor_bytes != kStreamBytes + kSampleBytes)
        return fail(error, "bad descriptor size");
    if (size != kFixed + payload_bytes)
        return fail(error, "payload size does not match the record length");
    if (get64(data + 32) != 0 || get64(data + 40) != 0 || get64(data + 56) != 0)
        return fail(error, "sender wall-clock fields must be zero without their flags");

    const uint8_t* stream = data + kHeaderBytes;
    if (get16(stream + 2) != kStreamPresent)
        return fail(error, "stream not present or unknown stream flags");
    if (stream[4] != static_cast<uint8_t>(Codec::H264))
        return fail(error, "codec is not H.264");
    if (stream[5] != 1)
        return fail(error, "pixel format is not NV12");
    if (get16(stream + 6) != 0)
        return fail(error, "rotation not supported");
    if (get32(stream + 20) == 0)
        return fail(error, "zero fps denominator");
    if (get16(stream + 32) != 0 || get16(stream + 34) != 1 || get32(stream + 36) != 0)
        return fail(error, "bad sample indexing");

    const uint8_t* sample = stream + kStreamBytes;
    const uint16_t flags = get16(sample + 4);
    if ((flags & ~kSampleKnownFlags) != 0)
        return fail(error, "unknown sample flags");
    if ((flags & kSamplePresent) == 0 || (flags & kSampleFrameTimestampValid) == 0)
        return fail(error, "sample must be present with a frame timestamp");
    if (get16(sample + 6) != 0)
        return fail(error, "reserved sample field is nonzero");
    const uint32_t payload_offset = get32(sample + 64);
    const uint32_t payload_length = get32(sample + 68);
    if (payload_length == 0 || payload_offset != 0 || payload_length != payload_bytes)
        return fail(error, "payload range is empty, offset, or does not cover the payload");

    Record record;
    record.sample.codec = Codec::H264;
    record.sample.width = get32(stream + 8);
    record.sample.height = get32(stream + 12);
    record.sample.fps_numerator = get32(stream + 16);
    record.sample.bundle_sequence = get64(data + 16);
    record.sample.session_id = get64(data + 24);
    record.sample.frame_number = get64(sample + 16);
    record.sample.frame_timestamp_ns = get64(sample + 32);
    record.payload = data + kFixed;
    record.payload_size = payload_length;
    return record;
}

bool contains_idr(const uint8_t* au, size_t size)
{
    for (size_t i = 0; i + 3 < size; ++i)
    {
        if (au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1)
        {
            if ((au[i + 3] & 0x1F) == 5)
                return true;
            i += 2;
        }
    }
    return false;
}

} // namespace avs1
} // namespace video_stage
