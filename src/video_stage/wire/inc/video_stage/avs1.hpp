// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace video_stage
{
namespace avs1
{

/// AVS1 version 1, one mono stream with one sample per record. The layout is the repository's AVS1 wire format
/// (examples/camera_streamer/argus_sender/WIRE_FORMAT.md): a 4-byte big-endian record length, a 64-byte bundle
/// header, one 40-byte stream descriptor, one 72-byte sample descriptor, then the payload. Two things are specific
/// to this use:
///  - codec 2 is H.264: one Annex-B access unit per record, with SPS and PPS before every IDR;
///  - `frame_number` is the source frame id and `frame_timestamp_ns` carries the source stamp on the sender's own
///    monotonic clock, the same value the LiveKit track publishes. Neither is a wall-clock time. A zero stamp and a
///    zero frame number mean the sender could not tie the unit to a capture: the receiver publishes it without a
///    capture time.
constexpr uint16_t kVersion = 1;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kStreamBytes = 40;
constexpr uint32_t kSampleBytes = 72;
constexpr uint32_t kMaxRecordBytes = 64U * 1024U * 1024U;

enum class Codec : uint8_t
{
    RawNv12 = 0,
    Hevc = 1,
    H264 = 2,
};

/// What one record carries: the picture description and its identity.
struct Sample
{
    Codec codec = Codec::H264;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps_numerator = 30;
    uint64_t bundle_sequence = 0;
    uint64_t session_id = 0;
    uint64_t frame_number = 0;
    uint64_t frame_timestamp_ns = 0;
};

/// A parsed record. `payload` points into the buffer given to parse(); it is valid while that buffer is.
struct Record
{
    Sample sample;
    const uint8_t* payload = nullptr;
    size_t payload_size = 0;
};

/// Appends one complete record (length prefix included) to `out`. `payload` is one access unit.
void encode(const Sample& sample, const uint8_t* payload, size_t payload_size, std::vector<uint8_t>& out);

/// Parses the bytes after the 4-byte length prefix. Returns nullopt and sets `error` when the record is not a valid
/// version-1 mono H.264 record: bad magic or version, nonzero reserved fields, unknown flags, inconsistent presence,
/// a payload range outside the record, or an empty access unit.
std::optional<Record> parse(const uint8_t* data, size_t size, std::string& error);

/// The record length from its 4-byte big-endian prefix.
uint32_t read_length(const uint8_t prefix[4]);

/// True when an Annex-B access unit contains an IDR slice (NAL type 5).
bool contains_idr(const uint8_t* au, size_t size);

} // namespace avs1
} // namespace video_stage
