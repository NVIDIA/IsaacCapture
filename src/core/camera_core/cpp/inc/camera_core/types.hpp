// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace core::camera
{

// A group publishes one image per stream, all captured together.
inline constexpr std::size_t MAX_STREAMS_PER_GROUP = 8;
inline constexpr std::uint32_t MAX_SLOTS_PER_STREAM = 64;

// Marks a time that is not known. All times are nanoseconds on the host's monotonic clock.
inline constexpr std::int64_t TIME_UNKNOWN = -1;

enum class PixelFormat : std::uint32_t
{
    BGRA8 = 1,
    RGBA8 = 2,
    NV12 = 3,
    GRAY8 = 4,
};

// Whether a stream's slots hold pixels or one access unit of encoded video each.
enum class Codec : std::uint32_t
{
    NONE = 0, // pixels, laid out as the stream's pixel format says
    H264 = 1,
    H265 = 2,
};

// How a slot's memory reaches another process. The core tracks slots by index only and never
// touches their memory, so a new kind needs no change here.
enum class MemoryKind : std::uint32_t
{
    HOST = 1, // shared host memory, passed as a file descriptor
    CUDA = 2, // CUDA memory on a discrete GPU
    DMA_BUF = 3, // Jetson buffer, passed as a file descriptor
    NVSCI_BUF = 4, // reserved for SIPL; nothing implements it yet
};

struct StreamDescription
{
    std::string id;
    Codec codec = Codec::NONE;
    PixelFormat pixel_format = PixelFormat::BGRA8; // of a pixel stream; unused when the stream is encoded
    MemoryKind memory = MemoryKind::HOST;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t stride_bytes = 0; // row pitch of the first plane; 0 for an encoded stream
    std::uint64_t slot_bytes = 0; // capacity of one slot
    std::uint32_t slot_count = 0;
    double frame_rate = 0.0; // configured or measured, not the sensor mode's maximum

    constexpr bool is_encoded() const
    {
        return codec != Codec::NONE;
    }
};

// One or more hardware-synchronised streams that are published as one frame set.
struct GroupDescription
{
    std::string id;
    std::vector<StreamDescription> streams;
};

enum ImageFlag : std::uint32_t
{
    IMAGE_FLAG_KEYFRAME = 1u << 0, // encoded stream: this access unit decodes on its own
    IMAGE_FLAG_SDK_WARNING = 1u << 1, // the camera SDK delivered the frame but flagged it as suspect
};

struct ImageInfo
{
    std::uint32_t slot = 0;
    std::uint32_t flags = 0; // ImageFlag bits
    std::uint64_t bytes_used = 0; // encoded stream: size of the access unit
    std::uint64_t device_frame_id = 0; // the SDK's own frame number
    std::int64_t device_time_ns = TIME_UNKNOWN; // the SDK's own timestamp, in the SDK's clock
};

// Where a frame set was at each step, so latency can be read off every frame.
struct FrameTimeline
{
    std::int64_t capture_ns = TIME_UNKNOWN; // sensor capture, when the backend can map it to the host clock
    std::int64_t sdk_ready_ns = TIME_UNKNOWN; // the SDK handed the frame to the backend
    std::int64_t published_ns = TIME_UNKNOWN; // set by Publisher::publish
    std::int64_t sent_ns = TIME_UNKNOWN; // set when the event is handed to a reader
};

struct FrameEvent
{
    std::uint64_t lease_id = 0; // return it with Publisher::release
    std::uint64_t sequence = 0; // counts frame sets of the group; never restarts
    std::uint32_t generation = 0; // changes when the camera restarts
    std::uint32_t skipped = 0; // frame sets this reader did not receive since its previous one
    std::uint32_t image_count = 0;
    FrameTimeline timeline;
    std::array<ImageInfo, MAX_STREAMS_PER_GROUP> images; // images[i] belongs to stream i
};

enum class DeliveryOrder
{
    DEFAULT, // NEWEST for a window of one, IN_ORDER otherwise
    NEWEST, // each time the reader has room it gets the newest frame set and skips older ones
    IN_ORDER, // the reader gets every frame set in sequence while it keeps up
};

struct ReaderOptions
{
    std::uint32_t window = 1; // frame sets kept for this reader at a time, unread and held together
    DeliveryOrder order = DeliveryOrder::DEFAULT;
};

} // namespace core::camera
