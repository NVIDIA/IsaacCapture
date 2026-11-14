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

// Larger than any image or access unit, and small enough that no size worked out from a
// description can wrap.
inline constexpr std::uint64_t MAX_SLOT_BYTES = std::uint64_t{ 1 } << 32;

// Marks a time that is not known. All times are nanoseconds on the host's monotonic clock.
inline constexpr std::int64_t TIME_UNKNOWN = -1;

enum class PixelFormat : std::uint32_t
{
    BGRA8 = 1,
    RGBA8 = 2,
    NV12 = 3,
    GRAY8 = 4,
};

// NV12 has two planes (luma, then interleaved chroma); the other formats have one.
inline constexpr std::size_t MAX_PLANES_PER_IMAGE = 2;

constexpr std::uint32_t plane_count_of(PixelFormat format)
{
    return format == PixelFormat::NV12 ? 2 : 1;
}

// Where one plane of an image sits inside a slot.
struct PlaneLayout
{
    std::uint32_t offset_bytes = 0;
    std::uint32_t stride_bytes = 0; // from the start of one row to the start of the next
};

// The pixels of one plane: the bytes one row holds, without padding, and the number of rows.
struct PlaneSize
{
    std::uint64_t row_bytes = 0;
    std::uint64_t rows = 0;
};

constexpr PlaneSize plane_size_of(PixelFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t plane)
{
    const std::uint64_t columns = width;
    const std::uint64_t rows = height;
    switch (format)
    {
    case PixelFormat::BGRA8:
    case PixelFormat::RGBA8:
        return { columns * 4, rows };
    case PixelFormat::GRAY8:
        return { columns, rows };
    case PixelFormat::NV12:
        // Luma at full size, then one interleaved chroma pair per two by two pixels.
        return plane == 0 ? PlaneSize{ columns, rows } : PlaneSize{ (columns + 1) / 2 * 2, (rows + 1) / 2 };
    }
    return {};
}

// Whether a stream's slots hold pixels or one access unit of encoded video each.
enum class Codec : std::uint32_t
{
    NONE = 0, // pixels, laid out as the stream's pixel format says
    H264 = 1,
    H265 = 2,
};

// Where a stream's slots live, which tells a reader how to map them. The core tracks slots by
// index only and never touches their memory; each further kind is added with its pool.
enum class MemoryKind : std::uint32_t
{
    HOST = 1, // shared host memory
};

struct StreamDescription
{
    std::string id;
    Codec codec = Codec::NONE;
    PixelFormat pixel_format = PixelFormat::BGRA8; // of a pixel stream; unused when the stream is encoded
    MemoryKind memory = MemoryKind::HOST;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t plane_count = 0; // 0 for an encoded stream
    std::array<PlaneLayout, MAX_PLANES_PER_IMAGE> planes; // the same for every slot of the stream
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

    // Names one run of the group: the same for every frame set from the moment the plugin
    // starts publishing it until that plugin ends. With a frame set's sequence it identifies a
    // capture for good. Left 0, Publisher::add_group draws one.
    std::uint64_t session_id = 0;

    std::vector<StreamDescription> streams;
};

/**
 * @brief Why a group cannot be published or read as described: sizes, the planes its pixel
 * format needs, strides and offsets against the slot, slot counts. For both sides of a
 * connection: what a backend declares and what a reader was sent.
 * @return The reason, or nothing when the description is usable.
 */
std::string problem_with(const GroupDescription& group);

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
    std::uint64_t lease_id = 0; // this reader's hold on the frame set; return it with Publisher::release
    // The frame set's place among the group's captures in this session. The same for every
    // reader, so readers see gaps rather than a numbering of their own; a restart does not reset it.
    std::uint64_t sequence = 0;
    std::uint32_t generation = 0; // restarts of the camera within the session
    std::uint32_t skipped = 0; // frame sets this reader did not receive since its previous one
    std::uint32_t image_count = 0;
    FrameTimeline timeline;
    std::array<ImageInfo, MAX_STREAMS_PER_GROUP> images; // images[i] belongs to stream i
};

enum class DeliveryOrder
{
    DEFAULT, // NEWEST for a window of one, IN_ORDER otherwise
    NEWEST, // each time the reader asks it gets the newest frame set there is then, and skips older ones
    IN_ORDER, // the reader is sent every frame set in sequence while it keeps up
};

struct ReaderOptions
{
    std::uint32_t window = 1; // frame sets kept for this reader at a time, unread and held together
    DeliveryOrder order = DeliveryOrder::DEFAULT;
};

// The order a reader with these options gets: what it asked for, or the default for its window.
constexpr DeliveryOrder resolved_order(const ReaderOptions& options)
{
    if (options.order != DeliveryOrder::DEFAULT)
    {
        return options.order;
    }
    return options.window == 1 ? DeliveryOrder::NEWEST : DeliveryOrder::IN_ORDER;
}

} // namespace core::camera
