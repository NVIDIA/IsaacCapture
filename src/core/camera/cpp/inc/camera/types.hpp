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

// Larger than any image, and small enough that no size worked out from a
// description can wrap.
inline constexpr std::uint64_t MAX_SLOT_BYTES = std::uint64_t{ 1 } << 32;

// Marks a time that is not known. A frame set's timeline is in nanoseconds on the host's
// monotonic clock; an image's device_time_ns is in its SDK's clock.
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

// Where a stream's slots live, which tells a reader how to map them. The publisher tracks slots by
// index only and never touches their memory; each further kind is added with its pool.
enum class MemoryKind : std::uint32_t
{
    HOST = 1, // shared host memory
    CUDA = 2, // memory of one CUDA device (cuda_pool.hpp)
};

// Names a CUDA device in every process: the driver's UUID for it. All zero names none.
using CudaDeviceUuid = std::array<std::uint8_t, 16>;

// The address of a byte of device memory. It means something only on the device it belongs
// to, in that device's primary context, which is the context CuPy, PyTorch and most SDKs use.
using CudaAddress = std::uintptr_t;

struct StreamDescription
{
    std::string id;
    PixelFormat pixel_format = PixelFormat::BGRA8;
    MemoryKind memory = MemoryKind::HOST;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t plane_count = 0;
    std::array<PlaneLayout, MAX_PLANES_PER_IMAGE> planes; // the same for every slot of the stream
    std::uint64_t slot_bytes = 0; // capacity of one slot
    std::uint32_t slot_count = 0;
    double frame_rate = 0.0; // configured or measured, not the sensor mode's maximum

    // CUDA memory only: the device the stream's pool is on.
    CudaDeviceUuid cuda_device = {};

    // Whether the images are on a GPU, whichever kind of memory carries them there: a reader
    // then has device addresses for them (ImageView::cuda_plane), not host bytes.
    constexpr bool on_gpu() const
    {
        return memory != MemoryKind::HOST;
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
    IMAGE_FLAG_SDK_WARNING = 1u << 0, // the camera SDK delivered the frame but flagged it as suspect
};

struct ImageInfo
{
    std::uint32_t slot = 0;
    std::uint32_t flags = 0; // ImageFlag bits

    // The capture's number: the SDK's, or where it has none the backend's count of the captures
    // the SDK handed over. A capture that is not published keeps its number, so a gap between
    // consecutive frame sets is what was dropped before publication.
    std::uint64_t device_frame_id = 0;

    // The SDK's own time for this image, as it gave it, in a clock the plugin's README names.
    // An SDK that also gives a time on the host's clock has that one in capture_ns.
    std::int64_t device_time_ns = TIME_UNKNOWN;
};

// Where a frame set was at each step, so latency can be read off every frame.
struct FrameTimeline
{
    // When the sensors captured the frame set, where the backend can place that on the host
    // clock: one time for the group, whose streams are captured together. Which moment of the
    // exposure it marks is the SDK's, and is in the plugin's README.
    std::int64_t capture_ns = TIME_UNKNOWN;
    std::int64_t sdk_ready_ns = TIME_UNKNOWN; // the SDK handed the frame to the backend
    std::int64_t published_ns = TIME_UNKNOWN; // set by Publisher::publish
    std::int64_t sent_ns = TIME_UNKNOWN; // set when the event is handed to a reader
};

struct FrameEvent
{
    std::uint64_t lease_id = 0; // this reader's hold on the frame set; return it with Publisher::release
    // The frame set's place among those the group published in this session; a capture that
    // was dropped before publication takes no number. The same for every reader, so readers
    // see gaps rather than a numbering of their own; a restart does not reset it.
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
