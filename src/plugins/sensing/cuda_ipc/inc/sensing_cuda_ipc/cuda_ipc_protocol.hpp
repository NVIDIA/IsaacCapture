// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Wire format for the CUDA frame IPC socket.
//
// The Python consumer decodes these with struct.unpack in
// examples/camera_viz/sources/cuda_ipc.py. Both sides hardcode the layout, so
// any change here is a wire break: bump kProtocolVersion and update the
// consumer's format strings in the same commit.
//
// Layout is fixed little-endian with explicit padding rather than whatever the
// compiler picks, so the Python side can spell it as a plain struct format.

#pragma once

#include <cstdint>

namespace plugins
{
namespace sensing
{
namespace ipc
{

constexpr uint32_t kHelloMagic = 0x44554349; // 'ICUD'
constexpr uint32_t kFrameMagic = 0x4d524649; // 'IFRM'
constexpr uint32_t kReleaseMagic = 0x4c455249; // 'IREL'
constexpr uint32_t kProtocolVersion = 1;

/// Pixel format tag carried in Hello::format.
enum class PixelFormat : uint32_t
{
    Rgba8 = 0,
};

/**
 * @brief Sent once on connect, alongside the export fd via SCM_RIGHTS.
 *
 * The single fd maps one allocation holding `slot_count` frames; slot i starts
 * at `i * slot_stride` and each row is `pitch` bytes.
 */
struct Hello
{
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t slot_count;
    uint32_t device_id;
    uint32_t sensor_id;
    uint64_t pitch;
    uint64_t slot_stride;
    /// Total mapped size, already rounded up to the CUDA allocation granularity.
    uint64_t total_bytes;
};
static_assert(sizeof(Hello) == 56, "Hello layout is wire format; see cuda_ipc.py");

/// Sent per published frame. The named slot is the consumer's until it releases it.
struct FrameReady
{
    uint32_t magic;
    uint32_t slot;
    uint64_t sequence;
    uint64_t timestamp_ns;
};
static_assert(sizeof(FrameReady) == 24, "FrameReady layout is wire format; see cuda_ipc.py");

/// Consumer -> producer: the slot is done being read and may be overwritten.
struct SlotRelease
{
    uint32_t magic;
    uint32_t slot;
};
static_assert(sizeof(SlotRelease) == 8, "SlotRelease layout is wire format; see cuda_ipc.py");

} // namespace ipc
} // namespace sensing
} // namespace plugins
