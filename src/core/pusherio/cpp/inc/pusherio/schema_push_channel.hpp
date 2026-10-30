// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace core
{

/*!
 * @brief Configuration for one schema push channel.
 *
 * This struct contains all parameters needed to set up a channel for pushing
 * FlatBuffer schema data. Local OpenXR sessions map it to a tensor collection.
 */
struct SchemaPusherConfig
{
    //! Tensor collection identifier for discovery (e.g., "head_data").
    //! Both pusher and reader must use the same collection_id to communicate.
    std::string collection_id;

    //! Maximum serialized FlatBuffer message size in bytes.
    //! The tensor collection is created with this fixed buffer size.
    //! Serialized messages larger than this will be rejected.
    size_t max_flatbuffer_size;

    //! Tensor name within the collection (e.g., "head_pose").
    //! This identifies the specific tensor holding the serialized data.
    std::string tensor_identifier;

    //! Human-readable description for debugging and runtime display.
    std::string localized_name;

    //! OpenXR application name. If empty, defaults to "Pusher" or "Reader".
    std::string app_name = "";
};

/*!
 * @brief Transport-owned channel for one configured schema stream.
 *
 * Implementations own any child resources needed to keep the channel valid.
 */
class ISchemaPushChannel
{
public:
    virtual ~ISchemaPushChannel() = default;

    virtual const SchemaPusherConfig& config() const = 0;

    /*!
     * @brief Push one serialized FlatBuffer sample.
     *
     * Both timestamps are in nanoseconds. The local common clock is system
     * monotonic time (CLOCK_MONOTONIC on Linux, QueryPerformanceCounter on
     * Windows), comparable across all sources on the same machine. If the raw
     * device clock is not available, pass the local common clock value as a
     * best-effort substitute.
     *
     * buffer is borrowed for this call only; asynchronous transports must copy it before returning.
     */
    virtual void push_buffer(const uint8_t* buffer,
                             size_t size,
                             int64_t sample_time_local_common_clock_ns,
                             int64_t sample_time_raw_device_clock_ns) = 0;
};

} // namespace core
