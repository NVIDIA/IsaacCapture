// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_push_tensor.h>
#include <XR_NVX1_tensor_data.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Configuration for SchemaPusher.
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

    // buffer is borrowed for this call only; asynchronous transports must copy it before returning.
    virtual void push_buffer(const uint8_t* buffer,
                             size_t size,
                             int64_t sample_time_local_common_clock_ns,
                             int64_t sample_time_raw_device_clock_ns) = 0;
};

/*!
 * @brief Pushes FlatBuffer schema data through a transport-owned channel.
 *
 * This is the plugin-facing facade. It preserves the existing opaque FlatBuffer
 * contract while the owning session selects the channel implementation. Use
 * composition to add typed push methods where useful.
 *
 * Example usage with composition:
 * @code
 * class HeadPosePusher {
 * public:
 *     explicit HeadPosePusher(std::unique_ptr<ISchemaPushChannel> channel)
 *         : m_pusher(std::move(channel)) {}
 *
 *     void push(const HeadPoseT& data,
 *              int64_t sample_time_local_common_clock_ns,
 *              int64_t sample_time_raw_device_clock_ns) {
 *         flatbuffers::FlatBufferBuilder builder(m_pusher.config().max_flatbuffer_size);
 *         auto offset = HeadPose::Pack(builder, &data);
 *         builder.Finish(offset);
 *         m_pusher.push_buffer(builder.GetBufferPointer(), builder.GetSize(),
 *                              sample_time_local_common_clock_ns,
 *                              sample_time_raw_device_clock_ns);
 *     }
 *
 * private:
 *     SchemaPusher m_pusher;
 * };
 * @endcode
 */
class SchemaPusher
{
public:
    /*!
     * @brief Get required OpenXR extensions for pushing tensor data.
     *
     * Includes platform-specific time conversion extension.
     */
    static std::vector<std::string> get_required_extensions();

    /*!
     * @brief Constructs the pusher and initializes the OpenXR tensor collection.
     * @param handles OpenXR session handles (caller must create session with required extensions).
     * @param config Configuration for the tensor collection.
     * @throws std::runtime_error if initialization fails.
     */
    SchemaPusher(const OpenXRSessionHandles& handles, SchemaPusherConfig config);

    /*!
     * @brief Constructs a pusher from an already-created transport channel.
     *
     * The owning plugin session selects and creates the concrete channel before
     * constructing this transport-independent facade.
     */
    explicit SchemaPusher(std::unique_ptr<ISchemaPushChannel> channel);

    /*!
     * @brief Destroys the pusher and its transport-owned channel.
     */
    ~SchemaPusher();

    // Non-copyable, non-movable
    SchemaPusher(const SchemaPusher&) = delete;
    SchemaPusher& operator=(const SchemaPusher&) = delete;
    SchemaPusher(SchemaPusher&&) = delete;
    SchemaPusher& operator=(SchemaPusher&&) = delete;

    /*!
     * @brief Push raw serialized FlatBuffer data with timestamps.
     *
     * The channel receives the logical serialized size. The local OpenXR channel
     * pads it to max_flatbuffer_size to satisfy the fixed-size tensor contract.
     *
     * Both timestamp parameters must be in nanoseconds. The local common clock is
     * system monotonic time (CLOCK_MONOTONIC on Linux, QueryPerformanceCounter on
     * Windows) — values are comparable across all sources on the same machine.
     * The local OpenXR channel converts the local common clock value to XrTime
     * before storing; the reader side (SchemaTracker) converts it back so that
     * DeviceDataTimestamp carries monotonic nanoseconds in both local-common-clock
     * fields.
     *
     * If the raw device clock is not available, pass the local common clock value
     * as a best-effort substitute.
     *
     * @param buffer Pointer to serialized FlatBuffer data.
     * @param size Size of the serialized data in bytes.
     * @param sample_time_local_common_clock_ns Sample time in system monotonic nanoseconds.
     * @param sample_time_raw_device_clock_ns Sample time in the source device's own clock (nanoseconds).
     * @throws std::runtime_error if the push fails.
     */
    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns);

    /*!
     * @brief Access the configuration.
     */
    const SchemaPusherConfig& config() const;

private:
    std::unique_ptr<ISchemaPushChannel> m_channel;
};

} // namespace core
