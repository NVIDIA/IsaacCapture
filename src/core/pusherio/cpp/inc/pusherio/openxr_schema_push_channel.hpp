// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_push_channel.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_push_tensor.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace core
{

/*!
 * @brief OpenXR implementation of one schema-push channel.
 *
 * The caller owns the parent OpenXR session and must keep it alive until this
 * channel has been destroyed.
 */
class OpenXRSchemaPushChannel final : public ISchemaPushChannel
{
public:
    /*!
     * @param handles OpenXR session handles (the session must enable get_required_extensions()).
     * @throws std::runtime_error if the tensor collection cannot be created.
     */
    OpenXRSchemaPushChannel(const OpenXRSessionHandles& handles, SchemaPusherConfig config);
    ~OpenXRSchemaPushChannel() override = default;

    OpenXRSchemaPushChannel(const OpenXRSchemaPushChannel&) = delete;
    OpenXRSchemaPushChannel& operator=(const OpenXRSchemaPushChannel&) = delete;
    OpenXRSchemaPushChannel(OpenXRSchemaPushChannel&&) = delete;
    OpenXRSchemaPushChannel& operator=(OpenXRSchemaPushChannel&&) = delete;

    //! Required OpenXR extensions, including the platform time-conversion extension.
    static std::vector<std::string> get_required_extensions();

    const SchemaPusherConfig& config() const override;

    /*!
     * Pads the sample to max_flatbuffer_size to satisfy the fixed-size tensor, and
     * converts the local common clock to XrTime; the reader side (SchemaTracker)
     * converts it back so DeviceDataTimestamp carries monotonic nanoseconds.
     *
     * @throws std::runtime_error if size exceeds max_flatbuffer_size, the raw device
     *         time is negative, or the push fails.
     * @throws std::invalid_argument if buffer is null for a non-empty sample.
     */
    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns) override;

private:
    using PushTensorPtr =
        std::unique_ptr<std::remove_pointer_t<XrPushTensorCollectionNV>, PFN_xrDestroyPushTensorCollectionNV>;

    PushTensorPtr create_tensor_collection(const OpenXRSessionHandles& handles);

    SchemaPusherConfig config_;
    XrTimeConverter time_converter_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.OpenXRSchemaPushChannel");
    PFN_xrPushTensorCollectionDataNV push_fn_{ nullptr };
    PushTensorPtr push_tensor_;
};

} // namespace core
