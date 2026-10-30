// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_pusher.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_push_tensor.h>
#include <memory>
#include <string>
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
    OpenXRSchemaPushChannel(const OpenXRSessionHandles& handles, SchemaPusherConfig config);
    ~OpenXRSchemaPushChannel() override;

    OpenXRSchemaPushChannel(const OpenXRSchemaPushChannel&) = delete;
    OpenXRSchemaPushChannel& operator=(const OpenXRSchemaPushChannel&) = delete;
    OpenXRSchemaPushChannel(OpenXRSchemaPushChannel&&) = delete;
    OpenXRSchemaPushChannel& operator=(OpenXRSchemaPushChannel&&) = delete;

    static std::vector<std::string> get_required_extensions();

    const SchemaPusherConfig& config() const override;
    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns) override;

private:
    void create_tensor_collection(const OpenXRSessionHandles& handles, PFN_xrCreatePushTensorCollectionNV create_fn);

    SchemaPusherConfig config_;
    XrTimeConverter time_converter_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.OpenXRSchemaPushChannel");
    XrPushTensorCollectionNV push_tensor_{ XR_NULL_HANDLE };
    PFN_xrPushTensorCollectionDataNV push_fn_{ nullptr };
    PFN_xrDestroyPushTensorCollectionNV destroy_fn_{ nullptr };
};

} // namespace core
