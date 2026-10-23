// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_pusher.hpp"

#include <oxr_utils/oxr_session_handles.hpp>

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
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<ISchemaPushChannel> make_openxr_schema_push_channel(const OpenXRSessionHandles& handles,
                                                                    SchemaPusherConfig config);

} // namespace core
