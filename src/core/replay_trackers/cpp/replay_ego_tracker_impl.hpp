// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/ego_frame_metadata_tracker_base.hpp>
#include <deviceio_base/ego_imu_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/orbbec_ego_camera_generated.h>
#include <schema/orbbec_ego_imu_generated.h>

#include <memory>
#include <string_view>
#include <vector>

namespace core
{

class ReplayEgoFrameMetadataTrackerImpl : public IEgoFrameMetadataTrackerImpl
{
public:
    ReplayEgoFrameMetadataTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                      std::string_view base_name,
                                      const std::vector<std::string>& channels,
                                      const RecordedSchemas& recorded);
    void update(int64_t monotonic_time_ns) override;
    const Serialized<OrbbecEgoFrameMetadata>& get_stream_data(size_t stream_index) const override;

private:
    std::unique_ptr<McapTrackerViewers<OrbbecEgoFrameMetadataRecord>> viewers_;
    std::vector<Serialized<OrbbecEgoFrameMetadata>> streams_;
    std::shared_ptr<spdlog::logger> logger_;
    std::vector<bool> warned_no_data_;
};

class ReplayEgoImuTrackerImpl : public IEgoImuTrackerImpl
{
public:
    ReplayEgoImuTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                            std::string_view base_name,
                            const std::vector<std::string>& channels,
                            const RecordedSchemas& recorded);
    void update(int64_t monotonic_time_ns) override;
    const Serialized<OrbbecEgoImuBatch>& get_stream_data(size_t stream_index) const override;

private:
    std::unique_ptr<McapTrackerViewers<OrbbecEgoImuBatchRecord>> viewers_;
    std::vector<Serialized<OrbbecEgoImuBatch>> streams_;
    std::shared_ptr<spdlog::logger> logger_;
    std::vector<bool> warned_no_data_;
};

} // namespace core
