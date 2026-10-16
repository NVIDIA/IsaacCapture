// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_ego_tracker_impl.hpp"

#include <schema/ego_camera_bfbs_generated.h>
#include <schema/ego_imu_bfbs_generated.h>

#include <stdexcept>
#include <utility>

namespace core
{

ReplayEgoFrameMetadataTrackerImpl::ReplayEgoFrameMetadataTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                                     std::string_view base_name,
                                                                     const std::vector<std::string>& channels,
                                                                     const RecordedSchemas& recorded)
    : viewers_(std::make_unique<McapTrackerViewers<EgoFrameMetadataRecord>>(
          std::move(reader), base_name, channels, recorded)),
      streams_(channels.size()),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayEgoFrameMetadataTrackerImpl")),
      warned_no_data_(channels.size(), false)
{
}

void ReplayEgoFrameMetadataTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (size_t index = 0; index < streams_.size(); ++index)
    {
        auto record = viewers_->read(index);
        streams_[index] = record ? record.narrow(record->data()) : Serialized<EgoFrameMetadata>{};
        if (record)
        {
            warned_no_data_[index] = false;
        }
        else if (!warned_no_data_[index])
        {
            logger_->warn("no data (EOF or gap), stream_index={}", index);
            warned_no_data_[index] = true;
        }
    }
}

const Serialized<EgoFrameMetadata>& ReplayEgoFrameMetadataTrackerImpl::get_stream_data(size_t index) const
{
    if (index >= streams_.size())
        throw std::out_of_range("EgoFrameMetadataTracker: invalid stream index");
    return streams_[index];
}

ReplayEgoImuTrackerImpl::ReplayEgoImuTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                 std::string_view base_name,
                                                 const std::vector<std::string>& channels,
                                                 const RecordedSchemas& recorded)
    : viewers_(std::make_unique<McapTrackerViewers<EgoImuBatchRecord>>(std::move(reader), base_name, channels, recorded)),
      streams_(channels.size()),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayEgoImuTrackerImpl")),
      warned_no_data_(channels.size(), false)
{
}

void ReplayEgoImuTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (size_t index = 0; index < streams_.size(); ++index)
    {
        auto record = viewers_->read(index);
        streams_[index] = record ? record.narrow(record->data()) : Serialized<EgoImuBatch>{};
        if (record)
        {
            warned_no_data_[index] = false;
        }
        else if (!warned_no_data_[index])
        {
            logger_->warn("no data (EOF or gap), stream_index={}", index);
            warned_no_data_[index] = true;
        }
    }
}

const Serialized<EgoImuBatch>& ReplayEgoImuTrackerImpl::get_stream_data(size_t index) const
{
    if (index >= streams_.size())
        throw std::out_of_range("EgoImuTracker: invalid stream index");
    return streams_[index];
}

} // namespace core
