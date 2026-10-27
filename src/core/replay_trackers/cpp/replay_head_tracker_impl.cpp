// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_head_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/head_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <cassert>
#include <cstring>

namespace core
{

// ============================================================================
// ReplayHeadTrackerImpl
// ============================================================================

ReplayHeadTrackerImpl::ReplayHeadTrackerImpl(TrackerSampleSource source, std::string_view base_name)
    : mcap_viewers_(
          std::make_unique<HeadMcapViewers>(std::move(source),
                                            base_name,
                                            std::vector<std::string>(HeadRecordingTraits::replay_channels.begin(),
                                                                     HeadRecordingTraits::replay_channels.end()))),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayHeadTrackerImpl"))
{
}

const Serialized<HeadPose>& ReplayHeadTrackerImpl::get_head() const
{
    return tracked_;
}

void ReplayHeadTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    auto record = mcap_viewers_->read(0);
    if (record)
    {
        tracked_ = record.narrow(record->data());
        warned_no_data_ = false;
    }
    else
    {
        if (!warned_no_data_)
        {
            logger_->warn("head data not found");
            warned_no_data_ = true;
        }
        tracked_.reset();
    }
}

} // namespace core
