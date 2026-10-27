// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_controller_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/controller_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <cassert>
#include <cstring>

namespace core
{

// ============================================================================
// ReplayControllerTrackerImpl
// ============================================================================

ReplayControllerTrackerImpl::ReplayControllerTrackerImpl(TrackerSampleSource source, std::string_view base_name)
    : mcap_viewers_(std::make_unique<ControllerMcapViewers>(
          std::move(source),
          base_name,
          std::vector<std::string>(
              ControllerRecordingTraits::replay_channels.begin(), ControllerRecordingTraits::replay_channels.end()))),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayControllerTrackerImpl"))
{
}

const Serialized<ControllerSnapshot>& ReplayControllerTrackerImpl::get_left_controller() const
{
    return left_tracked_;
}

const Serialized<ControllerSnapshot>& ReplayControllerTrackerImpl::get_right_controller() const
{
    return right_tracked_;
}

void ReplayControllerTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    auto left_record = mcap_viewers_->read(0);
    auto right_record = mcap_viewers_->read(1);
    if (left_record)
    {
        left_tracked_ = left_record.narrow(left_record->data());
        warned_no_left_data_ = false;
    }
    else
    {
        if (!warned_no_left_data_)
        {
            logger_->warn("left controller data not found");
            warned_no_left_data_ = true;
        }
        left_tracked_.reset();
    }

    if (right_record)
    {
        right_tracked_ = right_record.narrow(right_record->data());
        warned_no_right_data_ = false;
    }
    else
    {
        if (!warned_no_right_data_)
        {
            logger_->warn("right controller data not found");
            warned_no_right_data_ = true;
        }
        right_tracked_.reset();
    }
}

} // namespace core
