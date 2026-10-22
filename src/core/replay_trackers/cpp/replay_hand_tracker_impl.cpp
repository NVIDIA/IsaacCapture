// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_hand_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/hand_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <cassert>
#include <cstring>

namespace core
{

// ============================================================================
// ReplayHandTrackerImpl
// ============================================================================

ReplayHandTrackerImpl::ReplayHandTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                             std::string_view base_name,
                                             const RecordedSchemas& recorded)
    : mcap_viewers_(
          std::make_unique<HandMcapViewers>(std::move(reader),
                                            base_name,
                                            std::vector<std::string>(HandRecordingTraits::replay_channels.begin(),
                                                                     HandRecordingTraits::replay_channels.end()),
                                            recorded)),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayHandTrackerImpl"))
{
}

const Serialized<HandPose>& ReplayHandTrackerImpl::get_left_hand() const
{
    return left_tracked_;
}

const Serialized<HandPose>& ReplayHandTrackerImpl::get_right_hand() const
{
    return right_tracked_;
}

void ReplayHandTrackerImpl::update(int64_t /*monotonic_time_ns*/)
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
            logger_->warn("left hand data not found");
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
            logger_->warn("right hand data not found");
            warned_no_right_data_ = true;
        }
        right_tracked_.reset();
    }
}

} // namespace core
