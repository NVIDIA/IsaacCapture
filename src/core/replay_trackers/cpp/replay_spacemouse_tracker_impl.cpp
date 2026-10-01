// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_spacemouse_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/serialized.hpp>
#include <schema/spacemouse_bfbs_generated.h>

#include <utility>
#include <vector>

namespace core
{

ReplaySpaceMouseTrackerImpl::ReplaySpaceMouseTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                         std::string_view base_name,
                                                         const RecordedSchemas& recorded)
    : mcap_viewers_(std::make_unique<SpaceMouseMcapViewers>(
          std::move(reader),
          base_name,
          std::vector<std::string>(
              SpaceMouseRecordingTraits::replay_channels.begin(), SpaceMouseRecordingTraits::replay_channels.end()),
          recorded)),
      base_name_(base_name),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplaySpaceMouseTrackerImpl"))
{
}

const Serialized<SpaceMouseOutput>& ReplaySpaceMouseTrackerImpl::get_data() const
{
    return tracked_;
}

void ReplaySpaceMouseTrackerImpl::update(int64_t /*monotonic_time_ns*/)
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
            logger_->warn("[{}] no data (EOF or gap)", base_name_);
            warned_no_data_ = true;
        }
        tracked_.reset();
    }
}

} // namespace core
