// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_keyboard_tracker_impl.hpp"

#include <deviceio_trackers/keyboard_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <schema/keyboard_bfbs_generated.h>
#include <schema/serialized.hpp>

#include <utility>
#include <vector>

namespace core
{

ReplayKeyboardTrackerImpl::ReplayKeyboardTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                     std::string_view base_name,
                                                     const RecordedSchemas& recorded,
                                                     std::shared_ptr<KeyboardInputState> state)
    : mcap_viewers_(std::make_unique<KeyboardMcapViewers>(
          std::move(reader),
          base_name,
          std::vector<std::string>(
              KeyboardRecordingTraits::replay_channels.begin(), KeyboardRecordingTraits::replay_channels.end()),
          recorded)),
      state_(std::move(state)),
      base_name_(base_name),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayKeyboardTrackerImpl"))
{
}

const Serialized<KeyboardOutput>& ReplayKeyboardTrackerImpl::get_data() const
{
    return tracked_;
}

void ReplayKeyboardTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    if (!state_->drain().events.empty() && !warned_live_input_)
    {
        logger_->warn("[{}] ignoring live keyboard provider input during replay", base_name_);
        warned_live_input_ = true;
    }

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
