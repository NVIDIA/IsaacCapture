// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/haptic_command_reader_tracker.hpp"

#include <stdexcept>

namespace core
{

HapticCommandReaderTracker::HapticCommandReaderTracker(const std::string& collection_id,
                                                       std::size_t max_payload_size,
                                                       std::optional<std::chrono::nanoseconds> max_sample_age)
    : collection_id_(collection_id), max_payload_size_(max_payload_size), max_sample_age_(max_sample_age)
{
    if (collection_id_.empty())
    {
        throw std::invalid_argument("HapticCommandReaderTracker: collection_id must be non-empty");
    }
    if (max_payload_size_ == 0)
    {
        throw std::invalid_argument("HapticCommandReaderTracker: max_payload_size must be > 0");
    }
    if (max_sample_age_ && *max_sample_age_ <= std::chrono::nanoseconds::zero())
    {
        throw std::invalid_argument("HapticCommandReaderTracker: max_sample_age must be > 0");
    }
}

const Serialized<HapticCommand>& HapticCommandReaderTracker::get_data(const ITrackerSession& session) const
{
    return static_cast<const IHapticCommandReaderTrackerImpl&>(session.get_tracker_impl(*this)).get_data();
}

const Serialized<HapticCommand>& HapticCommandReaderTracker::get_data(const ITrackerSession& session,
                                                                      std::string_view endpoint) const
{
    return static_cast<const IHapticCommandReaderTrackerImpl&>(session.get_tracker_impl(*this)).get_data(endpoint);
}

} // namespace core
