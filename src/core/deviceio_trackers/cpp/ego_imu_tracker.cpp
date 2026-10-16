// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/ego_imu_tracker.hpp"

#include <set>
#include <stdexcept>
#include <utility>

namespace core
{

namespace
{
void validate_common(const std::string& tracker, const std::string& prefix, size_t max_size)
{
    if (prefix.empty())
        throw std::invalid_argument(tracker + ": collection_prefix is required");
    if (max_size == 0)
        throw std::invalid_argument(tracker + ": max_flatbuffer_size must be positive");
}
} // namespace

EgoImuTracker::EgoImuTracker(std::string collection_prefix, std::vector<EgoImuSensor> sensors, size_t max_flatbuffer_size)
    : collection_prefix_(std::move(collection_prefix)),
      sensors_(std::move(sensors)),
      max_flatbuffer_size_(max_flatbuffer_size)
{
    validate_common("EgoImuTracker", collection_prefix_, max_flatbuffer_size_);
    if (sensors_.empty())
        throw std::invalid_argument("EgoImuTracker: at least one sensor is required");
    std::set<EgoImuSensor> unique;
    for (const auto sensor : sensors_)
    {
        const char* name = EnumNameEgoImuSensor(sensor);
        if (name == nullptr || *name == '\0')
            throw std::invalid_argument("EgoImuTracker: invalid sensor");
        if (!unique.insert(sensor).second)
            throw std::invalid_argument("EgoImuTracker: duplicate sensor " + std::string(name));
        stream_names_.emplace_back(name);
    }
}

const Serialized<EgoImuBatch>& EgoImuTracker::get_stream_data(const ITrackerSession& session, size_t stream_index) const
{
    return static_cast<const IEgoImuTrackerImpl&>(session.get_tracker_impl(*this)).get_stream_data(stream_index);
}

} // namespace core
