// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/ego_imu_tracker_base.hpp>
#include <schema/ego_limits.hpp>
#include <schema/orbbec_ego_imu_generated.h>

#include <cstddef>
#include <string>
#include <vector>

namespace core
{

class OrbbecEgoImuTracker : public ITracker
{
public:
    OrbbecEgoImuTracker(std::string collection_prefix,
                        std::vector<OrbbecEgoImuSensor> sensors = { OrbbecEgoImuSensor_Accel, OrbbecEgoImuSensor_Gyro },
                        size_t max_flatbuffer_size = EGO_MAX_FLATBUFFER_SIZE);
    std::string_view get_name() const override
    {
        return "OrbbecEgoImuTracker";
    }
    const Serialized<OrbbecEgoImuBatch>& get_stream_data(const ITrackerSession& session, size_t stream_index) const;
    const std::string& collection_prefix() const
    {
        return collection_prefix_;
    }
    const std::vector<OrbbecEgoImuSensor>& sensors() const
    {
        return sensors_;
    }
    const std::vector<std::string>& stream_names() const
    {
        return stream_names_;
    }
    size_t max_flatbuffer_size() const
    {
        return max_flatbuffer_size_;
    }
    size_t get_stream_count() const
    {
        return sensors_.size();
    }

private:
    std::string collection_prefix_;
    std::vector<OrbbecEgoImuSensor> sensors_;
    std::vector<std::string> stream_names_;
    size_t max_flatbuffer_size_;
};

} // namespace core
