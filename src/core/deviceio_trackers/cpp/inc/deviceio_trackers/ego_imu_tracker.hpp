// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/ego_imu_tracker_base.hpp>
#include <schema/ego_imu_generated.h>
#include <schema/ego_limits.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace core
{

class EgoImuTracker : public ITracker
{
public:
    EgoImuTracker(std::string collection_prefix,
                  std::vector<EgoImuSensor> sensors = { EgoImuSensor_Accel, EgoImuSensor_Gyro },
                  size_t max_flatbuffer_size = EGO_MAX_FLATBUFFER_SIZE);
    std::string_view get_name() const override
    {
        return "EgoImuTracker";
    }
    const Serialized<EgoImuBatch>& get_stream_data(const ITrackerSession& session, size_t stream_index) const;
    const std::string& collection_prefix() const
    {
        return collection_prefix_;
    }
    const std::vector<EgoImuSensor>& sensors() const
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
    std::vector<EgoImuSensor> sensors_;
    std::vector<std::string> stream_names_;
    size_t max_flatbuffer_size_;
};

} // namespace core
