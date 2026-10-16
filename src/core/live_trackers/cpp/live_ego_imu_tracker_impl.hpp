// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/live_trackers/schema_tracker.hpp"

#include <deviceio_trackers/ego_imu_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>

#include <memory>
#include <string_view>
#include <vector>

namespace core
{

using EgoImuMcapChannels = McapTrackerChannels<EgoImuBatchRecord>;

class LiveEgoImuTrackerImpl : public IEgoImuTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return SchemaTrackerBase::get_required_extensions();
    }
    static std::unique_ptr<EgoImuMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                    std::string_view base_name,
                                                                    const EgoImuTracker* tracker);
    LiveEgoImuTrackerImpl(const OpenXRSessionHandles& handles,
                          const EgoImuTracker* tracker,
                          std::unique_ptr<EgoImuMcapChannels> channels);
    void update(int64_t monotonic_time_ns) override;
    const Serialized<EgoImuBatch>& get_stream_data(size_t stream_index) const override;

private:
    struct StreamState
    {
        std::unique_ptr<SchemaTracker<EgoImuBatchRecord, EgoImuBatch>> reader;
        Serialized<EgoImuBatch> tracked;
    };
    std::unique_ptr<EgoImuMcapChannels> channels_;
    std::vector<StreamState> streams_;
};

} // namespace core
