// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_full_body_tracker_jmex_impl.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

namespace
{

constexpr std::string_view COLLECTION_ID_PARAM = "collection_id";

SchemaTrackerConfig make_jmex_tensor_config(const TrackerVendor& vendor)
{
    if (vendor.id != LiveFullBodyTrackerJmexImpl::VENDOR_ID)
    {
        throw std::invalid_argument("j-mex full-body vendor id must be '" +
                                    std::string(LiveFullBodyTrackerJmexImpl::VENDOR_ID) + "'");
    }

    for (const auto& [key, value] : vendor.params)
    {
        (void)value;
        if (key != COLLECTION_ID_PARAM)
        {
            throw std::invalid_argument("j-mex full-body vendor does not support parameter '" + key + "'");
        }
    }

    SchemaTrackerConfig config;
    config.collection_id = std::string(LiveFullBodyTrackerJmexImpl::DEFAULT_COLLECTION_ID);
    config.max_flatbuffer_size = LiveFullBodyTrackerJmexImpl::MAX_FLATBUFFER_SIZE;
    config.tensor_identifier = std::string(LiveFullBodyTrackerJmexImpl::TENSOR_IDENTIFIER);
    config.localized_name = "j-mex Full Body";

    if (auto it = vendor.params.find(std::string(COLLECTION_ID_PARAM)); it != vendor.params.end())
    {
        if (it->second.empty())
        {
            throw std::invalid_argument("j-mex full-body collection_id must not be empty");
        }
        config.collection_id = it->second;
    }

    return config;
}

} // namespace

void LiveFullBodyTrackerJmexImpl::validate_vendor(const TrackerVendor& vendor)
{
    (void)make_jmex_tensor_config(vendor);
}

LiveFullBodyTrackerJmexImpl::LiveFullBodyTrackerJmexImpl(const OpenXRSessionHandles& handles,
                                                         const TrackerVendor& vendor,
                                                         std::unique_ptr<FullBodyMcapChannels> mcap_channels)
    : mcap_channels_(std::move(mcap_channels)),
      schema_reader_(handles, make_jmex_tensor_config(vendor), mcap_channels_.get(), /*mcap_channel_index=*/0)
{
}

void LiveFullBodyTrackerJmexImpl::update(int64_t /*monotonic_time_ns*/)
{
    schema_reader_.update(tracked_);
}

const Serialized<FullBodyPose>& LiveFullBodyTrackerJmexImpl::get_body_pose() const
{
    return tracked_;
}

} // namespace core
