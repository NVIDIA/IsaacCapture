// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/spacemouse_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/spacemouse_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace core
{

using SpaceMouseMcapViewers = McapTrackerViewers<SpaceMouseOutputRecord>;

// Replays recorded spacemouse state; the physical device is never opened.
class ReplaySpaceMouseTrackerImpl : public ISpaceMouseTrackerImpl
{
public:
    ReplaySpaceMouseTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                std::string_view base_name,
                                const RecordedSchemas& recorded);

    ReplaySpaceMouseTrackerImpl(const ReplaySpaceMouseTrackerImpl&) = delete;
    ReplaySpaceMouseTrackerImpl& operator=(const ReplaySpaceMouseTrackerImpl&) = delete;
    ReplaySpaceMouseTrackerImpl(ReplaySpaceMouseTrackerImpl&&) = delete;
    ReplaySpaceMouseTrackerImpl& operator=(ReplaySpaceMouseTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<SpaceMouseOutput>& get_data() const override;

private:
    Serialized<SpaceMouseOutput> tracked_;
    std::unique_ptr<SpaceMouseMcapViewers> mcap_viewers_;
    std::string base_name_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_data_ = false;
};

} // namespace core
