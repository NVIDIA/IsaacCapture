// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/full_body_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/full_body_generated.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace core
{

using FullBodyMcapViewers = McapTrackerViewers<FullBodyPoseRecord>;

// Vendor-neutral replay impl: reads the recorded full-body channel regardless of
// which live vendor produced it (all vendors record core.FullBodyPoseRecord).
class ReplayFullBodyTrackerImpl : public IFullBodyTrackerImpl
{
public:
    ReplayFullBodyTrackerImpl(TrackerSampleSource source, std::string_view base_name);

    ReplayFullBodyTrackerImpl(const ReplayFullBodyTrackerImpl&) = delete;
    ReplayFullBodyTrackerImpl& operator=(const ReplayFullBodyTrackerImpl&) = delete;
    ReplayFullBodyTrackerImpl(ReplayFullBodyTrackerImpl&&) = delete;
    ReplayFullBodyTrackerImpl& operator=(ReplayFullBodyTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<FullBodyPose>& get_body_pose() const override;

private:
    Serialized<FullBodyPose> tracked_;
    std::unique_ptr<FullBodyMcapViewers> mcap_viewers_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_data_ = false;
};

// Deprecated alias for the renamed ReplayFullBodyTrackerImpl (was
// ReplayFullBodyTrackerPicoImpl before the vendor-neutral rename). Retained so source
// referencing the old type name keeps compiling (with a deprecation warning); prefer
// ReplayFullBodyTrackerImpl.
using ReplayFullBodyTrackerPicoImpl [[deprecated("renamed to core::ReplayFullBodyTrackerImpl")]] =
    ReplayFullBodyTrackerImpl;

} // namespace core
