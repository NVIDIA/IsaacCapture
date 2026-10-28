// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_device_state_generated.h>

#include <optional>
#include <string>
#include <vector>

namespace plugins::ego
{
struct DiagnosticOrigin
{
    std::string uid;
    uint32_t epoch;
    int64_t observed_at_ns;
};
core::OrbbecEgoDeviceStateT sdk_report(const DiagnosticOrigin&,
                                       uint64_t sequence,
                                       int32_t work_mode,
                                       uint64_t status_flags,
                                       uint64_t error_flags,
                                       uint64_t storage_free_bytes);
core::OrbbecEgoDeviceStateT capture_summary(const DiagnosticOrigin&, core::OrbbecEgoCaptureSummaryT);
core::OrbbecEgoDeviceStateT recovery_event(const DiagnosticOrigin&,
                                           core::OrbbecEgoRecoveryStage,
                                           const std::string& reason,
                                           uint32_t reconnect_attempt);
core::OrbbecEgoDeviceStateT device_snapshot(const DiagnosticOrigin&,
                                            std::optional<float> temperature,
                                            const std::vector<core::OrbbecEgoDevicePropertyValue>& properties);
} // namespace plugins::ego
