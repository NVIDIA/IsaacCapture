// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/ego_device_state_generated.h>

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
core::EgoDeviceStateT sdk_report(const DiagnosticOrigin&,
                                 uint64_t sequence,
                                 int32_t work_mode,
                                 uint64_t status_flags,
                                 uint64_t error_flags,
                                 uint64_t storage_free_bytes);
core::EgoDeviceStateT capture_summary(const DiagnosticOrigin&, core::EgoCaptureSummaryT);
core::EgoDeviceStateT recovery_event(const DiagnosticOrigin&,
                                     core::EgoRecoveryStage,
                                     const std::string& reason,
                                     uint32_t reconnect_attempt);
core::EgoDeviceStateT device_snapshot(const DiagnosticOrigin&,
                                      std::optional<float> temperature,
                                      const std::vector<core::EgoDevicePropertyValue>& properties);
} // namespace plugins::ego
