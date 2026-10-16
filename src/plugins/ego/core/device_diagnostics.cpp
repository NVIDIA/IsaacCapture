// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "device_diagnostics.hpp"

#include <schema/ego_device_state_utils.hpp>

#include <utility>

namespace plugins::ego
{
namespace
{
template <typename Payload>
core::EgoDeviceStateT event(const DiagnosticOrigin& origin, Payload payload)
{
    core::EgoDeviceStateT value;
    value.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    value.observed_at_ns = origin.observed_at_ns;
    value.device_uid = origin.uid;
    value.capture_epoch = origin.epoch;
    value.event.Set(std::move(payload));
    return value;
}
} // namespace

core::EgoDeviceStateT sdk_report(const DiagnosticOrigin& origin,
                                 uint64_t sequence,
                                 int32_t work_mode,
                                 uint64_t status_flags,
                                 uint64_t error_flags,
                                 uint64_t storage_free_bytes)
{
    core::EgoSdkReportT payload;
    payload.sequence_number = sequence;
    payload.work_mode = work_mode;
    payload.status_flags = status_flags;
    payload.error_flags = error_flags;
    payload.storage_free_bytes = storage_free_bytes;
    return event(origin, std::move(payload));
}

core::EgoDeviceStateT capture_summary(const DiagnosticOrigin& origin, core::EgoCaptureSummaryT summary)
{
    return event(origin, std::move(summary));
}

core::EgoDeviceStateT recovery_event(const DiagnosticOrigin& origin,
                                     core::EgoRecoveryStage stage,
                                     const std::string& reason,
                                     uint32_t reconnect_attempt)
{
    core::EgoRecoveryEventT payload;
    payload.stage = stage;
    payload.reason = reason;
    payload.reconnect_attempt = reconnect_attempt;
    return event(origin, std::move(payload));
}

core::EgoDeviceStateT device_snapshot(const DiagnosticOrigin& origin,
                                      std::optional<float> temperature,
                                      const std::vector<core::EgoDevicePropertyValue>& properties)
{
    core::EgoDeviceSnapshotT payload;
    if (temperature && std::isfinite(*temperature))
        payload.imu_temperature_c = *temperature;
    payload.properties = properties;
    return event(origin, std::move(payload));
}
} // namespace plugins::ego
