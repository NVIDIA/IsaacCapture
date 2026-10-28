// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_device_state_generated.h>

#include <cmath>
#include <stdexcept>

namespace core
{
inline constexpr uint32_t EGO_DIAGNOSTIC_VERSION = 2;

inline void validate_ego_diagnostic(const OrbbecEgoDeviceState& value)
{
    if (value.diagnostic_version() != EGO_DIAGNOSTIC_VERSION || value.observed_at_ns() <= 0 || !value.device_uid() ||
        value.device_uid()->size() == 0 || !value.event())
        throw std::invalid_argument("EGO diagnostics require version 2, a device UID, observation time, and an event");
    switch (value.event_type())
    {
    case OrbbecEgoDiagnosticEvent_OrbbecEgoSdkReport:
        break;
    case OrbbecEgoDiagnosticEvent_OrbbecEgoCaptureSummary:
        if (const auto health = value.event_as_OrbbecEgoCaptureSummary()->capture_health();
            health < OrbbecEgoCaptureHealth_MIN || health > OrbbecEgoCaptureHealth_MAX)
            throw std::invalid_argument("Invalid EGO capture health");
        break;
    case OrbbecEgoDiagnosticEvent_OrbbecEgoRecoveryEvent:
        if (const auto stage = value.event_as_OrbbecEgoRecoveryEvent()->stage();
            stage < OrbbecEgoRecoveryStage_MIN || stage > OrbbecEgoRecoveryStage_MAX)
            throw std::invalid_argument("Invalid EGO recovery stage");
        break;
    case OrbbecEgoDiagnosticEvent_OrbbecEgoDeviceSnapshot:
        if (const auto temperature = value.event_as_OrbbecEgoDeviceSnapshot()->imu_temperature_c();
            temperature && !std::isfinite(*temperature))
            throw std::invalid_argument("EGO IMU temperature must be finite or absent");
        break;
    default:
        throw std::invalid_argument("Unknown EGO diagnostic event type");
    }
}
} // namespace core
