// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/ego_device_state_generated.h>

#include <cmath>
#include <stdexcept>

namespace core
{
inline constexpr uint32_t EGO_DIAGNOSTIC_VERSION = 2;

inline void validate_ego_diagnostic(const EgoDeviceState& value)
{
    if (value.diagnostic_version() != EGO_DIAGNOSTIC_VERSION || value.observed_at_ns() <= 0 || !value.device_uid() ||
        value.device_uid()->size() == 0 || !value.event())
        throw std::invalid_argument("EGO diagnostics require version 2, a device UID, observation time, and an event");
    switch (value.event_type())
    {
    case EgoDiagnosticEvent_EgoSdkReport:
        break;
    case EgoDiagnosticEvent_EgoCaptureSummary:
        if (const auto health = value.event_as_EgoCaptureSummary()->capture_health();
            health < EgoCaptureHealth_MIN || health > EgoCaptureHealth_MAX)
            throw std::invalid_argument("Invalid EGO capture health");
        break;
    case EgoDiagnosticEvent_EgoRecoveryEvent:
        if (const auto stage = value.event_as_EgoRecoveryEvent()->stage();
            stage < EgoRecoveryStage_MIN || stage > EgoRecoveryStage_MAX)
            throw std::invalid_argument("Invalid EGO recovery stage");
        break;
    case EgoDiagnosticEvent_EgoDeviceSnapshot:
        if (const auto temperature = value.event_as_EgoDeviceSnapshot()->imu_temperature_c();
            temperature && !std::isfinite(*temperature))
            throw std::invalid_argument("EGO IMU temperature must be finite or absent");
        break;
    default:
        throw std::invalid_argument("Unknown EGO diagnostic event type");
    }
}
} // namespace core
