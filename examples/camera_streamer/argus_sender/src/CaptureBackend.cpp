/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureBackend.h"

#include "CaptureFrame.h"

namespace argus_sender
{

const char* captureBackendName(CaptureBackendKind kind)
{
    switch (kind)
    {
    case CAPTURE_BACKEND_ARGUS:
        return "argus";
    case CAPTURE_BACKEND_SIPL:
        return "sipl";
    case CAPTURE_BACKEND_AUTO:
        break;
    }
    return "auto";
}

const char* sensorClockDomainName(SensorClockDomain domain)
{
    switch (domain)
    {
    case SENSOR_CLOCK_TSC_NS:
        return "tsc_ns";
    case SENSOR_CLOCK_SIPL_COUNTER:
        return "sipl_counter";
    case SENSOR_CLOCK_NONE:
        break;
    }
    return "none";
}

bool parseCaptureBackend(const std::string& text, CaptureBackendKind* kind)
{
    if (text == "auto")
        *kind = CAPTURE_BACKEND_AUTO;
    else if (text == "argus")
        *kind = CAPTURE_BACKEND_ARGUS;
    else if (text == "sipl")
        *kind = CAPTURE_BACKEND_SIPL;
    else
        return false;
    return true;
}

bool selectCaptureBackend(CaptureBackendKind requested,
                          const BackendAvailability& available,
                          CaptureBackendKind* selected,
                          std::string* error)
{
    if (requested == CAPTURE_BACKEND_ARGUS || requested == CAPTURE_BACKEND_SIPL)
    {
        const bool built = requested == CAPTURE_BACKEND_ARGUS ? available.argus : available.sipl;
        if (!built)
        {
            if (error)
                *error = std::string("capture backend '") + captureBackendName(requested) +
                         "' was requested but this build does not include it";
            return false;
        }
        *selected = requested;
        return true;
    }
    if (available.argus && available.sipl)
    {
        if (error)
            *error = "this build includes both the argus and sipl capture backends; set capture_backend to one of them";
        return false;
    }
    if (!available.argus && !available.sipl)
    {
        if (error)
            *error = "this build includes no capture backend";
        return false;
    }
    *selected = available.argus ? CAPTURE_BACKEND_ARGUS : CAPTURE_BACKEND_SIPL;
    return true;
}

} // namespace argus_sender
