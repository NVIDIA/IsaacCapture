/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SenderSession.h"

#include "CaptureBackend.h"
#include "CaptureCoordinator.h"
#include "Config.h"
#include "Runtime.h"

#ifdef SENDER_WITH_SIPL
#    include "SiplSender.h"
#endif

#include <cstdio>
#include <string>

namespace argus_sender
{

bool executeConfiguredSender(const AppConfig& config)
{
    BackendAvailability available;
    available.argus = true;
#ifdef SENDER_WITH_SIPL
    available.sipl = true;
#endif
    CaptureBackendKind backend = CAPTURE_BACKEND_ARGUS;
    std::string backendError;
    if (!selectCaptureBackend(config.captureBackend, available, &backend, &backendError))
    {
        std::fprintf(stderr, "Configuration error: %s\n", backendError.c_str());
        return false;
    }
    std::printf("Capture backend: %s\n", captureBackendName(backend));
    // `auto` resolves to SIPL here, after validation, which checked the SIPL restrictions only for an explicit choice.
    if (backend == CAPTURE_BACKEND_SIPL && config.captureBackend != CAPTURE_BACKEND_SIPL)
    {
        std::string siplError;
        if (!validateSiplConfig(config, &siplError))
        {
            std::fprintf(stderr, "Configuration error: %s\n", siplError.c_str());
            return false;
        }
    }

    uint64_t reconnectSession = 0;
    while (!stopRequested())
    {
        bool disconnected = false;
        bool ran = false;
#ifdef SENDER_WITH_SIPL
        if (backend == CAPTURE_BACKEND_SIPL)
            ran = runSiplSender(config, &disconnected, reconnectSession > 0);
        else
#endif
            ran = runConfiguredCapture(config, &disconnected, reconnectSession > 0);
        if (!ran)
            return false;
        if (!disconnected || config.runtime.frames != 0)
            return true;
        ++reconnectSession;
        std::printf("AVS1 receiver disconnected; rebuilding the complete capture and processing pipeline\n");
    }
    return true;
}

} // namespace argus_sender
