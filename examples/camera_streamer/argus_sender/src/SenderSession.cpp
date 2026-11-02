/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SenderSession.h"

#include "CaptureCoordinator.h"
#include "Config.h"
#include "Runtime.h"

#include <cstdio>

namespace argus_sender
{

bool executeConfiguredSender(const AppConfig& config)
{
    uint64_t reconnectSession = 0;
    while (!stopRequested())
    {
        bool disconnected = false;
        if (!runConfiguredCapture(config, &disconnected, reconnectSession > 0))
        {
            return false;
        }
        if (!disconnected || config.runtime.frames != 0)
            return true;
        ++reconnectSession;
        std::printf("AVS1 receiver disconnected; rebuilding the complete capture and processing pipeline\n");
    }
    return true;
}

} // namespace argus_sender
