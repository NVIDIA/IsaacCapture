/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime.h"

#include <atomic>
#include <signal.h>

namespace argus_sender
{
namespace
{

volatile sig_atomic_t g_stop = 0;
std::atomic<bool> g_verboseLogging(false);

} // namespace

void handleSignal(int)
{
    g_stop = 1;
}

bool stopRequested()
{
    return g_stop != 0;
}

void setVerboseLogging(bool enabled)
{
    g_verboseLogging.store(enabled, std::memory_order_relaxed);
}

bool verboseLoggingEnabled()
{
    return g_verboseLogging.load(std::memory_order_relaxed);
}

} // namespace argus_sender
