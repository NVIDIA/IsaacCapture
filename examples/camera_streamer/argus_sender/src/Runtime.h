/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_RUNTIME_H
#define ARGUS_SENDER_RUNTIME_H

namespace argus_sender
{

void handleSignal(int signalNumber);
bool stopRequested();
void setVerboseLogging(bool enabled);
bool verboseLoggingEnabled();

} // namespace argus_sender

#endif // ARGUS_SENDER_RUNTIME_H
