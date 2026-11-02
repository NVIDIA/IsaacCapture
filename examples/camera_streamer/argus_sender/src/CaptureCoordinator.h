/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_COORDINATOR_H
#define ARGUS_SENDER_CAPTURE_COORDINATOR_H

#include <cstdint>

namespace argus_sender
{

struct AppConfig;

bool runConfiguredCapture(const AppConfig& config, bool* receiverDisconnected, bool appendDump);
bool printCameraModes();

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_COORDINATOR_H
