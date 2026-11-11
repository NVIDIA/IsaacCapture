/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_SIPL_SENDER_H
#define ARGUS_SENDER_SIPL_SENDER_H

namespace argus_sender
{

struct AppConfig;

// The AVS1 sender fed by the SIPL backend: independent mono HEVC streams, each frame encoded from the ISP0 surface
// and sent as its own bundle. Same return convention as runConfiguredCapture.
bool runSiplSender(const AppConfig& config, bool* receiverDisconnected, bool appendDump);

} // namespace argus_sender

#endif // ARGUS_SENDER_SIPL_SENDER_H
