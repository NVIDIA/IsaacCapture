/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_SENDER_SESSION_H
#define ARGUS_SENDER_SENDER_SESSION_H

namespace argus_sender
{

struct AppConfig;

bool executeConfiguredSender(const AppConfig& config);

} // namespace argus_sender

#endif // ARGUS_SENDER_SENDER_SESSION_H
