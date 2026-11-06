// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "producer_status.hpp"

#include <string_view>

namespace plugins
{
namespace sensing
{

/// A failure explained for an operator, and the state the producer is in because of it.
struct Diagnosis
{
    Fault fault;
    ProducerState state = ProducerState::Error;
};

/**
 * @brief Turn the text of a failure into what an operator should check.
 *
 * The producer's own components (the SIPL and Argus capture backends, the Jetson encoder, the LiveKit session)
 * report failures as exception text, and SIPL as bare status codes ("Init failed with SIPLStatus 10"). A camera that
 * is unpowered or unplugged produces the same failure as several other conditions, so the message names the physical
 * checks first and keeps the original text for anyone who needs it. Camera failures put the producer in
 * CameraFault; encoder, publishing and other failures in Error. Text nothing matches still gets a code and the
 * original text, never silence.
 */
Diagnosis diagnose_failure(std::string_view what);

/// The fault part of diagnose_failure().
Fault diagnose_camera_failure(std::string_view what);

} // namespace sensing
} // namespace plugins
