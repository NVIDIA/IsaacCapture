// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "producer_status.hpp"

#include <cstdint>
#include <string>

namespace plugins
{
namespace sensing
{

/// The outcome of an action the channel asked the producer to take.
struct ActionResult
{
    bool ok = true;
    std::string code;
    std::string message;
};

/// What a management client can change. The producer implements it; the handler only validates and forwards.
class ControlActions
{
public:
    virtual ~ControlActions() = default;

    /// Send stream `track` to host:port from now on, or stop the direct link when `host` is empty.
    virtual ActionResult set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) = 0;
};

/**
 * @brief The management protocol: one JSON object per WebSocket text message.
 *
 *   request   {"id": 7, "type": "get_status"}
 *             {"id": 8, "type": "set_direct_target", "stream": "sensing-left", "host": "192.168.1.205", "port": 5600}
 *             {"id": 9, "type": "ping"}
 *   reply     {"id": 7, "ok": true, "type": "status", "status": {...}}
 *             {"id": 8, "ok": false, "error": {"code": "invalid_host", "message": "..."}}
 *   events    {"type": "event", "event": "state_changed", "status": {...}}
 *             {"type": "event", "event": "error", "error": {"code": "...", "message": "..."}}
 *
 * Every failure is answered with a code and a sentence; nothing is dropped silently.
 */
class ControlHandler
{
public:
    ControlHandler(const ProducerStatus& status, ControlActions& actions);

    /// The reply to one received text message.
    std::string handle(const std::string& text) const;

private:
    const ProducerStatus& m_status;
    ControlActions& m_actions;
};

/// The status as a JSON object text.
std::string status_json(const StatusSnapshot& snapshot);

/// A pushed notification that the state changed.
std::string state_event(const StatusSnapshot& snapshot);

/// A pushed notification of a failure.
std::string error_event(const Fault& fault);

} // namespace sensing
} // namespace plugins
