// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/control_handler.hpp"

#include <arpa/inet.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>

namespace plugins
{
namespace sensing
{

namespace
{

using Json = nlohmann::json;

Json error_object(const std::string& code, const std::string& message)
{
    return Json{ { "code", code }, { "message", message } };
}

std::string failure(const Json& id, const std::string& code, const std::string& message)
{
    return Json{ { "id", id }, { "ok", false }, { "error", error_object(code, message) } }.dump();
}

/// An IP literal or a plain host name; anything else (spaces, slashes, a URL) is refused with a reason.
bool valid_host(const std::string& host)
{
    if (host.empty() || host.size() > 253)
        return false;
    unsigned char buffer[sizeof(in6_addr)];
    if (::inet_pton(AF_INET, host.c_str(), buffer) == 1 || ::inet_pton(AF_INET6, host.c_str(), buffer) == 1)
        return true;
    return std::all_of(host.begin(), host.end(), [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '-'; }) &&
           host.front() != '-' && host.front() != '.';
}

} // namespace

std::string status_json(const StatusSnapshot& snapshot)
{
    Json streams = Json::array();
    for (const auto& stream : snapshot.streams)
    {
        streams.push_back(Json{
            { "stream", stream.track },
            { "sensor", stream.sensor_id },
            { "direct", Json{ { "host", stream.direct_host },
                              { "port", stream.direct_port },
                              { "connected", stream.direct_connected } } },
            { "frames", stream.frames },
            { "published", stream.published },
        });
    }
    Json out{
        { "state", to_string(snapshot.state) },
        { "release", snapshot.release },
        { "uptime_seconds", snapshot.uptime_seconds },
        { "streams", streams },
    };
    if (snapshot.fault)
        out["fault"] = error_object(snapshot.fault->code, snapshot.fault->message);
    return out.dump();
}

std::string state_event(const StatusSnapshot& snapshot)
{
    return Json{
        { "type", "event" }, { "event", "state_changed" }, { "status", Json::parse(status_json(snapshot)) }
    }.dump();
}

std::string error_event(const Fault& fault)
{
    return Json{ { "type", "event" }, { "event", "error" }, { "error", error_object(fault.code, fault.message) } }.dump();
}

ControlHandler::ControlHandler(const ProducerStatus& status, ControlActions& actions)
    : m_status(status), m_actions(actions)
{
}

std::string ControlHandler::handle(const std::string& text) const
{
    Json request = Json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (request.is_discarded() || !request.is_object())
        return failure(nullptr, "bad_json", "the message must be one JSON object");

    const Json id = request.contains("id") ? request["id"] : Json(nullptr);
    if (!request.contains("type") || !request["type"].is_string())
        return failure(id, "missing_type", "the request needs a string \"type\"");
    const std::string type = request["type"];

    if (type == "ping")
        return Json{ { "id", id }, { "ok", true }, { "type", "pong" } }.dump();

    if (type == "get_status")
    {
        return Json{
            { "id", id }, { "ok", true }, { "type", "status" }, { "status", Json::parse(status_json(m_status.snapshot())) }
        }.dump();
    }

    if (type == "set_direct_target")
    {
        if (!request.contains("stream") || !request["stream"].is_string())
            return failure(id, "missing_stream", "set_direct_target needs a string \"stream\" (a track name)");
        if (!request.contains("host") || !request["host"].is_string())
            return failure(id, "missing_host", "set_direct_target needs a string \"host\" (\"\" turns the link off)");
        const std::string stream = request["stream"];
        const std::string host = request["host"];

        std::uint16_t port = 0;
        if (!host.empty())
        {
            if (!valid_host(host))
                return failure(id, "invalid_host", "\"host\" must be an IP address or a host name, not \"" + host + "\"");
            if (!request.contains("port") || !request["port"].is_number_integer() || request["port"] < 1 ||
                request["port"] > 65535)
                return failure(id, "invalid_port", "\"port\" must be an integer from 1 to 65535");
            port = request["port"].get<std::uint16_t>();
        }

        const auto known = m_status.snapshot().streams;
        if (std::none_of(known.begin(), known.end(), [&](const StreamStatus& s) { return s.track == stream; }))
        {
            std::string names;
            for (const auto& s : known)
                names += (names.empty() ? "" : ", ") + s.track;
            return failure(id, "unknown_stream", "no stream named \"" + stream + "\"; the streams are: " + names);
        }

        const ActionResult result = m_actions.set_direct_target(stream, host, port);
        if (!result.ok)
            return failure(id, result.code, result.message);
        return Json{
            { "id", id }, { "ok", true }, { "type", "status" }, { "status", Json::parse(status_json(m_status.snapshot())) }
        }.dump();
    }

    return failure(
        id, "unknown_type", "unknown request type \"" + type + "\"; supported: ping, get_status, set_direct_target");
}

} // namespace sensing
} // namespace plugins
