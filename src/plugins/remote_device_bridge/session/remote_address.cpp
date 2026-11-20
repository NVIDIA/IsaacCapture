// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/remote_address.hpp"

#include "inc/remote_plugin_session/tcp_transport.hpp"
#include "inc/remote_plugin_session/websocket_transport.hpp"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace remote_plugin_session
{

namespace
{

//! HTTP Host value: IPv6 literals need brackets before the port.
std::string host_header(const Endpoint& endpoint)
{
    const bool v6 = endpoint.host.find(':') != std::string::npos;
    return (v6 ? "[" + endpoint.host + "]" : endpoint.host) + ":" + std::to_string(endpoint.port);
}

} // namespace

Endpoint parse_endpoint(const std::string& text, uint16_t default_port)
{
    Endpoint endpoint;
    endpoint.port = default_port;
    std::string port_text;
    bool has_port = false;
    if (!text.empty() && text.front() == '[')
    {
        // "[v6-address]" or "[v6-address]:port"
        const size_t close = text.find(']');
        if (close == std::string::npos || (close + 1 < text.size() && text[close + 1] != ':'))
        {
            throw std::invalid_argument("invalid bracketed host in endpoint '" + text + "'");
        }
        endpoint.host = text.substr(1, close - 1);
        has_port = close + 1 < text.size();
        port_text = has_port ? text.substr(close + 2) : "";
    }
    else if (std::count(text.begin(), text.end(), ':') > 1)
    {
        endpoint.host = text; // bare IPv6 address, no port
    }
    else
    {
        const size_t colon = text.find(':');
        endpoint.host = text.substr(0, colon);
        has_port = colon != std::string::npos;
        port_text = has_port ? text.substr(colon + 1) : "";
    }
    if (has_port)
    {
        unsigned long port = 0;
        const char* end = port_text.data() + port_text.size();
        const auto [stop, error] = std::from_chars(port_text.data(), end, port);
        if (error != std::errc{} || stop != end || port == 0 || port > 65535)
        {
            throw std::invalid_argument("invalid port in endpoint '" + text + "'");
        }
        endpoint.port = static_cast<uint16_t>(port);
    }
    if (endpoint.host.empty())
    {
        throw std::invalid_argument("missing host in endpoint '" + text + "'");
    }
    return endpoint;
}

RemoteAddress parse_remote(const std::string& text, uint16_t default_port)
{
    const std::string scheme = "ws://";
    if (text.starts_with("wss://"))
    {
        throw std::invalid_argument("wss:// is not supported ('" + text + "'); terminate TLS in a proxy, use ws://");
    }
    RemoteAddress remote;
    if (!text.starts_with(scheme))
    {
        remote.endpoint = parse_endpoint(text, default_port);
        return remote;
    }
    std::string authority = text.substr(scheme.size());
    const size_t slash = authority.find('/');
    remote.ws_path = slash == std::string::npos ? "/" : authority.substr(slash);
    if (slash != std::string::npos)
    {
        authority.resize(slash);
    }
    remote.endpoint = parse_endpoint(authority, default_port);
    return remote;
}

RemoteAddress parse_bridge_remote(const std::string& text)
{
    return parse_remote(text, text.starts_with("ws://") ? kDefaultBridgeWsPort : kDefaultBridgePort);
}

std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote, std::chrono::milliseconds timeout)
{
    TcpSocket socket = TcpSocket::connect(remote.endpoint, timeout);
    if (!remote.ws_path)
    {
        return std::make_unique<TcpTransport>(std::move(socket));
    }
    return std::make_unique<WebSocketTransport>(
        std::move(socket), host_header(remote.endpoint), *remote.ws_path, timeout);
}

} // namespace remote_plugin_session
