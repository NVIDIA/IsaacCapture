// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/tcp_transport.hpp"

#include "inc/remote_plugin_session/protocol_frame.hpp"

#include <utility>

namespace remote_plugin_session
{

namespace protocol = remote_plugin_protocol;

TcpTransport::TcpTransport(TcpSocket socket) : m_socket(std::move(socket))
{
}

bool TcpTransport::send(const uint8_t* frame, size_t size)
{
    return m_socket.send_all(frame, size);
}

ReceiveResult TcpTransport::receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout)
{
    frame.resize(protocol::kHeaderSize);
    switch (m_socket.recv_all(frame.data(), protocol::kHeaderSize, timeout))
    {
    case TcpSocket::RecvResult::Timeout:
        return ReceiveResult::Timeout;
    case TcpSocket::RecvResult::Closed:
        return ReceiveResult::Closed;
    case TcpSocket::RecvResult::Ok:
        break;
    }

    uint32_t payload_size = 0;
    try
    {
        payload_size = protocol::decode_header(frame).payload_size;
    }
    catch (const protocol::DecodeError&)
    {
        return ReceiveResult::Malformed;
    }
    if (payload_size == 0)
    {
        return ReceiveResult::Ok;
    }
    frame.resize(protocol::kHeaderSize + payload_size);
    // A header without its payload within the stall window is a broken peer, not a quiet one.
    return m_socket.recv_all(frame.data() + protocol::kHeaderSize, payload_size, kMidMessageStall) ==
                   TcpSocket::RecvResult::Ok ?
               ReceiveResult::Ok :
               ReceiveResult::Closed;
}

void TcpTransport::shutdown()
{
    m_socket.shutdown();
}

std::unique_ptr<ITransport> TcpTransport::accept(TcpSocket& listener, std::chrono::milliseconds poll, std::string* peer)
{
    TcpSocket client = listener.accept(poll, peer);
    return client.valid() ? std::make_unique<TcpTransport>(std::move(client)) : nullptr;
}

} // namespace remote_plugin_session
