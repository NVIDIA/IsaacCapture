// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tcp_socket.hpp"
#include "transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace remote_plugin_session
{

//! ITransport over one TCP stream; messages are delimited by the wire header's payload_size.
class TcpTransport final : public ITransport
{
public:
    explicit TcpTransport(TcpSocket socket);

    //! One transport per connection accepted on `listener`; null when the poll expires. `peer`, when given,
    //! receives the client's numeric host:port.
    static std::unique_ptr<ITransport> accept(TcpSocket& listener,
                                              std::chrono::milliseconds poll,
                                              std::string* peer = nullptr);

    bool send(const uint8_t* frame, size_t size) override;
    ReceiveResult receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout) override;
    void shutdown() override;

private:
    TcpSocket m_socket;
};

} // namespace remote_plugin_session
