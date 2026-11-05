// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace video_stage
{

/// One listening TCP socket and one accepted connection that delivers length-prefixed AVS1 records.
class TcpReceiver
{
public:
    /// Listens on `port` (all interfaces). Throws std::runtime_error on failure.
    explicit TcpReceiver(uint16_t port);
    ~TcpReceiver();

    TcpReceiver(const TcpReceiver&) = delete;
    TcpReceiver& operator=(const TcpReceiver&) = delete;

    /// Blocks for one connection, with TCP_NODELAY. Any earlier connection is closed.
    void accept_connection();

    /// Reads one record's bytes (after the length prefix) into `record`. False on a clean end of stream; throws on a
    /// truncated or oversized record.
    bool read_record(std::vector<uint8_t>& record);

private:
    int m_listen = -1;
    int m_connection = -1;
};

} // namespace video_stage
