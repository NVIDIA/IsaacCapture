// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>
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

    /// Blocks until the sender has opened both connections, each announced by a 4-byte hello: "DATA" carries the
    /// records and "CTRL" the small control messages (keyframe requests and the clock exchange). TCP_NODELAY on both.
    /// Any earlier session is closed.
    void accept_session();

    /// Ends a blocked accept_session() (it then throws or returns without a session) and stops accepting.
    void stop_listening();

    /// Unblocks every thread waiting in a read or write on this session.
    void shutdown_session();

    /// Control link: write all of `data` / read exactly `size` bytes. send_control is safe from several threads;
    /// recv_control returns false when the link ended.
    bool send_control(const uint8_t* data, size_t size);
    bool recv_control(uint8_t* out, size_t size);

    /// Bytes already received and waiting in the socket (how far behind the reader is).
    size_t pending_bytes() const;

    /// Asks the sender for a keyframe: one byte back on the connection. Safe to call from another thread.
    void request_keyframe();

    /// Reads one record's bytes (after the length prefix) into `record`. False on a clean end of stream; throws on a
    /// truncated or oversized record.
    bool read_record(std::vector<uint8_t>& record);

private:
    int m_listen = -1;
    int m_connection = -1; ///< DATA
    int m_control = -1; ///< CTRL
    std::mutex m_control_send;
};

} // namespace video_stage
