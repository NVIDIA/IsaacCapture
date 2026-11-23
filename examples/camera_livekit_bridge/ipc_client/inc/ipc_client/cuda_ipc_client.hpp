// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace camera_bridge
{

/// One ring slot of a stream: device memory opened from the publisher's CUDA IPC handle.
struct IpcSlot
{
    void* device_ptr = nullptr;
    std::size_t nbytes = 0;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
};

struct IpcStream
{
    std::string name;
    int width = 0;
    int height = 0;
    std::string pixel_format;
    int gpu_id = 0;
    /// Source colour as configured in the publisher: quantisation range and matrix ("bt601" or "bt709").
    bool full_range = false;
    std::string color_matrix;
    nlohmann::json layout;
    std::vector<IpcSlot> slots;
};

/// A frame notification: the slot now holds a new picture. The slot is overwritten without notice when the
/// publisher's ring wraps, so a consumer must finish with it before `slot_generation` advances by the ring size.
struct FrameNotice
{
    std::string stream;
    int slot = 0;
    std::uint64_t slot_generation = 0;
    /// When the publisher announced the picture, on its CLOCK_MONOTONIC (the same clock on this host); 0 if absent.
    std::uint64_t timestamp_ns = 0;
    nlohmann::json metadata;
};

/**
 * @brief Client of camera_receiver's CUDA IPC publisher (newline-delimited JSON on a unix socket).
 *
 * Protocol: the publisher sends `hello`; the client answers `subscribe` with stream names (all when empty); the
 * publisher replies `subscribed` with one CUDA IPC handle per slot, then sends `frame` notices. Pixels never cross
 * the socket. Must run in a different process from the publisher.
 */
class CudaIpcClient
{
public:
    CudaIpcClient(std::string socket_path, std::vector<std::string> streams);
    ~CudaIpcClient();

    CudaIpcClient(const CudaIpcClient&) = delete;
    CudaIpcClient& operator=(const CudaIpcClient&) = delete;

    /// The only user whose socket is trusted: the publisher's effective UID must match, or `connect()` refuses before
    /// reading anything it sends. Defaults to this process's effective UID; call before `connect()` to trust another.
    void set_trusted_peer_uid(uid_t uid)
    {
        m_trusted_peer_uid = uid;
        m_have_trusted_peer_uid = true;
    }

    /// Connects, subscribes and opens every slot. Throws std::runtime_error with the reason on failure.
    void connect();

    const std::vector<IpcStream>& streams() const
    {
        return m_streams;
    }

    /// The slot a frame notice refers to; throws std::out_of_range for an unknown stream or slot.
    const IpcSlot& slot(const FrameNotice& notice) const;

    /// Waits up to `timeout` for the next frame notice. Returns false on timeout; throws if the publisher went away.
    bool next_frame(FrameNotice& out, std::chrono::milliseconds timeout);

private:
    nlohmann::json read_message(std::chrono::milliseconds timeout, bool& timed_out);
    void send_message(const nlohmann::json& message);
    void open_streams(const nlohmann::json& subscribed);
    void close_all();

    std::string m_socket_path;
    std::vector<std::string> m_requested;
    int m_fd = -1;
    uid_t m_trusted_peer_uid = 0;
    bool m_have_trusted_peer_uid = false;
    std::string m_buffer;
    std::vector<IpcStream> m_streams;
};

} // namespace camera_bridge
