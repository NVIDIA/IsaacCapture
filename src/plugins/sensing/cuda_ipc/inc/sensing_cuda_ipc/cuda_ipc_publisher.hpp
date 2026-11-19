// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Publishes captured frames to another process as CUDA device memory, with no
// encode and no host round-trip.
//
// Legacy CUDA IPC (cudaIpcGetMemHandle) does not work here. On Orin the
// producer-side call returns cudaSuccess and the consumer's
// cudaIpcOpenMemHandle then fails with cudaErrorInvalidValue, so the failure
// only shows up in the far process. The working route on Tegra is the virtual
// memory management API: cuMemCreate exports a POSIX file descriptor that the
// consumer maps with cuMemImportFromShareableHandle. Do not "simplify" this
// back to cudaIpcMemHandle_t.
//
// Wire protocol lives in cuda_ipc_protocol.hpp, shared with the Python
// consumer in examples/camera_viz/sources/cuda_ipc.py.

#pragma once

#include <sensing_cuda_ipc/cuda_ipc_protocol.hpp>

#include <cstdint>
#include <cuda.h>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

struct CudaIpcConfig
{
    /// Unix domain socket path the consumer connects to.
    std::string socket_path;
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t sensor_id = 0;
    int gpu_id = 0;
    /// Ring depth. One slot is held by the consumer while it renders, so
    /// three is the minimum that lets the producer keep writing; four leaves
    /// headroom for a consumer that briefly stalls.
    uint32_t slot_count = 4;
};

/**
 * @brief Serves RGBA8 frames to one consumer process over CUDA VMM + a Unix socket.
 *
 * Single-threaded and non-blocking throughout: poll() accepts connections and
 * reaps slot releases, publish() copies into a free slot and notifies. Neither
 * ever blocks the capture loop, so a stalled or absent consumer costs the
 * pipeline nothing.
 *
 * One consumer at a time. A second connection replaces the first, so
 * restarting the viewer does not require restarting the plugin.
 */
class CudaIpcPublisher
{
public:
    explicit CudaIpcPublisher(const CudaIpcConfig& config);
    ~CudaIpcPublisher();

    CudaIpcPublisher(const CudaIpcPublisher&) = delete;
    CudaIpcPublisher& operator=(const CudaIpcPublisher&) = delete;

    /** @brief Accept a pending consumer and reap released slots. Never blocks. */
    void poll();

    /**
     * @brief Copy one RGBA8 frame into a free slot and publish it.
     *
     * @param src_ptr    Device pointer to RGBA8 source (e.g. SiplCamera's
     *                   converted buffer).
     * @param src_pitch  Source row stride in bytes.
     * @return false if no consumer is attached, or the frame was dropped.
     */
    bool publish(uintptr_t src_ptr, size_t src_pitch, uint64_t timestamp_ns);

    bool has_consumer() const
    {
        return m_client_fd >= 0;
    }
    uint64_t published_count() const
    {
        return m_sequence;
    }
    uint64_t dropped_count() const
    {
        return m_dropped;
    }

private:
    void allocate_slots();
    void open_socket();
    void accept_client();
    void drain_releases();
    void drop_client(const char* reason);
    /// Least-recently-published slot the consumer has released, or -1.
    int pick_slot() const;

    CudaIpcConfig m_config;

    CUdevice m_device = 0;
    CUcontext m_context = nullptr;
    bool m_context_retained = false;
    CUstream m_stream = nullptr;

    /// One allocation carrying every slot; one fd exports the lot.
    CUmemGenericAllocationHandle m_alloc_handle = 0;
    CUdeviceptr m_base_ptr = 0;
    size_t m_reserved_bytes = 0;
    int m_export_fd = -1;

    size_t m_pitch = 0;
    size_t m_slot_stride = 0;

    /// Publish sequence each slot last carried; 0 means never written.
    std::vector<uint64_t> m_slot_sequence;
    /// Bit i set while slot i is published but not yet released by the
    /// consumer. Overwriting one of these would tear the frame it is reading,
    /// so publish() drops instead. Caps slot_count at 64.
    uint64_t m_unreleased = 0;

    int m_listen_fd = -1;
    int m_client_fd = -1;
    bool m_socket_bound = false;

    uint64_t m_sequence = 0;
    uint64_t m_dropped = 0;
};

} // namespace sensing
} // namespace plugins
