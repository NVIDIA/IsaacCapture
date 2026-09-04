// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <sensing_cuda_ipc/cuda_ipc_publisher.hpp>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

#include <cstring>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace plugins
{
namespace sensing
{
namespace
{

void check_cuda(CUresult result, const char* what)
{
    if (result == CUDA_SUCCESS)
        return;

    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream oss;
    oss << "CudaIpcPublisher: " << what << " failed";
    if (name)
        oss << ": " << name;
    if (text)
        oss << " (" << text << ")";
    throw std::runtime_error(oss.str());
}

void check_runtime(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "CudaIpcPublisher: " << what << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

size_t round_up(size_t value, size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

/// Send a fixed-size record, optionally with one fd attached. Non-blocking:
/// returns false on EAGAIN so a wedged consumer never stalls capture.
bool send_record(int fd, const void* data, size_t size, int passed_fd)
{
    iovec io{ const_cast<void*>(data), size };
    msghdr msg{};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    // CMSG_SPACE is not constexpr-friendly on all libcs; size for exactly one fd.
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
    if (passed_fd >= 0)
    {
        std::memset(control, 0, sizeof(control));
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(cmsg), &passed_fd, sizeof(int));
    }

    // MSG_NOSIGNAL: a consumer that exits mid-send must not SIGPIPE the plugin.
    ssize_t sent = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    return sent == static_cast<ssize_t>(size);
}

} // namespace

CudaIpcPublisher::CudaIpcPublisher(const CudaIpcConfig& config) : m_config(config)
{
    if (m_config.socket_path.empty())
        throw std::runtime_error("CudaIpcPublisher: socket path is empty");
    if (m_config.slot_count < 2 || m_config.slot_count > 64)
        throw std::runtime_error("CudaIpcPublisher: slot_count must be in [2, 64]");

    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&m_device, m_config.gpu_id), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&m_context, m_device), "cuDevicePrimaryCtxRetain");
    m_context_retained = true;
    check_cuda(cuCtxSetCurrent(m_context), "cuCtxSetCurrent");
    check_cuda(cuStreamCreate(&m_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

    allocate_slots();
    open_socket();

    std::cout << "CUDA IPC: sensor " << m_config.sensor_id << " serving " << m_config.width << "x" << m_config.height
              << " RGBA8 on " << m_config.socket_path << " (" << m_config.slot_count << " slots, "
              << (m_reserved_bytes >> 20) << " MiB)" << std::endl;
}

CudaIpcPublisher::~CudaIpcPublisher()
{
    if (m_client_fd >= 0)
        ::close(m_client_fd);
    if (m_listen_fd >= 0)
        ::close(m_listen_fd);
    if (m_socket_bound)
        ::unlink(m_config.socket_path.c_str());
    if (m_export_fd >= 0)
        ::close(m_export_fd);

    if (m_base_ptr)
    {
        cuMemUnmap(m_base_ptr, m_reserved_bytes);
        cuMemAddressFree(m_base_ptr, m_reserved_bytes);
    }
    if (m_alloc_handle)
        cuMemRelease(m_alloc_handle);
    if (m_stream)
        cuStreamDestroy(m_stream);
    if (m_context_retained)
        cuDevicePrimaryCtxRelease(m_device);
}

void CudaIpcPublisher::allocate_slots()
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = m_config.gpu_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity = 0;
    check_cuda(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "cuMemGetAllocationGranularity");

    // Rows are tightly packed so the consumer can wrap a slot as a plain
    // contiguous HxWx4 array. Slots are 256-byte aligned to keep every slot
    // base at CUDA's texture alignment.
    m_pitch = static_cast<size_t>(m_config.width) * 4;
    m_slot_stride = round_up(m_pitch * m_config.height, 256);
    m_reserved_bytes = round_up(m_slot_stride * m_config.slot_count, granularity);

    check_cuda(cuMemCreate(&m_alloc_handle, m_reserved_bytes, &prop, 0), "cuMemCreate");
    check_cuda(cuMemExportToShareableHandle(&m_export_fd, m_alloc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
               "cuMemExportToShareableHandle");
    check_cuda(cuMemAddressReserve(&m_base_ptr, m_reserved_bytes, 0, 0, 0), "cuMemAddressReserve");
    check_cuda(cuMemMap(m_base_ptr, m_reserved_bytes, 0, m_alloc_handle, 0), "cuMemMap");

    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_cuda(cuMemSetAccess(m_base_ptr, m_reserved_bytes, &access, 1), "cuMemSetAccess");

    // Opaque memory starts undefined; a consumer that attaches before the
    // first frame would otherwise render whatever the allocator handed back.
    check_runtime(cudaMemset(reinterpret_cast<void*>(m_base_ptr), 0, m_reserved_bytes), "cudaMemset");

    m_slot_sequence.assign(m_config.slot_count, 0);
}

void CudaIpcPublisher::open_socket()
{
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (m_config.socket_path.size() >= sizeof(addr.sun_path))
    {
        throw std::runtime_error("CudaIpcPublisher: socket path exceeds " + std::to_string(sizeof(addr.sun_path) - 1) +
                                 " bytes: " + m_config.socket_path);
    }
    std::memcpy(addr.sun_path, m_config.socket_path.c_str(), m_config.socket_path.size());

    m_listen_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (m_listen_fd < 0)
        throw std::runtime_error(std::string("CudaIpcPublisher: socket() failed: ") + std::strerror(errno));

    // A previous run that died on SIGKILL leaves the node behind and bind()
    // would fail with EADDRINUSE.
    ::unlink(m_config.socket_path.c_str());

    if (::bind(m_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        std::string err = std::strerror(errno);
        ::close(m_listen_fd);
        m_listen_fd = -1;
        throw std::runtime_error("CudaIpcPublisher: bind(" + m_config.socket_path + ") failed: " + err);
    }
    m_socket_bound = true;

    if (::listen(m_listen_fd, 1) < 0)
        throw std::runtime_error(std::string("CudaIpcPublisher: listen() failed: ") + std::strerror(errno));
}

void CudaIpcPublisher::accept_client()
{
    int fd = ::accept4(m_listen_fd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0)
        return; // EAGAIN: nobody waiting.

    if (m_client_fd >= 0)
    {
        // Last connect wins, so restarting the viewer does not need a plugin
        // restart. The old consumer sees EOF.
        std::cout << "CUDA IPC: sensor " << m_config.sensor_id << " replacing consumer" << std::endl;
        ::close(m_client_fd);
        m_client_fd = -1;
        m_unreleased = 0;
    }

    ipc::Hello hello{};
    hello.magic = ipc::kHelloMagic;
    hello.version = ipc::kProtocolVersion;
    hello.width = m_config.width;
    hello.height = m_config.height;
    hello.format = static_cast<uint32_t>(ipc::PixelFormat::Rgba8);
    hello.slot_count = m_config.slot_count;
    hello.device_id = static_cast<uint32_t>(m_config.gpu_id);
    hello.sensor_id = m_config.sensor_id;
    hello.pitch = m_pitch;
    hello.slot_stride = m_slot_stride;
    hello.total_bytes = m_reserved_bytes;

    if (!send_record(fd, &hello, sizeof(hello), m_export_fd))
    {
        std::cerr << "CUDA IPC: sensor " << m_config.sensor_id << " handshake send failed: " << std::strerror(errno)
                  << std::endl;
        ::close(fd);
        return;
    }

    m_client_fd = fd;
    std::cout << "CUDA IPC: sensor " << m_config.sensor_id << " consumer attached" << std::endl;
}

void CudaIpcPublisher::drain_releases()
{
    if (m_client_fd < 0)
        return;

    ipc::SlotRelease release{};
    while (true)
    {
        ssize_t got = ::recv(m_client_fd, &release, sizeof(release), MSG_DONTWAIT);
        if (got == 0)
        {
            drop_client("consumer disconnected");
            return;
        }
        if (got < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return;
            drop_client(std::strerror(errno));
            return;
        }
        if (got != sizeof(release) || release.magic != ipc::kReleaseMagic)
        {
            drop_client("protocol desync on release message");
            return;
        }
        if (release.slot >= m_config.slot_count)
        {
            drop_client("release named an out-of-range slot");
            return;
        }
        m_unreleased &= ~(uint64_t{ 1 } << release.slot);
    }
}

void CudaIpcPublisher::drop_client(const char* reason)
{
    if (m_client_fd < 0)
        return;
    std::cout << "CUDA IPC: sensor " << m_config.sensor_id << " consumer detached (" << reason << ")" << std::endl;
    ::close(m_client_fd);
    m_client_fd = -1;
    m_unreleased = 0;
}

void CudaIpcPublisher::poll()
{
    drain_releases();
    accept_client();
}

int CudaIpcPublisher::pick_slot() const
{
    int best = -1;
    uint64_t best_sequence = UINT64_MAX;
    for (uint32_t i = 0; i < m_config.slot_count; ++i)
    {
        if (m_unreleased & (uint64_t{ 1 } << i))
            continue;
        if (m_slot_sequence[i] < best_sequence)
        {
            best_sequence = m_slot_sequence[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

bool CudaIpcPublisher::publish(uintptr_t src_ptr, size_t src_pitch, uint64_t timestamp_ns)
{
    if (m_client_fd < 0 || src_ptr == 0)
        return false;

    const int slot = pick_slot();
    if (slot < 0)
    {
        ++m_dropped;
        return false;
    }

    check_cuda(cuCtxSetCurrent(m_context), "cuCtxSetCurrent publish");

    void* dst = reinterpret_cast<void*>(m_base_ptr + static_cast<size_t>(slot) * m_slot_stride);
    check_runtime(cudaMemcpy2DAsync(dst, m_pitch, reinterpret_cast<const void*>(src_ptr), src_pitch, m_pitch,
                                    m_config.height, cudaMemcpyDeviceToDevice, m_stream),
                  "cudaMemcpy2DAsync");

    // The consumer reads on its own context and stream and has no way to wait
    // on ours, so the copy must be complete before it is told the slot is
    // ready. Without this it renders a torn frame.
    check_runtime(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");

    ++m_sequence;
    m_slot_sequence[slot] = m_sequence;
    m_unreleased |= uint64_t{ 1 } << slot;

    ipc::FrameReady ready{};
    ready.magic = ipc::kFrameMagic;
    ready.slot = static_cast<uint32_t>(slot);
    ready.sequence = m_sequence;
    ready.timestamp_ns = timestamp_ns;

    if (!send_record(m_client_fd, &ready, sizeof(ready), -1))
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            // Consumer is behind and its socket buffer is full. Dropping the
            // notification is the mailbox-correct outcome: it will pick up the
            // next frame instead of falling further behind.
            m_unreleased &= ~(uint64_t{ 1 } << slot);
            ++m_dropped;
            return false;
        }
        drop_client(std::strerror(errno));
        return false;
    }

    return true;
}

} // namespace sensing
} // namespace plugins
