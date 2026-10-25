// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Acts as a CUDA IPC consumer of the plugin: connects to its socket, receives the
// memory handle as a file descriptor, maps the shared allocation through the CUDA
// virtual memory API, and reads the frames it publishes. This is the path any
// out-of-process consumer, such as a JPEG-encoding server, would take.

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <core/frame_sink.hpp>
#include <core/sensing_camera.hpp>
#include <sensing_cuda_ipc/cuda_ipc_protocol.hpp>
#include <sys/socket.h>
#include <sys/un.h>

#include <atomic>
#include <cstring>
#include <cuda.h>
#include <memory>
#include <poll.h>
#include <thread>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

void check_cu(CUresult result, const char* what)
{
    if (result != CUDA_SUCCESS)
    {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string(what) + " failed: " + (name ? name : "unknown"));
    }
}

class IpcConsumer
{
public:
    explicit IpcConsumer(const std::string& socket_path)
    {
        m_socket = ::socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(m_socket >= 0);
        timeval timeout{ 2, 0 };
        ::setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path.c_str());
        REQUIRE(::connect(m_socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

        int fd = receive_hello();
        import_and_map(fd);
        ::close(fd); // the import keeps what it needs
    }

    ~IpcConsumer()
    {
        if (m_ptr)
        {
            cuMemUnmap(m_ptr, m_hello.total_bytes);
            cuMemAddressFree(m_ptr, m_hello.total_bytes);
        }
        if (m_handle)
            cuMemRelease(m_handle);
        if (m_socket >= 0)
            ::close(m_socket);
    }

    IpcConsumer(const IpcConsumer&) = delete;
    IpcConsumer& operator=(const IpcConsumer&) = delete;

    const ipc::Hello& hello() const
    {
        return m_hello;
    }

    /// Waits up to `limit` for the next published frame.
    bool next_frame(ipc::FrameReady& out, std::chrono::milliseconds limit)
    {
        pollfd pfd{ m_socket, POLLIN, 0 };
        if (::poll(&pfd, 1, static_cast<int>(limit.count())) <= 0)
            return false;
        const ssize_t got = ::recv(m_socket, &out, sizeof(out), MSG_WAITALL);
        REQUIRE(got == static_cast<ssize_t>(sizeof(out)));
        REQUIRE(out.magic == ipc::kFrameMagic);
        REQUIRE(out.slot < m_hello.slot_count);
        return true;
    }

    std::vector<uint8_t> read_slot(uint32_t slot) const
    {
        std::vector<uint8_t> host(static_cast<size_t>(m_hello.pitch) * m_hello.height);
        check_cu(cuMemcpyDtoH(host.data(), m_ptr + slot * m_hello.slot_stride, host.size()), "cuMemcpyDtoH");
        return host;
    }

    void release(uint32_t slot)
    {
        ipc::SlotRelease message{ ipc::kReleaseMagic, slot };
        REQUIRE(::send(m_socket, &message, sizeof(message), MSG_NOSIGNAL) == static_cast<ssize_t>(sizeof(message)));
    }

private:
    int receive_hello()
    {
        char control[CMSG_SPACE(sizeof(int))] = {};
        iovec iov{ &m_hello, sizeof(m_hello) };
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        REQUIRE(::recvmsg(m_socket, &msg, 0) == static_cast<ssize_t>(sizeof(m_hello)));

        cmsghdr* header = CMSG_FIRSTHDR(&msg);
        REQUIRE(header != nullptr);
        REQUIRE(header->cmsg_type == SCM_RIGHTS);
        int fd = -1;
        std::memcpy(&fd, CMSG_DATA(header), sizeof(fd));
        return fd;
    }

    void import_and_map(int fd)
    {
        REQUIRE(m_hello.magic == ipc::kHelloMagic);
        REQUIRE(m_hello.version == ipc::kProtocolVersion);

        CUdevice device = 0;
        CUcontext context = nullptr;
        check_cu(cuInit(0), "cuInit");
        check_cu(cuDeviceGet(&device, static_cast<int>(m_hello.device_id)), "cuDeviceGet");
        check_cu(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        check_cu(cuCtxSetCurrent(context), "cuCtxSetCurrent");

        check_cu(cuMemImportFromShareableHandle(&m_handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
                                                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                 "cuMemImportFromShareableHandle");
        check_cu(cuMemAddressReserve(&m_ptr, m_hello.total_bytes, 0, 0, 0), "cuMemAddressReserve");
        check_cu(cuMemMap(m_ptr, m_hello.total_bytes, 0, m_handle, 0), "cuMemMap");

        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = static_cast<int>(m_hello.device_id);
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check_cu(cuMemSetAccess(m_ptr, m_hello.total_bytes, &access, 1), "cuMemSetAccess");
    }

    int m_socket = -1;
    ipc::Hello m_hello{};
    CUmemGenericAllocationHandle m_handle = 0;
    CUdeviceptr m_ptr = 0;
};

SensingConfig make_config()
{
    SensingConfig config;
    config.platform_config_json = SENSING_PLATFORM_CONFIG;
    return config;
}

} // namespace

TEST_CASE("a consumer receives a described, mappable allocation", "[unit][gpu][sensing]")
{
    FastMock fast;
    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 1;
    streams[0].ipc_socket_path = unique_path("sensing_mock", ".sock");

    SensingCamera camera(make_config(), streams, std::make_unique<FrameSink>(streams));
    UpdateLoop loop(camera);
    IpcConsumer consumer(streams[0].ipc_socket_path);

    const auto& hello = consumer.hello();
    CHECK(hello.width == 320);
    CHECK(hello.height == 248);
    CHECK(hello.format == static_cast<uint32_t>(ipc::PixelFormat::Rgba8));
    CHECK(hello.sensor_id == 1);
    CHECK(hello.slot_count >= 3);
    CHECK(hello.pitch >= static_cast<uint64_t>(hello.width) * 4);
    CHECK(hello.slot_stride >= hello.pitch * hello.height);
    CHECK(hello.total_bytes >= hello.slot_stride * hello.slot_count);
}

TEST_CASE("published frames are whole, in order, and the picture moves", "[unit][gpu][sensing]")
{
    FastMock fast;
    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 0;
    streams[0].ipc_socket_path = unique_path("sensing_mock", ".sock");

    SensingCamera camera(make_config(), streams, std::make_unique<FrameSink>(streams));
    UpdateLoop loop(camera);
    IpcConsumer consumer(streams[0].ipc_socket_path);
    REQUIRE(consumer.hello().pitch == static_cast<uint64_t>(consumer.hello().width) * 4);

    uint64_t last_sequence = 0;
    uint64_t last_stamp = 0;
    std::vector<Rgb> colours;
    for (int i = 0; i < 20; ++i)
    {
        ipc::FrameReady frame{};
        REQUIRE(consumer.next_frame(frame, 2000ms));
        Shot shot;
        shot.width = consumer.hello().width;
        shot.height = consumer.hello().height;
        shot.rgba = consumer.read_slot(frame.slot);
        consumer.release(frame.slot);

        CHECK(frame.sequence > last_sequence);
        CHECK(frame.timestamp_ns > last_stamp);
        CHECK(is_whole(shot)); // not torn between two captures
        last_sequence = frame.sequence;
        last_stamp = frame.timestamp_ns;
        colours.push_back(shot.at(shot.width / 2, shot.height / 2));
    }

    // Twenty frames span several 40 ms GIF frames, so more than one colour appears.
    bool changed = false;
    for (const Rgb& c : colours)
        changed = changed || !near(c, colours[0].r, colours[0].g, colours[0].b, 30);
    CHECK(changed);
}

TEST_CASE("a consumer that holds its slots cannot stall the producer", "[unit][gpu][sensing]")
{
    FastMock fast;
    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 0;
    streams[0].ipc_socket_path = unique_path("sensing_mock", ".sock");

    SensingCamera camera(make_config(), streams, std::make_unique<FrameSink>(streams));
    UpdateLoop loop(camera);
    IpcConsumer consumer(streams[0].ipc_socket_path);

    // Take frames without releasing any: the producer can publish only as many as there are slots.
    std::vector<uint32_t> held;
    ipc::FrameReady frame{};
    while (consumer.next_frame(frame, 500ms))
        held.push_back(frame.slot);
    CHECK(held.size() == consumer.hello().slot_count);

    // The capture loop kept running the whole time.
    const uint64_t ticks_before = loop.ticks();
    std::this_thread::sleep_for(100ms);
    CHECK(loop.ticks() > ticks_before);

    // Releasing the slots lets frames flow again.
    for (uint32_t slot : held)
        consumer.release(slot);
    CHECK(consumer.next_frame(frame, 2000ms));
}
