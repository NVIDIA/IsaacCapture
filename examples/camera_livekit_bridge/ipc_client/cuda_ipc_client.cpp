// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/ipc_client/cuda_ipc_client.hpp"

#include <sys/socket.h>
#include <sys/un.h>

#include <cerrno>
#include <cstring>
#include <cuda_runtime_api.h>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>

namespace camera_bridge
{

namespace
{

constexpr std::size_t kMaxMessageBytes = 1024 * 1024;

[[noreturn]] void fail(const std::string& what)
{
    throw std::runtime_error(what);
}

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        fail(std::string(what) + ": " + cudaGetErrorString(status));
}

std::vector<unsigned char> base64_decode(const std::string& text)
{
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> out;
    unsigned int accumulator = 0;
    int bits = 0;
    for (const char c : text)
    {
        if (c == '=')
            break;
        const auto position = alphabet.find(c);
        if (position == std::string::npos)
            fail("invalid base64 in an IPC handle");
        accumulator = (accumulator << 6) | static_cast<unsigned int>(position);
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((accumulator >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace

CudaIpcClient::CudaIpcClient(std::string socket_path, std::vector<std::string> streams)
    : m_socket_path(std::move(socket_path)), m_requested(std::move(streams))
{
}

CudaIpcClient::~CudaIpcClient()
{
    close_all();
}

void CudaIpcClient::close_all()
{
    for (auto& stream : m_streams)
    {
        for (auto& slot : stream.slots)
        {
            if (slot.device_ptr != nullptr)
            {
                cudaSetDevice(stream.gpu_id);
                cudaIpcCloseMemHandle(slot.device_ptr);
                slot.device_ptr = nullptr;
            }
        }
    }
    m_streams.clear();
    if (m_fd >= 0)
    {
        ::close(m_fd);
        m_fd = -1;
    }
}

void CudaIpcClient::send_message(const nlohmann::json& message)
{
    const std::string line = message.dump() + "\n";
    std::size_t sent = 0;
    while (sent < line.size())
    {
        const ssize_t count = ::send(m_fd, line.data() + sent, line.size() - sent, MSG_NOSIGNAL);
        if (count < 0)
            fail(std::string("cannot send to the IPC publisher: ") + std::strerror(errno));
        sent += static_cast<std::size_t>(count);
    }
}

nlohmann::json CudaIpcClient::read_message(std::chrono::milliseconds timeout, bool& timed_out)
{
    timed_out = false;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;)
    {
        const auto newline = m_buffer.find('\n');
        if (newline != std::string::npos)
        {
            const std::string line = m_buffer.substr(0, newline);
            m_buffer.erase(0, newline + 1);
            auto message = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
            if (message.is_discarded() || !message.is_object())
                fail("the IPC publisher sent a message that is not a JSON object");
            return message;
        }
        if (m_buffer.size() > kMaxMessageBytes)
            fail("the IPC publisher sent an oversized message");
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (remaining.count() < 0)
        {
            timed_out = true;
            return {};
        }
        pollfd descriptor{ m_fd, POLLIN, 0 };
        const int ready = ::poll(&descriptor, 1, static_cast<int>(remaining.count()));
        if (ready < 0 && errno != EINTR)
            fail(std::string("poll on the IPC socket failed: ") + std::strerror(errno));
        if (ready <= 0)
            continue;
        char chunk[65536];
        const ssize_t count = ::recv(m_fd, chunk, sizeof(chunk), 0);
        if (count == 0)
            fail("the IPC publisher closed the connection");
        if (count < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            fail(std::string("cannot read from the IPC publisher: ") + std::strerror(errno));
        }
        m_buffer.append(chunk, static_cast<std::size_t>(count));
    }
}

void CudaIpcClient::connect()
{
    close_all();
    m_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_fd < 0)
        fail(std::string("cannot create a socket: ") + std::strerror(errno));
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (m_socket_path.size() >= sizeof(address.sun_path))
        fail("the IPC socket path is too long");
    std::strncpy(address.sun_path, m_socket_path.c_str(), sizeof(address.sun_path) - 1);
    if (::connect(m_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
        fail("cannot connect to the IPC socket " + m_socket_path + ": " + std::strerror(errno));

    bool timed_out = false;
    const auto hello = read_message(std::chrono::seconds(5), timed_out);
    if (timed_out || hello.value("type", "") != "hello")
        fail("expected a hello message from the IPC publisher");
    send_message({ { "type", "subscribe" }, { "streams", m_requested } });
    const auto subscribed = read_message(std::chrono::seconds(5), timed_out);
    if (timed_out)
        fail("the IPC publisher did not answer the subscription");
    if (subscribed.value("type", "") == "error")
        fail("the IPC subscription was refused: " + subscribed.value("error", std::string("no reason given")));
    if (subscribed.value("type", "") != "subscribed")
        fail("expected a subscribed message from the IPC publisher");
    open_streams(subscribed);
}

void CudaIpcClient::open_streams(const nlohmann::json& subscribed)
{
    for (const auto& entry : subscribed.at("streams"))
    {
        IpcStream stream;
        stream.name = entry.at("name").get<std::string>();
        stream.width = entry.at("width").get<int>();
        stream.height = entry.at("height").get<int>();
        stream.pixel_format = entry.value("pixel_format", std::string());
        stream.gpu_id = entry.value("gpu_id", 0);
        stream.layout = entry.value("layout", nlohmann::json::object());
        check_cuda(cudaSetDevice(stream.gpu_id), "cudaSetDevice");
        for (const auto& item : entry.at("slots"))
        {
            const auto handle_bytes = base64_decode(item.at("handle").get<std::string>());
            cudaIpcMemHandle_t handle{};
            if (handle_bytes.size() != sizeof(handle))
                fail("an IPC handle has the wrong size");
            std::memcpy(&handle, handle_bytes.data(), sizeof(handle));
            IpcSlot slot;
            check_cuda(
                cudaIpcOpenMemHandle(&slot.device_ptr, handle, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
            slot.nbytes = item.at("nbytes").get<std::size_t>();
            slot.shape = item.at("shape").get<std::vector<std::int64_t>>();
            slot.strides = item.at("strides").get<std::vector<std::int64_t>>();
            const auto index = static_cast<std::size_t>(item.at("slot").get<int>());
            if (stream.slots.size() <= index)
                stream.slots.resize(index + 1);
            stream.slots[index] = slot;
        }
        m_streams.push_back(std::move(stream));
    }
}

const IpcSlot& CudaIpcClient::slot(const FrameNotice& notice) const
{
    for (const auto& stream : m_streams)
    {
        if (stream.name == notice.stream)
            return stream.slots.at(static_cast<std::size_t>(notice.slot));
    }
    throw std::out_of_range("unknown IPC stream " + notice.stream);
}

bool CudaIpcClient::next_frame(FrameNotice& out, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;)
    {
        bool timed_out = false;
        const auto remaining =
            std::max(std::chrono::milliseconds(0),
                     std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()));
        auto message = read_message(remaining, timed_out);
        if (timed_out)
            return false;
        if (message.value("type", "") != "frame")
            continue;
        out.stream = message.at("stream").get<std::string>();
        out.slot = message.at("slot").get<int>();
        out.slot_generation = message.value("slot_generation", std::uint64_t{ 0 });
        out.metadata = message.value("metadata", nlohmann::json::object());
        return true;
    }
}

} // namespace camera_bridge
