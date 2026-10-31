/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TcpTransport.h"
#include "TestHarness.h"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

std::vector<uint8_t> readBytes(int fd, size_t size)
{
    std::vector<uint8_t> data(size);
    size_t received = 0;
    while (received < size)
    {
        const ssize_t count = recv(fd, data.data() + received, size - received, 0);
        if (count <= 0)
        {
            data.resize(received);
            break;
        }
        received += static_cast<size_t>(count);
    }
    return data;
}

std::vector<uint8_t> readFile(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

} // namespace

int main()
{
    TestContext test("tcp_transport");
    const std::vector<uint8_t> packet = { 0x00, 0x00, 0x00, 0x05, 0x48, 0x56, 0x53, 0x32, 0x03 };

    {
        int sockets[2] = { -1, -1 };
        TEST_CHECK_EQ(test, socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
        argus_sender::TcpPacketSenderSettings settings;
        settings.kpi = false;
        int senderFd = sockets[0];
        argus_sender::TcpPacketSender sender(&senderFd, settings, 1, settings.port, "test");
        TEST_CHECK(test, sender.start());
        std::vector<uint8_t> queued = packet;
        TEST_CHECK(test, sender.enqueueFramed(std::move(queued)));
        sender.shutdown(true);
        TEST_CHECK(test, readBytes(sockets[1], packet.size()) == packet);
        close(sockets[0]);
        close(sockets[1]);
    }

    {
        char path[] = "/tmp/argus_sender_tcp_test_XXXXXX";
        const int temporaryFd = mkstemp(path);
        TEST_CHECK(test, temporaryFd >= 0);
        if (temporaryFd >= 0)
            close(temporaryFd);

        int sockets[2] = { -1, -1 };
        TEST_CHECK_EQ(test, socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
        argus_sender::TcpPacketSenderSettings settings;
        settings.kpi = false;
        settings.dumpWirePath = path;
        int senderFd = sockets[0];
        argus_sender::TcpPacketSender sender(&senderFd, settings, 1, settings.port, "dump-test");
        TEST_CHECK(test, sender.start());
        std::vector<uint8_t> queued = packet;
        TEST_CHECK(test, sender.enqueueFramed(std::move(queued)));
        sender.shutdown(true);
        TEST_CHECK(test, readBytes(sockets[1], packet.size()) == packet);
        TEST_CHECK(test, readFile(path) == packet);
        unlink(path);
        close(sockets[0]);
        close(sockets[1]);
    }

    {
        int sockets[2] = { -1, -1 };
        TEST_CHECK_EQ(test, socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
        argus_sender::TcpPacketSenderSettings settings;
        settings.kpi = false;
        int senderFd = sockets[0];
        argus_sender::TcpPacketSender sender(&senderFd, settings, 1, settings.port, "invalid-test");
        TEST_CHECK(test, sender.start());
        std::vector<uint8_t> malformed = { 0x00, 0x00, 0x00, 0x02, 0x01, 0x02, 0x03 };
        TEST_CHECK(test, !sender.enqueueFramed(std::move(malformed)));
        TEST_CHECK(test, sender.hasError());
        sender.shutdown(false);
        close(sockets[0]);
        close(sockets[1]);
    }

    {
        int sockets[2] = { -1, -1 };
        TEST_CHECK_EQ(test, socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
        argus_sender::TcpPacketSenderSettings settings;
        settings.kpi = false;
        int senderFd = sockets[0];
        argus_sender::TcpPacketSender sender(&senderFd, settings, 1, settings.port, "disconnect-test");
        TEST_CHECK(test, sender.start());
        close(sockets[1]);
        sockets[1] = -1;
        std::vector<uint8_t> queued = packet;
        TEST_CHECK(test, sender.enqueueFramed(std::move(queued)));
        for (uint32_t attempt = 0; attempt < 100 && !sender.disconnected(); ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        TEST_CHECK(test, sender.disconnected());
        TEST_CHECK(test, sender.hasError());
        sender.shutdown(false);
        close(sockets[0]);
    }

    {
        int sockets[2] = { -1, -1 };
        TEST_CHECK_EQ(test, socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
        int sendBufferBytes = 4096;
        TEST_CHECK_EQ(test, setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &sendBufferBytes, sizeof(sendBufferBytes)), 0);
        argus_sender::TcpPacketSenderSettings settings;
        settings.kpi = false;
        settings.sendPollTimeoutMs = 20;
        settings.drainTimeoutMs = 100;
        int senderFd = sockets[0];
        argus_sender::TcpPacketSender sender(&senderFd, settings, 1, settings.port, "stalled-drain-test");
        TEST_CHECK(test, sender.start());
        const size_t payloadBytes = 8U * 1024U * 1024U;
        std::vector<uint8_t> queued(payloadBytes + sizeof(uint32_t), 0x5a);
        const uint32_t framedSize = htonl(static_cast<uint32_t>(payloadBytes));
        std::memcpy(queued.data(), &framedSize, sizeof(framedSize));
        TEST_CHECK(test, sender.enqueueFramed(std::move(queued)));
        const std::chrono::steady_clock::time_point shutdownStart = std::chrono::steady_clock::now();
        sender.shutdown(true);
        const double shutdownSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - shutdownStart).count();
        TEST_CHECK(test, shutdownSeconds < 1.0);
        TEST_CHECK(test, sender.hasError());
        close(sockets[0]);
        close(sockets[1]);
    }

    return test.finish();
}
