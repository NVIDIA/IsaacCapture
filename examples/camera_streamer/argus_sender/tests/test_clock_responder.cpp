/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#include "ClockExchange.h"
#include "ClockResponder.h"
#include "TestHarness.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

using argus_sender::kClockReplyBytes;
using argus_sender::kClockRequestBytes;

class UdpClient
{
public:
    explicit UdpClient(uint16_t serverPort, int timeoutMs = 1000)
    {
        m_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        timeval timeout;
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;
        ::setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::memset(&m_server, 0, sizeof(m_server));
        m_server.sin_family = AF_INET;
        m_server.sin_port = htons(serverPort);
        ::inet_pton(AF_INET, "127.0.0.1", &m_server.sin_addr);
    }

    ~UdpClient()
    {
        if (m_fd >= 0)
            ::close(m_fd);
    }

    UdpClient(const UdpClient&) = delete;
    UdpClient& operator=(const UdpClient&) = delete;

    bool ok() const
    {
        return m_fd >= 0;
    }

    bool send(const std::string& data)
    {
        return ::sendto(m_fd, data.data(), data.size(), 0, reinterpret_cast<const sockaddr*>(&m_server),
                        sizeof(m_server)) == static_cast<ssize_t>(data.size());
    }

    // Returns the datagram received, or false on timeout.
    bool receive(std::string* data, int timeoutMs)
    {
        timeval timeout;
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;
        ::setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        char buffer[256];
        const ssize_t n = ::recv(m_fd, buffer, sizeof(buffer), 0);
        if (n < 0)
            return false;
        data->assign(buffer, static_cast<size_t>(n));
        return true;
    }

private:
    int m_fd = -1;
    sockaddr_in m_server;
};

void putLe(std::string* out, uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; ++i)
        out->push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

uint64_t getLe(const std::string& data, size_t offset, int bytes)
{
    uint64_t value = 0;
    for (int i = bytes - 1; i >= 0; --i)
        value = (value << 8) | static_cast<uint8_t>(data[offset + static_cast<size_t>(i)]);
    return value;
}

std::string makeRequest(uint32_t seq, int64_t t1)
{
    std::string request(1, 'Q');
    putLe(&request, seq, 4);
    putLe(&request, static_cast<uint64_t>(t1), 8);
    return request;
}

struct Reply
{
    uint32_t seq;
    int64_t t1;
    int64_t t2;
    int64_t t3;
};

bool decodeReply(const std::string& data, Reply* reply)
{
    if (data.size() != kClockReplyBytes || data[0] != 'R')
        return false;
    reply->seq = static_cast<uint32_t>(getLe(data, 1, 4));
    reply->t1 = static_cast<int64_t>(getLe(data, 5, 8));
    reply->t2 = static_cast<int64_t>(getLe(data, 13, 8));
    reply->t3 = static_cast<int64_t>(getLe(data, 21, 8));
    return true;
}

argus_sender::ClockResponderConfig loopbackConfig()
{
    argus_sender::ClockResponderConfig config;
    config.bindHost = "127.0.0.1";
    config.port = 0;
    return config;
}

} // namespace

int main()
{
    TestContext test("clock_responder");

    // One request, timestamps bracketed by the test's own clock readings.
    {
        argus_sender::ClockResponder responder(loopbackConfig());
        TEST_CHECK(test, responder.port() != 0);
        UdpClient client(responder.port());
        TEST_CHECK(test, client.ok());
        const int64_t before = argus_sender::monotonicClockNs();
        TEST_CHECK(test, client.send(makeRequest(0xFFFFFFF0U, -1234)));
        std::string data;
        TEST_CHECK(test, client.receive(&data, 1000));
        const int64_t after = argus_sender::monotonicClockNs();
        TEST_CHECK_EQ(test, data.size(), kClockReplyBytes);
        Reply reply;
        TEST_CHECK(test, decodeReply(data, &reply));
        TEST_CHECK_EQ(test, reply.seq, 0xFFFFFFF0U);
        TEST_CHECK_EQ(test, reply.t1, static_cast<int64_t>(-1234));
        TEST_CHECK(test, reply.t2 <= reply.t3);
        TEST_CHECK(test, before <= reply.t2);
        TEST_CHECK(test, reply.t3 <= after);
    }

    // Back-to-back requests, and the answered() count.
    {
        argus_sender::ClockResponder responder(loopbackConfig());
        UdpClient client(responder.port());
        int matched = 0;
        for (uint32_t seq = 0; seq < 50; ++seq)
        {
            std::string data;
            Reply reply;
            if (client.send(makeRequest(seq, 1000 + seq)) && client.receive(&data, 1000) && decodeReply(data, &reply) &&
                reply.seq == seq && reply.t1 == static_cast<int64_t>(1000 + seq))
            {
                ++matched;
            }
        }
        TEST_CHECK_EQ(test, matched, 50);
        TEST_CHECK_EQ(test, responder.answered(), static_cast<uint64_t>(50));
    }

    // Anything that is not exactly one request gets no reply; a valid one afterwards still works.
    {
        argus_sender::ClockResponder responder(loopbackConfig());
        UdpClient client(responder.port());
        const std::string good = makeRequest(9, 77);
        std::string wrongTag = good;
        wrongTag[0] = 'X';
        const std::vector<std::string> bad = { std::string(), good.substr(0, 12), good + 'x', wrongTag };
        for (const std::string& datagram : bad)
        {
            TEST_CHECK(test, client.send(datagram));
            std::string data;
            TEST_CHECK(test, !client.receive(&data, 300));
        }
        TEST_CHECK_EQ(test, responder.answered(), static_cast<uint64_t>(0));

        TEST_CHECK(test, client.send(good));
        std::string data;
        Reply reply;
        TEST_CHECK(test, client.receive(&data, 1000));
        TEST_CHECK(test, decodeReply(data, &reply));
        TEST_CHECK_EQ(test, reply.seq, 9U);
        TEST_CHECK_EQ(test, reply.t1, static_cast<int64_t>(77));
        TEST_CHECK_EQ(test, responder.answered(), static_cast<uint64_t>(1));
    }

    // Two clients each get their own reply.
    {
        argus_sender::ClockResponder responder(loopbackConfig());
        UdpClient a(responder.port());
        UdpClient b(responder.port());
        TEST_CHECK(test, a.send(makeRequest(1, 111)));
        TEST_CHECK(test, b.send(makeRequest(2, 222)));
        std::string dataA;
        std::string dataB;
        TEST_CHECK(test, a.receive(&dataA, 1000));
        TEST_CHECK(test, b.receive(&dataB, 1000));
        Reply replyA;
        Reply replyB;
        TEST_CHECK(test, decodeReply(dataA, &replyA));
        TEST_CHECK(test, decodeReply(dataB, &replyB));
        TEST_CHECK_EQ(test, replyA.seq, 1U);
        TEST_CHECK_EQ(test, replyA.t1, static_cast<int64_t>(111));
        TEST_CHECK_EQ(test, replyB.seq, 2U);
        TEST_CHECK_EQ(test, replyB.t1, static_cast<int64_t>(222));
        TEST_CHECK_EQ(test, responder.answered(), static_cast<uint64_t>(2));
    }

    // Destruction is prompt.
    {
        std::unique_ptr<argus_sender::ClockResponder> responder(new argus_sender::ClockResponder(loopbackConfig()));
        const auto start = std::chrono::steady_clock::now();
        responder.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        TEST_CHECK(test, elapsed < std::chrono::seconds(1));
    }

    // The same address cannot be bound twice.
    {
        argus_sender::ClockResponder first(loopbackConfig());
        argus_sender::ClockResponderConfig config = loopbackConfig();
        config.port = first.port();
        bool threw = false;
        std::string message;
        try
        {
            argus_sender::ClockResponder second(config);
        }
        catch (const std::runtime_error& e)
        {
            threw = true;
            message = e.what();
        }
        TEST_CHECK(test, threw);
        TEST_CHECK(test, !message.empty());
    }

    // A bind host that is not an IPv4 address.
    {
        for (const char* host : { "not-an-address", "::1", "" })
        {
            argus_sender::ClockResponderConfig config = loopbackConfig();
            config.bindHost = host;
            bool threw = false;
            std::string message;
            try
            {
                argus_sender::ClockResponder responder(config);
            }
            catch (const std::runtime_error& e)
            {
                threw = true;
                message = e.what();
            }
            TEST_CHECK(test, threw);
            TEST_CHECK(test, !message.empty());
        }
    }

    return test.finish();
}
