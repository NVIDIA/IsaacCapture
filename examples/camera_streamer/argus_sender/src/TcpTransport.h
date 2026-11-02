/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_TCP_TRANSPORT_H
#define ARGUS_SENDER_TCP_TRANSPORT_H

#include "PacketSink.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace argus_sender
{

struct TcpPacketSenderSettings
{
    uint16_t port = 5000;
    std::string dumpWirePath;
    std::vector<std::string> streamLabels;
    std::vector<std::string> sampleLabels;
    bool kpi = true;
    uint32_t logIntervalSeconds = 1;
    uint32_t sendPollTimeoutMs = 100;
    uint32_t drainTimeoutMs = 2000;
};

int acceptTcpClient(const std::string& bindHost,
                    uint16_t port,
                    uint32_t retryDelayMs,
                    const std::atomic<bool>* abortFlag = NULL);

class TcpPacketSender : public PacketSink
{
public:
    TcpPacketSender(int* fd,
                    const TcpPacketSenderSettings& settings,
                    uint32_t maxQueuedPackets,
                    uint16_t port = 0,
                    const char* label = NULL,
                    bool appendDump = false);
    ~TcpPacketSender();

    bool start();
    bool enqueueFramed(std::vector<uint8_t>&& packet) override;
    bool enqueueFramedWithTiming(std::vector<uint8_t>&& packet, const PacketTiming& timing) override;
    bool hasError() const override;
    bool disconnected() const;
    void shutdown(bool drain);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_TCP_TRANSPORT_H
