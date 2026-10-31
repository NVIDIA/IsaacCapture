/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TcpTransport.h"

#include "Runtime.h"
#include "StereoTiming.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <errno.h>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <netdb.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace argus_sender
{
uint64_t tcpSteadyNowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct LatencyAccumulator
{
    uint64_t samples = 0;
    uint64_t sumNs = 0;
    uint64_t maxNs = 0;

    void record(uint64_t valueNs)
    {
        ++samples;
        sumNs += valueNs;
        maxNs = std::max(maxNs, valueNs);
    }

    double averageMs() const
    {
        return samples > 0 ? static_cast<double>(sumNs) / static_cast<double>(samples) / 1000000.0 : 0.0;
    }

    double maximumMs() const
    {
        return static_cast<double>(maxNs) / 1000000.0;
    }

    void reset()
    {
        samples = 0;
        sumNs = 0;
        maxNs = 0;
    }
};

struct StreamLatencyStats
{
    bool encoded = false;
    LatencyAccumulator acquireOffset;
    LatencyAccumulator acquireSpread;
    LatencyAccumulator captureToCopyDone;
    LatencyAccumulator captureToProcessStart;
    LatencyAccumulator captureToProcessDone;
    LatencyAccumulator processDoneToSent;
    LatencyAccumulator captureToSent;

    void reset()
    {
        acquireOffset.reset();
        acquireSpread.reset();
        captureToCopyDone.reset();
        captureToProcessStart.reset();
        captureToProcessDone.reset();
        processDoneToSent.reset();
        captureToSent.reset();
    }
};

void recordDuration(uint64_t startNs, uint64_t endNs, LatencyAccumulator* metric)
{
    if (metric && startNs != 0 && endNs >= startNs)
        metric->record(endNs - startNs);
}

bool shouldStop(const std::atomic<bool>* abortFlag)
{
    return stopRequested() || (abortFlag && abortFlag->load());
}

int acceptTcpClient(const std::string& bindHost, uint16_t port, uint32_t retryDelayMs, const std::atomic<bool>* abortFlag)
{
    char portString[16];
    std::snprintf(portString, sizeof(portString), "%u", static_cast<unsigned>(port));

    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    while (!shouldStop(abortFlag))
    {
        struct addrinfo* result = NULL;
        const char* node = bindHost.empty() ? NULL : bindHost.c_str();
        int rc = getaddrinfo(node, portString, &hints, &result);
        if (rc != 0)
        {
            std::fprintf(stderr, "getaddrinfo(%s:%s) failed: %s\n", bindHost.c_str(), portString, gai_strerror(rc));
        }
        else
        {
            for (struct addrinfo* rp = result; rp != NULL && !shouldStop(abortFlag); rp = rp->ai_next)
            {
                int listenFd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
                if (listenFd < 0)
                    continue;

                int yes = 1;
                setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
                struct timeval tv;
                tv.tv_sec = 0;
                tv.tv_usec = 500000;
                setsockopt(listenFd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

                if (bind(listenFd, rp->ai_addr, rp->ai_addrlen) != 0 || listen(listenFd, 1) != 0)
                {
                    close(listenFd);
                    continue;
                }

                std::printf("Listening for TCP receiver on %s:%u\n", bindHost.c_str(), port);
                while (!shouldStop(abortFlag))
                {
                    struct sockaddr_storage peerAddr;
                    socklen_t peerLen = sizeof(peerAddr);
                    int fd = accept(listenFd, reinterpret_cast<struct sockaddr*>(&peerAddr), &peerLen);
                    if (fd >= 0)
                    {
                        int nodelay = 1;
                        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

                        char peerHost[NI_MAXHOST];
                        char peerService[NI_MAXSERV];
                        if (getnameinfo(reinterpret_cast<struct sockaddr*>(&peerAddr), peerLen, peerHost, sizeof(peerHost),
                                        peerService, sizeof(peerService), NI_NUMERICHOST | NI_NUMERICSERV) == 0)
                        {
                            std::printf("Receiver connected from %s:%s\n", peerHost, peerService);
                        }
                        else
                        {
                            std::printf("Receiver connected\n");
                        }
                        close(listenFd);
                        freeaddrinfo(result);
                        return fd;
                    }
                    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                        continue;
                    std::fprintf(stderr, "accept() failed: %s\n", std::strerror(errno));
                    break;
                }
                close(listenFd);
            }
            freeaddrinfo(result);
        }

        std::fprintf(stderr, "Waiting to listen for TCP receiver on %s:%u...\n", bindHost.c_str(), port);
        const uint32_t sleepStepMs = 50;
        uint32_t sleptMs = 0;
        while (sleptMs < retryDelayMs && !shouldStop(abortFlag))
        {
            const uint32_t remainingMs = retryDelayMs - sleptMs;
            const uint32_t waitMs = remainingMs < sleepStepMs ? remainingMs : sleepStepMs;
            std::this_thread::sleep_for(std::chrono::milliseconds(waitMs));
            sleptMs += waitMs;
        }
    }

    return -1;
}

bool sendAll(int fd,
             const uint8_t* data,
             size_t size,
             const std::atomic<bool>* abortFlag,
             const std::atomic<uint64_t>* deadlineSteadyNs)
{
    size_t sent = 0;
    while (sent < size && !shouldStop(abortFlag))
    {
        const uint64_t deadline = deadlineSteadyNs ? deadlineSteadyNs->load() : 0;
        if (deadline != 0 && tcpSteadyNowNs() >= deadline)
            return false;
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, data + sent, size - sent, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, data + sent, size - sent, 0);
#endif
        if (n > 0)
        {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        if (!shouldStop(abortFlag))
        {
            if (n == 0)
                std::fprintf(stderr, "TCP send returned zero; closing receiver session\n");
            else
                std::fprintf(stderr, "TCP send failed (%s); closing receiver session\n", std::strerror(errno));
        }
        return false;
    }
    return !shouldStop(abortFlag);
}


class TcpPacketSender::Impl
{
public:
    struct QueuedPacket
    {
        std::vector<uint8_t> bytes;
        PacketTiming timing;
    };

    Impl(int* fd,
         const TcpPacketSenderSettings& settings,
         uint32_t maxQueuedPackets,
         uint16_t port = 0,
         const char* label = NULL,
         bool appendDump = false)
        : m_fd(fd),
          m_settings(settings),
          m_port(port ? port : settings.port),
          m_label(label ? label : ""),
          m_maxQueuedPackets(maxQueuedPackets ? maxQueuedPackets : 1),
          m_appendDump(appendDump),
          m_dumpFd(-1),
          m_dumpBytes(0),
          m_accepting(false),
          m_abort(false),
          m_drainDeadlineSteadyNs(0),
          m_error(false),
          m_disconnected(false),
          m_totalPackets(0),
          m_intervalPackets(0),
          m_intervalBytes(0),
          m_enqueueWaitSamples(0),
          m_enqueueWaitSumNs(0),
          m_enqueueWaitMaxNs(0),
          m_intervalSendSamples(0),
          m_intervalSendSumNs(0),
          m_intervalSendMaxNs(0),
          m_intervalLatencySamples(0),
          m_intervalEncodeStartSamples(0),
          m_acquireSpreadSumNs(0),
          m_acquireSpreadMaxNs(0),
          m_captureToEncodeStartSumNs(0),
          m_captureToEncodeStartMaxNs(0),
          m_captureToProcessDoneSumNs(0),
          m_captureToProcessDoneMaxNs(0),
          m_captureToSentSumNs(0),
          m_captureToSentMaxNs(0)
    {
    }

    ~Impl()
    {
        shutdown(false);
    }

    bool start()
    {
        if (m_thread.joinable())
            return true;
        if (!openWireDump())
            return false;
        if (!m_fd || *m_fd < 0 || m_settings.sendPollTimeoutMs == 0)
        {
            std::fprintf(stderr, "Invalid TCP sender socket or send timeout\n");
            closeWireDump();
            return false;
        }
        struct timeval sendTimeout;
        sendTimeout.tv_sec = m_settings.sendPollTimeoutMs / 1000U;
        sendTimeout.tv_usec = static_cast<suseconds_t>(m_settings.sendPollTimeoutMs % 1000U) * 1000;
        if (setsockopt(*m_fd, SOL_SOCKET, SO_SNDTIMEO, &sendTimeout, sizeof(sendTimeout)) != 0)
        {
            std::fprintf(stderr, "Failed to configure TCP send timeout: %s\n", std::strerror(errno));
            closeWireDump();
            return false;
        }
        m_statsStart = std::chrono::steady_clock::now();
        m_accepting = true;
        m_abort = false;
        m_drainDeadlineSteadyNs = 0;
        m_error = false;
        m_disconnected = false;
        try
        {
            m_thread = std::thread(&Impl::threadMain, this);
        }
        catch (...)
        {
            m_accepting = false;
            m_error = true;
            closeWireDump();
            std::fprintf(stderr, "Failed to start HEVC TCP sender thread\n");
            return false;
        }
        return true;
    }

    bool enqueueFramed(std::vector<uint8_t>&& packet, const PacketTiming& timing)
    {
        if (packet.size() <= sizeof(uint32_t) || packet.size() - sizeof(uint32_t) > std::numeric_limits<uint32_t>::max())
        {
            std::fprintf(stderr, "Invalid length-prefixed TCP packet size: %zu bytes\n", packet.size());
            m_error = true;
            return false;
        }

        uint32_t framedSize = 0;
        std::memcpy(&framedSize, packet.data(), sizeof(framedSize));
        const size_t payloadSize = packet.size() - sizeof(framedSize);
        if (ntohl(framedSize) != payloadSize)
        {
            std::fprintf(
                stderr, "Length-prefixed TCP packet mismatch: prefix=%u payload=%zu\n", ntohl(framedSize), payloadSize);
            m_error = true;
            return false;
        }
        const auto enqueueWaitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_queue.size() >= m_maxQueuedPackets && m_accepting && !m_abort && !m_error && !stopRequested())
        {
            m_spaceAvailable.wait_for(lock, std::chrono::milliseconds(1));
        }

        if (!m_accepting || m_abort || m_error || stopRequested())
            return false;

        QueuedPacket queued;
        queued.bytes = std::move(packet);
        queued.timing = timing;
        m_queue.push_back(std::move(queued));
        lock.unlock();

        const uint64_t enqueueWaitNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - enqueueWaitStart)
                .count());
        {
            std::lock_guard<std::mutex> metricsLock(m_metricsMutex);
            ++m_enqueueWaitSamples;
            m_enqueueWaitSumNs += enqueueWaitNs;
            if (enqueueWaitNs > m_enqueueWaitMaxNs)
                m_enqueueWaitMaxNs = enqueueWaitNs;
        }
        m_packetAvailable.notify_one();
        return true;
    }

    bool hasError() const
    {
        return m_error.load();
    }

    bool disconnected() const
    {
        return m_disconnected.load();
    }

    void shutdown(bool drain)
    {
        if (m_thread.joinable())
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_accepting = false;
                if (drain)
                {
                    const uint64_t timeoutNs = static_cast<uint64_t>(m_settings.drainTimeoutMs) * 1000000ULL;
                    const uint64_t nowNs = tcpSteadyNowNs();
                    m_drainDeadlineSteadyNs = timeoutNs > std::numeric_limits<uint64_t>::max() - nowNs ?
                                                  std::numeric_limits<uint64_t>::max() :
                                                  nowNs + timeoutNs;
                }
                else
                {
                    m_abort = true;
                    m_queue.clear();
                }
            }

            m_packetAvailable.notify_all();
            m_spaceAvailable.notify_all();
            if (!drain && m_fd && *m_fd >= 0)
                ::shutdown(*m_fd, SHUT_RDWR);

            m_thread.join();
        }
        closeWireDump();
    }

private:
    bool openWireDump()
    {
        if (m_settings.dumpWirePath.empty())
            return true;

        const int flags = O_WRONLY | O_CREAT | (m_appendDump ? O_APPEND : O_TRUNC);
        m_dumpFd = ::open(m_settings.dumpWirePath.c_str(), flags, 0666);
        if (m_dumpFd < 0)
        {
            std::fprintf(
                stderr, "Failed to open wire dump %s: %s\n", m_settings.dumpWirePath.c_str(), std::strerror(errno));
            m_error = true;
            return false;
        }

        std::printf("Wire dump %s: %s\n", m_appendDump ? "appending" : "started", m_settings.dumpWirePath.c_str());
        return true;
    }

    bool writeWireDump(const uint8_t* data, size_t size)
    {
        if (m_dumpFd < 0)
            return true;

        size_t written = 0;
        while (written < size)
        {
            const ssize_t n = ::write(m_dumpFd, data + written, size - written);
            if (n > 0)
            {
                written += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && errno == EINTR)
                continue;

            if (n == 0)
                std::fprintf(stderr, "Wire dump write returned zero for %s\n", m_settings.dumpWirePath.c_str());
            else
                std::fprintf(stderr, "Wire dump write failed for %s: %s\n", m_settings.dumpWirePath.c_str(),
                             std::strerror(errno));
            return false;
        }

        m_dumpBytes += size;
        return true;
    }

    void closeWireDump()
    {
        if (m_dumpFd < 0)
            return;

        if (::close(m_dumpFd) != 0)
            std::fprintf(
                stderr, "Failed to close wire dump %s: %s\n", m_settings.dumpWirePath.c_str(), std::strerror(errno));
        else
            std::printf("Wire dump closed: %s, session_bytes=%llu\n", m_settings.dumpWirePath.c_str(),
                        static_cast<unsigned long long>(m_dumpBytes));
        m_dumpFd = -1;
    }

    void recordStreamLatency(const PacketTiming& timing, uint64_t sendDoneSteadyNs)
    {
        if (m_streamLatencyStats.size() < timing.streams.size())
            m_streamLatencyStats.resize(timing.streams.size());
        for (size_t i = 0; i < timing.streams.size(); ++i)
        {
            const PacketStreamTiming& stream = timing.streams[i];
            if (!stream.present || stream.captureStartSteadyNs == 0)
                continue;
            StreamLatencyStats& stats = m_streamLatencyStats[i];
            stats.encoded = stream.encoded;
            if (timing.captureStartSteadyNs != 0 && stream.captureStartSteadyNs >= timing.captureStartSteadyNs)
            {
                stats.acquireOffset.record(stream.captureStartSteadyNs - timing.captureStartSteadyNs);
            }
            stats.acquireSpread.record(stream.acquireSpreadNs);
            recordDuration(stream.captureStartSteadyNs, stream.copyDoneSteadyNs, &stats.captureToCopyDone);
            recordDuration(stream.captureStartSteadyNs, stream.processStartSteadyNs, &stats.captureToProcessStart);
            recordDuration(stream.captureStartSteadyNs, stream.processDoneSteadyNs, &stats.captureToProcessDone);
            recordDuration(stream.processDoneSteadyNs, sendDoneSteadyNs, &stats.processDoneToSent);
            recordDuration(stream.captureStartSteadyNs, sendDoneSteadyNs, &stats.captureToSent);
        }
    }

    void recordSofToSentLatency(const PacketTiming& timing, uint64_t sendDoneTscNs)
    {
        if (sendDoneTscNs == 0)
            return;
        if (m_sofToSentStats.size() < timing.samples.size())
            m_sofToSentStats.resize(timing.samples.size());
        for (size_t i = 0; i < timing.samples.size(); ++i)
        {
            const PacketSampleTiming& sample = timing.samples[i];
            if (sample.present && sample.sensorSofTscNs != 0 && sendDoneTscNs >= sample.sensorSofTscNs)
            {
                m_sofToSentStats[i].record(sendDoneTscNs - sample.sensorSofTscNs);
            }
        }
    }

    void logAndResetSofToSentLatency()
    {
        bool first = true;
        for (size_t i = 0; i < m_sofToSentStats.size(); ++i)
        {
            LatencyAccumulator& stats = m_sofToSentStats[i];
            if (stats.samples == 0)
                continue;
            if (first)
            {
                std::printf("argus_sender[sof-to-sent]: sensor_sof_to_tcp_sent_ms(avg/max) ");
            }
            const std::string label = i < m_settings.sampleLabels.size() && !m_settings.sampleLabels[i].empty() ?
                                          m_settings.sampleLabels[i] :
                                          std::string("camera") + std::to_string(i);
            std::printf("%s%s=%.3f/%.3f", first ? "" : ", ", label.c_str(), stats.averageMs(), stats.maximumMs());
            first = false;
            stats.reset();
        }
        if (!first)
            std::printf("\n");
    }

    void logAndResetStreamLatency()
    {
        if (!verboseLoggingEnabled())
        {
            bool first = true;
            for (size_t i = 0; i < m_streamLatencyStats.size(); ++i)
            {
                StreamLatencyStats& stats = m_streamLatencyStats[i];
                if (stats.captureToSent.samples == 0)
                    continue;
                const std::string label = i < m_settings.streamLabels.size() && !m_settings.streamLabels[i].empty() ?
                                              m_settings.streamLabels[i] :
                                              std::string("stream") + std::to_string(i);
                if (first)
                {
                    std::printf("argus_sender[e2e]: capture_to_tcp_sent_ms(avg/max) ");
                }
                std::printf("%s%s=%.3f/%.3f", first ? "" : ", ", label.c_str(), stats.captureToSent.averageMs(),
                            stats.captureToSent.maximumMs());
                first = false;
                stats.reset();
            }
            if (!first)
                std::printf("\n");
            return;
        }

        for (size_t i = 0; i < m_streamLatencyStats.size(); ++i)
        {
            StreamLatencyStats& stats = m_streamLatencyStats[i];
            if (stats.captureToSent.samples == 0)
                continue;
            const std::string label = i < m_settings.streamLabels.size() && !m_settings.streamLabels[i].empty() ?
                                          m_settings.streamLabels[i] :
                                          std::string("stream") + std::to_string(i);
            if (stats.encoded)
            {
                std::printf(
                    "argus_sender[latency:%s]: n=%llu, "
                    "acquire_offset_ms(avg/max)=%.3f/%.3f, "
                    "stream_acquire_spread_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_copy_done_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_encode_start_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_encode_done_ms(avg/max)=%.3f/%.3f, "
                    "encode_done_to_tcp_sent_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_tcp_sent_ms(avg/max)=%.3f/%.3f\n",
                    label.c_str(), static_cast<unsigned long long>(stats.captureToSent.samples),
                    stats.acquireOffset.averageMs(), stats.acquireOffset.maximumMs(), stats.acquireSpread.averageMs(),
                    stats.acquireSpread.maximumMs(), stats.captureToCopyDone.averageMs(),
                    stats.captureToCopyDone.maximumMs(), stats.captureToProcessStart.averageMs(),
                    stats.captureToProcessStart.maximumMs(), stats.captureToProcessDone.averageMs(),
                    stats.captureToProcessDone.maximumMs(), stats.processDoneToSent.averageMs(),
                    stats.processDoneToSent.maximumMs(), stats.captureToSent.averageMs(),
                    stats.captureToSent.maximumMs());
            }
            else
            {
                std::printf(
                    "argus_sender[latency:%s]: n=%llu, "
                    "acquire_offset_ms(avg/max)=%.3f/%.3f, "
                    "stream_acquire_spread_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_copy_done_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_raw_ready_ms(avg/max)=%.3f/%.3f, "
                    "raw_ready_to_tcp_sent_ms(avg/max)=%.3f/%.3f, "
                    "capture_to_tcp_sent_ms(avg/max)=%.3f/%.3f\n",
                    label.c_str(), static_cast<unsigned long long>(stats.captureToSent.samples),
                    stats.acquireOffset.averageMs(), stats.acquireOffset.maximumMs(), stats.acquireSpread.averageMs(),
                    stats.acquireSpread.maximumMs(), stats.captureToCopyDone.averageMs(),
                    stats.captureToCopyDone.maximumMs(), stats.captureToProcessDone.averageMs(),
                    stats.captureToProcessDone.maximumMs(), stats.processDoneToSent.averageMs(),
                    stats.processDoneToSent.maximumMs(), stats.captureToSent.averageMs(),
                    stats.captureToSent.maximumMs());
            }
            stats.reset();
        }
    }

    void threadMain()
    {
        while (!m_abort && !stopRequested())
        {
            QueuedPacket packet;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_packetAvailable.wait(
                    lock, [this]() { return m_abort || !m_accepting || !m_queue.empty() || stopRequested(); });

                if (m_queue.empty())
                {
                    if (m_abort || !m_accepting || stopRequested())
                        break;
                    continue;
                }

                packet = std::move(m_queue.front());
                m_queue.pop_front();
                m_spaceAvailable.notify_one();
            }

            const auto sendStart = std::chrono::steady_clock::now();
            if (!sendAll(*m_fd, packet.bytes.data(), packet.bytes.size(), &m_abort, &m_drainDeadlineSteadyNs))
            {
                if (!m_abort && !stopRequested())
                {
                    const uint64_t deadline = m_drainDeadlineSteadyNs.load();
                    const bool drainTimedOut = deadline != 0 && tcpSteadyNowNs() >= deadline;
                    if (drainTimedOut)
                    {
                        std::fprintf(stderr, "TCP drain timed out after %u ms; aborting the sender socket\n",
                                     m_settings.drainTimeoutMs);
                        if (m_fd && *m_fd >= 0)
                            ::shutdown(*m_fd, SHUT_RDWR);
                    }
                    else
                    {
                        m_disconnected = true;
                    }
                    m_error = true;
                }
                break;
            }
            const uint64_t sendNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sendStart).count());
            ++m_intervalSendSamples;
            m_intervalSendSumNs += sendNs;
            if (sendNs > m_intervalSendMaxNs)
                m_intervalSendMaxNs = sendNs;

            const uint64_t sendDoneSteadyNs = tcpSteadyNowNs();
            // Argus sensor SOF metadata and CNTVCT_EL0 use the same Tegra TSC
            // domain. This endpoint is sampled after the complete framed
            // packet has been written to the TCP socket.
            const uint64_t sendDoneTscNs = readTscNowNs();
            if (packet.timing.captureStartSteadyNs != 0 && sendDoneSteadyNs >= packet.timing.captureStartSteadyNs)
            {
                const uint64_t captureToSentNs = sendDoneSteadyNs - packet.timing.captureStartSteadyNs;
                ++m_intervalLatencySamples;
                m_acquireSpreadSumNs += packet.timing.acquireSpreadNs;
                if (packet.timing.acquireSpreadNs > m_acquireSpreadMaxNs)
                    m_acquireSpreadMaxNs = packet.timing.acquireSpreadNs;
                if (packet.timing.captureToEncodeStartNs != 0)
                {
                    ++m_intervalEncodeStartSamples;
                    m_captureToEncodeStartSumNs += packet.timing.captureToEncodeStartNs;
                    if (packet.timing.captureToEncodeStartNs > m_captureToEncodeStartMaxNs)
                    {
                        m_captureToEncodeStartMaxNs = packet.timing.captureToEncodeStartNs;
                    }
                }
                m_captureToProcessDoneSumNs += packet.timing.captureToProcessDoneNs;
                if (packet.timing.captureToProcessDoneNs > m_captureToProcessDoneMaxNs)
                {
                    m_captureToProcessDoneMaxNs = packet.timing.captureToProcessDoneNs;
                }
                m_captureToSentSumNs += captureToSentNs;
                if (captureToSentNs > m_captureToSentMaxNs)
                    m_captureToSentMaxNs = captureToSentNs;
            }

            if (m_settings.kpi && m_settings.logIntervalSeconds > 0)
            {
                recordStreamLatency(packet.timing, sendDoneSteadyNs);
                recordSofToSentLatency(packet.timing, sendDoneTscNs);
            }
            if (!writeWireDump(packet.bytes.data(), packet.bytes.size()))
            {
                m_error = true;
                break;
            }

            ++m_totalPackets;
            ++m_intervalPackets;
            if (packet.bytes.size() >= 4)
                m_intervalBytes += packet.bytes.size() - 4;

            if (m_settings.kpi && m_settings.logIntervalSeconds > 0)
            {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - m_statsStart).count();
                if (elapsed >= static_cast<double>(m_settings.logIntervalSeconds))
                {
                    const double fpsActual = static_cast<double>(m_intervalPackets) / elapsed;
                    const double mbps = static_cast<double>(m_intervalBytes) * 8.0 / elapsed / 1.0e6;
                    uint64_t enqueueWaitSamples = 0;
                    uint64_t enqueueWaitSumNs = 0;
                    uint64_t enqueueWaitMaxNs = 0;
                    {
                        std::lock_guard<std::mutex> metricsLock(m_metricsMutex);
                        enqueueWaitSamples = m_enqueueWaitSamples;
                        enqueueWaitSumNs = m_enqueueWaitSumNs;
                        enqueueWaitMaxNs = m_enqueueWaitMaxNs;
                        m_enqueueWaitSamples = 0;
                        m_enqueueWaitSumNs = 0;
                        m_enqueueWaitMaxNs = 0;
                    }
                    const double enqueueWaitAvgUs =
                        enqueueWaitSamples > 0 ? static_cast<double>(enqueueWaitSumNs) / enqueueWaitSamples / 1000.0 :
                                                 0.0;
                    const double sendAvgUs = m_intervalSendSamples > 0 ? static_cast<double>(m_intervalSendSumNs) /
                                                                             m_intervalSendSamples / 1000.0 :
                                                                         0.0;
                    const double captureToEncodeStartAvgMs =
                        m_intervalEncodeStartSamples > 0 ?
                            static_cast<double>(m_captureToEncodeStartSumNs) / m_intervalEncodeStartSamples / 1000000.0 :
                            0.0;
                    const double acquireSpreadAvgMs =
                        m_intervalLatencySamples > 0 ?
                            static_cast<double>(m_acquireSpreadSumNs) / m_intervalLatencySamples / 1000000.0 :
                            0.0;
                    const double captureToProcessDoneAvgMs =
                        m_intervalLatencySamples > 0 ?
                            static_cast<double>(m_captureToProcessDoneSumNs) / m_intervalLatencySamples / 1000000.0 :
                            0.0;
                    const double captureToSentAvgMs =
                        m_intervalLatencySamples > 0 ?
                            static_cast<double>(m_captureToSentSumNs) / m_intervalLatencySamples / 1000000.0 :
                            0.0;
                    if (verboseLoggingEnabled())
                    {
                        if (m_label.empty())
                            std::printf("argus_sender[tcp-hevc]: ");
                        else
                            std::printf("argus_sender[tcp-hevc:%s]: ", m_label.c_str());
                        std::printf(
                            "%.2f FPS, %.2f Mbps, "
                            "tcp_enqueue_wait_us(avg/max)=%.3f/%.3f, "
                            "tcp_send_us(avg/max)=%.3f/%.3f, "
                            "acquire_spread_ms(avg/max)=%.3f/%.3f, "
                            "capture_to_encode_start_ms(avg/max)=%.3f/%.3f, "
                            "capture_to_encode_done_ms(avg/max)=%.3f/%.3f, "
                            "capture_to_tcp_sent_ms(avg/max)=%.3f/%.3f, "
                            "total=%llu\n",
                            fpsActual, mbps, enqueueWaitAvgUs, static_cast<double>(enqueueWaitMaxNs) / 1000.0,
                            sendAvgUs, static_cast<double>(m_intervalSendMaxNs) / 1000.0, acquireSpreadAvgMs,
                            static_cast<double>(m_acquireSpreadMaxNs) / 1000000.0, captureToEncodeStartAvgMs,
                            static_cast<double>(m_captureToEncodeStartMaxNs) / 1000000.0, captureToProcessDoneAvgMs,
                            static_cast<double>(m_captureToProcessDoneMaxNs) / 1000000.0, captureToSentAvgMs,
                            static_cast<double>(m_captureToSentMaxNs) / 1000000.0,
                            static_cast<unsigned long long>(m_totalPackets));
                    }
                    logAndResetStreamLatency();
                    logAndResetSofToSentLatency();
                    m_intervalPackets = 0;
                    m_intervalBytes = 0;
                    m_intervalSendSamples = 0;
                    m_intervalSendSumNs = 0;
                    m_intervalSendMaxNs = 0;
                    m_intervalLatencySamples = 0;
                    m_intervalEncodeStartSamples = 0;
                    m_acquireSpreadSumNs = 0;
                    m_acquireSpreadMaxNs = 0;
                    m_captureToEncodeStartSumNs = 0;
                    m_captureToEncodeStartMaxNs = 0;
                    m_captureToProcessDoneSumNs = 0;
                    m_captureToProcessDoneMaxNs = 0;
                    m_captureToSentSumNs = 0;
                    m_captureToSentMaxNs = 0;
                    m_statsStart = now;
                }
            }
        }
    }

    int* m_fd;
    const TcpPacketSenderSettings& m_settings;
    const uint16_t m_port;
    const std::string m_label;
    const size_t m_maxQueuedPackets;
    const bool m_appendDump;
    int m_dumpFd;
    uint64_t m_dumpBytes;
    std::thread m_thread;
    std::mutex m_mutex;
    std::mutex m_metricsMutex;
    std::condition_variable m_packetAvailable;
    std::condition_variable m_spaceAvailable;
    std::deque<QueuedPacket> m_queue;
    std::atomic<bool> m_accepting;
    std::atomic<bool> m_abort;
    std::atomic<uint64_t> m_drainDeadlineSteadyNs;
    std::atomic<bool> m_error;
    std::atomic<bool> m_disconnected;
    uint64_t m_totalPackets;
    uint64_t m_intervalPackets;
    uint64_t m_intervalBytes;
    uint64_t m_enqueueWaitSamples;
    uint64_t m_enqueueWaitSumNs;
    uint64_t m_enqueueWaitMaxNs;
    uint64_t m_intervalSendSamples;
    uint64_t m_intervalSendSumNs;
    uint64_t m_intervalSendMaxNs;
    uint64_t m_intervalLatencySamples;
    std::vector<StreamLatencyStats> m_streamLatencyStats;
    std::vector<LatencyAccumulator> m_sofToSentStats;
    uint64_t m_intervalEncodeStartSamples;
    uint64_t m_acquireSpreadSumNs;
    uint64_t m_acquireSpreadMaxNs;
    uint64_t m_captureToEncodeStartSumNs;
    uint64_t m_captureToEncodeStartMaxNs;
    uint64_t m_captureToProcessDoneSumNs;
    uint64_t m_captureToProcessDoneMaxNs;
    uint64_t m_captureToSentSumNs;
    uint64_t m_captureToSentMaxNs;
    std::chrono::steady_clock::time_point m_statsStart;
};


TcpPacketSender::TcpPacketSender(int* fd,
                                 const TcpPacketSenderSettings& settings,
                                 uint32_t maxQueuedPackets,
                                 uint16_t port,
                                 const char* label,
                                 bool appendDump)
    : m_impl(new Impl(fd, settings, maxQueuedPackets, port, label, appendDump))
{
}

TcpPacketSender::~TcpPacketSender()
{
}

bool TcpPacketSender::start()
{
    return m_impl->start();
}

bool TcpPacketSender::enqueueFramed(std::vector<uint8_t>&& packet)
{
    return m_impl->enqueueFramed(std::move(packet), PacketTiming());
}

bool TcpPacketSender::enqueueFramedWithTiming(std::vector<uint8_t>&& packet, const PacketTiming& timing)
{
    return m_impl->enqueueFramed(std::move(packet), timing);
}

bool TcpPacketSender::hasError() const
{
    return m_impl->hasError();
}

bool TcpPacketSender::disconnected() const
{
    return m_impl->disconnected();
}

void TcpPacketSender::shutdown(bool drain)
{
    m_impl->shutdown(drain);
}

} // namespace argus_sender
