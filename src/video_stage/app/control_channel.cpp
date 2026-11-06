// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "control_channel.hpp"

#include "pipeline.hpp"

#include <video_stage/clock_sync.hpp>

#include <chrono>
#include <vector>

namespace video_stage
{

namespace cs = plugins::sensing;

ControlChannel::ControlChannel(TcpReceiver& link, uint64_t first_epoch) : m_link(link), m_epoch(first_epoch)
{
    m_reader = std::thread(&ControlChannel::read_loop, this);
    m_requester = std::thread(&ControlChannel::request_loop, this);
}

ControlChannel::~ControlChannel()
{
    {
        std::lock_guard<std::mutex> lock(m_wait_mutex);
        m_stop = true;
    }
    m_wait.notify_all();
    m_link.shutdown_session(); // unblocks the reader
    if (m_requester.joinable())
        m_requester.join();
    if (m_reader.joinable())
        m_reader.join();
}

std::optional<UpstreamMapping> ControlChannel::mapping() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto mapping = m_estimator.mapping(monotonic_ns());
    if (!mapping)
        return std::nullopt;
    return UpstreamMapping{ *mapping, m_epoch };
}

void ControlChannel::request_loop()
{
    uint32_t sequence = 0;
    while (true)
    {
        cs::ClockSyncRequest request;
        request.sequence = ++sequence;
        request.t1 = monotonic_ns();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_outstanding[m_next_outstanding] = { request.sequence, request.t1, false };
            m_next_outstanding = (m_next_outstanding + 1) % kOutstanding;
        }
        if (!m_link.send_control(cs::build_clock_sync_request(request).data(), cs::kClockSyncRequestSize))
            return;
        std::unique_lock<std::mutex> lock(m_wait_mutex);
        if (m_wait.wait_for(lock, std::chrono::seconds(1), [this] { return m_stop; }))
            return;
    }
}

void ControlChannel::read_loop()
{
    while (true)
    {
        uint8_t first = 0;
        if (!m_link.recv_control(&first, 1))
            return;
        const int64_t t4 = monotonic_ns(); // stamped before anything else runs
        if (first != 'R')
            continue; // the Orin sends nothing else; ignore rather than guess
        std::vector<uint8_t> reply(cs::kClockSyncReplySize);
        reply[0] = first;
        if (!m_link.recv_control(reply.data() + 1, reply.size() - 1))
            return;
        const auto parsed = cs::parse_clock_sync_reply(reply);
        if (!parsed)
            continue;
        const Exchange exchange{ parsed->t1, parsed->t2, parsed->t3, t4 };
        std::lock_guard<std::mutex> lock(m_mutex);
        Outstanding* match = nullptr;
        for (Outstanding& o : m_outstanding)
        {
            if (!o.used && o.sequence == parsed->sequence && o.t1 == parsed->t1)
                match = &o;
        }
        if (match == nullptr)
        {
            ++m_rejected; // not the answer to a request of this session
            continue;
        }
        match->used = true;
        if (!m_estimator.add(exchange).empty())
        {
            ++m_rejected;
            continue;
        }
        ++m_exchanges;
        if (m_estimator.discontinuity())
        {
            ++m_discontinuities;
            ++m_epoch;
            m_estimator.clear_discontinuity();
        }
    }
}

} // namespace video_stage
