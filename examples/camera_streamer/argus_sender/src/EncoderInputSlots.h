/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_ENCODER_INPUT_SLOTS_H
#define ARGUS_SENDER_ENCODER_INPUT_SLOTS_H

#include "CaptureFrame.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

namespace argus_sender
{

// The input slots of an encoder that reads captured frames in place. An occupied slot holds the encoder's wrapper of
// the frame's DMA-BUF and the frame's lease, so the backend gets the buffer back only once the encoder has dequeued
// it. `Wrapper` is a template parameter so the bookkeeping builds without the encoder's SDK.
template <typename Wrapper>
class EncoderInputSlots
{
public:
    explicit EncoderInputSlots(uint32_t count) : m_slots(count)
    {
        for (uint32_t index = 0; index < count; ++index)
            m_free.push_back(index);
    }

    bool hasFree() const
    {
        return !m_free.empty();
    }
    // The slot the next occupy() fills. Requires hasFree().
    uint32_t nextFree() const
    {
        return m_free.front();
    }
    uint32_t inFlight() const
    {
        return m_inFlight;
    }

    // The encoder accepted `wrapper` at nextFree(); hold it with `frame` until complete().
    void occupy(std::unique_ptr<Wrapper> wrapper, CaptureFrame frame)
    {
        Slot& slot = m_slots[m_free.front()];
        m_free.pop_front();
        slot.wrapper = std::move(wrapper);
        slot.frame = std::move(frame);
        slot.occupied = true;
        ++m_inFlight;
    }

    // The encoder dequeued `index`: return the buffer to the backend, then drop the wrapper. False for an index that
    // is not in flight.
    bool complete(uint32_t index)
    {
        if (index >= m_slots.size() || !m_slots[index].occupied)
            return false;
        clear(&m_slots[index]);
        m_free.push_back(index);
        --m_inFlight;
        return true;
    }

    // Drops every lease and wrapper. Call only once the encoder can no longer read an input, and before the backend
    // stops.
    void releaseAll()
    {
        m_free.clear();
        for (uint32_t index = 0; index < m_slots.size(); ++index)
        {
            clear(&m_slots[index]);
            m_free.push_back(index);
        }
        m_inFlight = 0;
    }

private:
    // Members destroy in reverse order, so a destroyed slot also drops the lease before the wrapper.
    struct Slot
    {
        std::unique_ptr<Wrapper> wrapper;
        CaptureFrame frame;
        bool occupied = false;
    };

    static void clear(Slot* slot)
    {
        slot->frame.lease.reset();
        slot->wrapper.reset();
        slot->occupied = false;
    }

    std::vector<Slot> m_slots;
    std::deque<uint32_t> m_free;
    uint32_t m_inFlight = 0;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_ENCODER_INPUT_SLOTS_H
