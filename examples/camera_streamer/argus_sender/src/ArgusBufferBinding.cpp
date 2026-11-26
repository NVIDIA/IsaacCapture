/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArgusBufferBinding.h"

#include <memory>

namespace argus_sender
{

ArgusBufferBinding::ArgusBufferBinding(Argus::IBufferOutputStream* stream, Argus::Buffer* buffer)
    : m_stream(stream), m_buffer(buffer), m_acquired(true)
{
}

bool ArgusBufferBinding::markAcquired(Argus::Buffer* buffer)
{
    if (buffer != m_buffer)
        return false;
    bool expected = false;
    return m_acquired.compare_exchange_strong(expected, true);
}

bool ArgusBufferBinding::release()
{
    bool expected = true;
    if (!m_acquired.compare_exchange_strong(expected, false))
        return false;
    if (m_stream->releaseBuffer(m_buffer) != Argus::STATUS_OK)
    {
        m_acquired.store(true);
        return false;
    }
    return true;
}

Argus::Buffer* ArgusBufferBinding::argusBuffer() const
{
    return m_buffer;
}

bool bindArgusBuffer(DmaBuffer* dmaBuffer, Argus::IBufferOutputStream* stream, Argus::Buffer* buffer)
{
    if (!dmaBuffer || !stream || !buffer || dmaBuffer->hasOwner())
        return false;
    dmaBuffer->setOwner(std::unique_ptr<DmaBufferOwner>(new ArgusBufferBinding(stream, buffer)));
    return true;
}

bool markArgusAcquired(DmaBuffer* dmaBuffer, Argus::Buffer* buffer)
{
    ArgusBufferBinding* binding = dmaBuffer ? dynamic_cast<ArgusBufferBinding*>(dmaBuffer->owner()) : NULL;
    return binding && binding->markAcquired(buffer);
}

Argus::Buffer* argusBufferOf(const DmaBuffer* dmaBuffer)
{
    const ArgusBufferBinding* binding = dmaBuffer ? dynamic_cast<const ArgusBufferBinding*>(dmaBuffer->owner()) : NULL;
    return binding ? binding->argusBuffer() : NULL;
}

} // namespace argus_sender
