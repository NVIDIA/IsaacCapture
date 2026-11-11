/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_ARGUS_BUFFER_BINDING_H
#define ARGUS_SENDER_ARGUS_BUFFER_BINDING_H

#include "HevcEncoder.h"

#include <Argus/Argus.h>
#include <Argus/BufferStream.h>

#include <atomic>

namespace argus_sender
{

// Ties a client-owned DMA-BUF to the Argus Buffer that writes it. A bound buffer is held by the application until
// release() returns it to the Argus stream.
class ArgusBufferBinding : public DmaBufferOwner
{
public:
    ArgusBufferBinding(Argus::IBufferOutputStream* stream, Argus::Buffer* buffer);

    // Marks the buffer acquired again after the stream hands it out; false if it was already held.
    bool markAcquired(Argus::Buffer* buffer);
    bool release() override;
    Argus::Buffer* argusBuffer() const;

private:
    Argus::IBufferOutputStream* m_stream;
    Argus::Buffer* m_buffer;
    std::atomic<bool> m_acquired;
};

// False if the buffer already has an owner or an argument is null.
bool bindArgusBuffer(DmaBuffer* dmaBuffer, Argus::IBufferOutputStream* stream, Argus::Buffer* buffer);
bool markArgusAcquired(DmaBuffer* dmaBuffer, Argus::Buffer* buffer);
// The Argus buffer a DMA-BUF is bound to, or NULL if it is not bound to one.
Argus::Buffer* argusBufferOf(const DmaBuffer* dmaBuffer);

} // namespace argus_sender

#endif // ARGUS_SENDER_ARGUS_BUFFER_BINDING_H
