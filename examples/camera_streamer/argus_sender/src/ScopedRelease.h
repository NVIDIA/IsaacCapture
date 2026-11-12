/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_SCOPED_RELEASE_H
#define ARGUS_SENDER_SCOPED_RELEASE_H

namespace argus_sender
{

// Owns one acquired object until ownership moves elsewhere: the object's Release() runs when the guard goes out of
// scope, on a return or an exception alike, unless dismiss() was called after the handoff completed. `Object` is a
// template parameter so the guard builds, and can be tested, without the SDK that supplies the real type.
template <typename Object>
class ScopedRelease
{
public:
    explicit ScopedRelease(Object* object) : m_object(object)
    {
    }

    ~ScopedRelease()
    {
        if (m_object)
            m_object->Release();
    }

    ScopedRelease(const ScopedRelease&) = delete;
    ScopedRelease& operator=(const ScopedRelease&) = delete;

    // Gives up ownership without releasing. Call it only once the new owner holds the object.
    void dismiss()
    {
        m_object = nullptr;
    }

private:
    Object* m_object;
};

} // namespace argus_sender

#endif
