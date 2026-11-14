// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unique_fd.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace core::camera
{

/**
 * @brief A pool of equally sized slots in one block of shared host memory.
 *
 * The process that creates the pool writes the slots. Any process that is sent the pool's
 * descriptor can map it, but only for reading: the memory is sealed against new writable
 * mappings, writes and resizing.
 */
class HostPool
{
public:
    /**
     * @param name Shows up in /proc for debugging; it does not have to be unique.
     * @throws std::invalid_argument for an empty pool.
     * @throws std::system_error if the memory cannot be allocated or sealed.
     */
    HostPool(std::uint32_t slot_count, std::uint64_t slot_bytes, const std::string& name);
    ~HostPool();

    HostPool(const HostPool&) = delete;
    HostPool& operator=(const HostPool&) = delete;

    std::uint32_t slot_count() const
    {
        return m_slot_count;
    }

    std::uint64_t slot_bytes() const
    {
        return m_slot_bytes;
    }

    /**
     * @brief The writable memory of one slot.
     * @throws std::out_of_range for a slot the pool does not have.
     */
    std::span<std::byte> slot(std::uint32_t index);

    /**
     * @brief The pool's descriptor, to send to another process. The pool keeps owning it.
     */
    int descriptor() const
    {
        return m_fd.get();
    }

private:
    UniqueFd m_fd;
    std::byte* m_memory = nullptr;
    std::uint32_t m_slot_count = 0;
    std::uint64_t m_slot_bytes = 0;
    std::uint64_t m_slot_stride = 0; // from the start of one slot to the start of the next
};

/**
 * @brief Another process's read-only mapping of a HostPool, made from the descriptor it was sent.
 */
class HostPoolView
{
public:
    /**
     * @param descriptor Of the pool. It is closed once the pool is mapped.
     * @param slot_count, slot_bytes As the pool was created with.
     * @throws std::system_error if the descriptor cannot be mapped.
     * @throws std::invalid_argument if the descriptor holds less than the pool needs.
     */
    HostPoolView(UniqueFd descriptor, std::uint32_t slot_count, std::uint64_t slot_bytes);
    ~HostPoolView();

    HostPoolView(HostPoolView&& other) noexcept;
    HostPoolView& operator=(HostPoolView&& other) noexcept;
    HostPoolView(const HostPoolView&) = delete;
    HostPoolView& operator=(const HostPoolView&) = delete;

    std::uint32_t slot_count() const
    {
        return m_slot_count;
    }

    /**
     * @brief The memory of one slot.
     * @throws std::out_of_range for a slot the pool does not have.
     */
    std::span<const std::byte> slot(std::uint32_t index) const;

private:
    void unmap();

    const std::byte* m_memory = nullptr;
    std::uint32_t m_slot_count = 0;
    std::uint64_t m_slot_bytes = 0;
    std::uint64_t m_slot_stride = 0;
};

} // namespace core::camera
