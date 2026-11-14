// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_ipc/host_pool.hpp"

#include <sys/mman.h>
#include <sys/stat.h>

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

// Older C library headers lack this seal although the kernel has it.
#ifndef F_SEAL_FUTURE_WRITE
#    define F_SEAL_FUTURE_WRITE 0x0010
#endif

namespace core::camera
{

namespace
{

[[noreturn]] void throw_errno(const std::string& what)
{
    throw std::system_error(errno, std::generic_category(), what);
}

// Slots start on page boundaries. Both sides derive the layout from the slot size alone.
std::uint64_t slot_stride_for(std::uint64_t slot_bytes)
{
    const auto page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    return (slot_bytes + page - 1) / page * page;
}

std::uint64_t pool_bytes_for(std::uint32_t slot_count, std::uint64_t slot_stride)
{
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (slot_count == 0 || slot_stride == 0 || slot_stride > limit / slot_count)
    {
        throw std::invalid_argument(
            "a host pool needs at least one slot of at least one byte, and a size a file can have");
    }
    return slot_stride * slot_count;
}

} // namespace

HostPool::HostPool(std::uint32_t slot_count, std::uint64_t slot_bytes, const std::string& name)
    : m_slot_count(slot_count), m_slot_bytes(slot_bytes), m_slot_stride(slot_stride_for(slot_bytes))
{
    const std::uint64_t pool_bytes = pool_bytes_for(slot_count, m_slot_stride);
    m_fd.reset(::memfd_create(name.c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (!m_fd.valid())
    {
        throw_errno("host pool '" + name + "': memfd_create");
    }
    if (::ftruncate(m_fd.get(), static_cast<off_t>(pool_bytes)) != 0)
    {
        throw_errno("host pool '" + name + "': sizing the pool");
    }
    void* memory = ::mmap(nullptr, pool_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd.get(), 0);
    if (memory == MAP_FAILED)
    {
        throw_errno("host pool '" + name + "': mapping the pool");
    }
    // From here on nobody can resize the pool or map it writable; the mapping above stays.
    if (::fcntl(m_fd.get(), F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_FUTURE_WRITE | F_SEAL_SEAL) != 0)
    {
        const int error = errno;
        ::munmap(memory, pool_bytes);
        throw std::system_error(error, std::generic_category(), "host pool '" + name + "': sealing the pool");
    }
    m_memory = static_cast<std::byte*>(memory);
}

HostPool::~HostPool()
{
    ::munmap(m_memory, m_slot_stride * m_slot_count);
}

std::span<std::byte> HostPool::slot(std::uint32_t index)
{
    if (index >= m_slot_count)
    {
        throw std::out_of_range("host pool has no slot " + std::to_string(index));
    }
    return { m_memory + m_slot_stride * index, m_slot_bytes };
}

HostPoolView::HostPoolView(UniqueFd descriptor, std::uint32_t slot_count, std::uint64_t slot_bytes)
    : m_slot_count(slot_count), m_slot_bytes(slot_bytes), m_slot_stride(slot_stride_for(slot_bytes))
{
    const std::uint64_t pool_bytes = pool_bytes_for(slot_count, m_slot_stride);
    struct stat info = {};
    if (::fstat(descriptor.get(), &info) != 0)
    {
        throw_errno("host pool view: inspecting the pool");
    }
    // Mapping more than the descriptor holds would fault on first access instead of failing here.
    if (static_cast<std::uint64_t>(info.st_size) < pool_bytes)
    {
        throw std::invalid_argument("host pool view: the descriptor holds less than the pool needs");
    }
    void* memory = ::mmap(nullptr, pool_bytes, PROT_READ, MAP_SHARED, descriptor.get(), 0);
    if (memory == MAP_FAILED)
    {
        throw_errno("host pool view: mapping the pool");
    }
    m_memory = static_cast<const std::byte*>(memory);
}

HostPoolView::~HostPoolView()
{
    unmap();
}

HostPoolView::HostPoolView(HostPoolView&& other) noexcept
    : m_memory(std::exchange(other.m_memory, nullptr)),
      m_slot_count(std::exchange(other.m_slot_count, 0)),
      m_slot_bytes(other.m_slot_bytes),
      m_slot_stride(other.m_slot_stride)
{
}

HostPoolView& HostPoolView::operator=(HostPoolView&& other) noexcept
{
    if (this != &other)
    {
        unmap();
        m_memory = std::exchange(other.m_memory, nullptr);
        m_slot_count = std::exchange(other.m_slot_count, 0);
        m_slot_bytes = other.m_slot_bytes;
        m_slot_stride = other.m_slot_stride;
    }
    return *this;
}

std::span<const std::byte> HostPoolView::slot(std::uint32_t index) const
{
    if (index >= m_slot_count)
    {
        throw std::out_of_range("host pool view has no slot " + std::to_string(index));
    }
    return { m_memory + m_slot_stride * index, m_slot_bytes };
}

void HostPoolView::unmap()
{
    if (m_memory != nullptr)
    {
        ::munmap(const_cast<std::byte*>(m_memory), m_slot_stride * m_slot_count);
        m_memory = nullptr;
        m_slot_count = 0;
    }
}

} // namespace core::camera
