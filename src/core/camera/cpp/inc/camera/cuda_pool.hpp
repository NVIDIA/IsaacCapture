// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "unique_fd.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace core::camera
{

/**
 * @brief Why this process cannot use CUDA memory: the library was built without the CUDA
 * Toolkit, the NVIDIA driver's library is not installed, or the driver finds no device.
 * @return The reason, or nothing when CUDA memory can be used.
 */
std::string cuda_unavailable();

/**
 * @brief The devices this process sees, by index, and the name each has in every process.
 * @throws std::runtime_error when CUDA cannot be used, or for an index past the last device.
 */
std::uint32_t cuda_device_count();
CudaDeviceUuid cuda_device_uuid(std::uint32_t index);

// A device's UUID as nvidia-smi prints it, for messages: "GPU-" and five groups of hex digits.
std::string cuda_device_name(const CudaDeviceUuid& device);

/**
 * @brief Copy between host memory and memory of a device, and wait until the copy is done.
 * For code that has no GPU pipeline of its own: a test, a tool, a camera whose SDK delivers
 * frames in host memory.
 * @throws std::runtime_error if the device is not there or the copy fails.
 */
void copy_to_cuda(const CudaDeviceUuid& device, CudaAddress to, std::span<const std::byte> from);
void copy_from_cuda(const CudaDeviceUuid& device, std::span<std::byte> to, CudaAddress from);

/**
 * @brief A pool of equally sized slots in one allocation of device memory.
 *
 * The process that creates the pool writes the slots. Any process that is sent the pool's
 * descriptor can map the allocation on the same device, and maps it read-only. The memory
 * lives until the pool and every mapping of it are gone, so a reader's view stays valid when
 * the process that made the pool ends.
 */
class CudaPool
{
public:
    /**
     * @throws std::invalid_argument for an empty pool.
     * @throws std::runtime_error when CUDA cannot be used, the device is not one of this
     * process's, or the memory cannot be allocated.
     */
    CudaPool(const CudaDeviceUuid& device, std::uint32_t slot_count, std::uint64_t slot_bytes);
    ~CudaPool();

    CudaPool(const CudaPool&) = delete;
    CudaPool& operator=(const CudaPool&) = delete;

    std::uint32_t slot_count() const;
    std::uint64_t slot_bytes() const;

    /**
     * @brief Where one slot starts on the device.
     * @throws std::out_of_range for a slot the pool does not have.
     */
    CudaAddress slot(std::uint32_t index) const;

    /**
     * @brief The pool's descriptor, to send to another process. The pool keeps owning it.
     */
    int descriptor() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Another process's read-only mapping of a CudaPool, made from the descriptor it was sent.
 */
class CudaPoolView
{
public:
    /**
     * @param descriptor Of the pool. It is closed once the pool is mapped.
     * @param device, slot_count, slot_bytes As the pool's stream describes it. Both sides
     * derive the size of the allocation from them, and a descriptor of another size is refused.
     * @throws std::invalid_argument for an empty pool.
     * @throws std::runtime_error when CUDA cannot be used, the device is not one of this
     * process's, or the descriptor cannot be mapped.
     */
    CudaPoolView(UniqueFd descriptor, const CudaDeviceUuid& device, std::uint32_t slot_count, std::uint64_t slot_bytes);
    ~CudaPoolView();

    CudaPoolView(CudaPoolView&& other) noexcept;
    CudaPoolView& operator=(CudaPoolView&& other) noexcept;
    CudaPoolView(const CudaPoolView&) = delete;
    CudaPoolView& operator=(const CudaPoolView&) = delete;

    std::uint32_t slot_count() const;

    // The index, in this process, of the device the memory is on.
    std::uint32_t device_index() const;

    /**
     * @brief Where one slot starts on the device.
     * @throws std::out_of_range for a slot the pool does not have.
     */
    CudaAddress slot(std::uint32_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace core::camera
