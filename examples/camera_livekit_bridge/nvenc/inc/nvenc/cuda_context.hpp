// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace camera_bridge
{

/// Throws std::runtime_error naming `what` when a CUDA driver call failed.
void check_cu(CUresult result, const char* what);

/**
 * @brief The device's primary CUDA context, shared by decode, the GPU operation and encode.
 *
 * The runtime API (kernels, streams) and the driver API (NVDEC, NVENC) both use the primary context, so surfaces
 * are visible to all of them. Every thread that touches CUDA calls make_current() once.
 */
class CudaContext
{
public:
    explicit CudaContext(int device_index = 0);
    ~CudaContext();

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    void make_current() const;
    /// For cleanup paths that must not throw: the result is ignored.
    void make_current_noexcept() const noexcept;
    CUcontext handle() const
    {
        return m_context;
    }
    const std::string& device_name() const
    {
        return m_name;
    }

private:
    CUdevice m_device = 0;
    CUcontext m_context = nullptr;
    std::string m_name;
};

} // namespace camera_bridge
