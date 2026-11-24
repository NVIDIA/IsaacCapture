// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/nvenc/cuda_context.hpp"

#include <cuda_runtime_api.h>

namespace camera_bridge
{

void check_cu(CUresult result, const char* what)
{
    if (result == CUDA_SUCCESS)
        return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(what) + ": " + (message ? message : "unknown CUDA driver error"));
}

CudaContext::CudaContext(int device_index)
{
    check_cu(cuInit(0), "cuInit");
    check_cu(cuDeviceGet(&m_device, device_index), "cuDeviceGet");
    char name[128] = {};
    check_cu(cuDeviceGetName(name, sizeof(name), m_device), "cuDeviceGetName");
    m_name = name;
    check_cu(cuDevicePrimaryCtxRetain(&m_context, m_device), "cuDevicePrimaryCtxRetain");
    make_current();
    // Bind the runtime API to the same primary context.
    if (cudaSetDevice(device_index) != cudaSuccess)
        throw std::runtime_error("cudaSetDevice failed");
}

CudaContext::~CudaContext()
{
    if (m_context)
        cuDevicePrimaryCtxRelease(m_device);
}

void CudaContext::make_current_noexcept() const noexcept
{
    cuCtxSetCurrent(m_context);
}

void CudaContext::make_current() const
{
    check_cu(cuCtxSetCurrent(m_context), "cuCtxSetCurrent");
}

} // namespace camera_bridge
