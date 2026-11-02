// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace
{

// Extracts device pointer + row pitch from an object exposing
// __cuda_array_interface__. Expects HxWx4 RGBA8 matching the configured
// (width, height).
struct CudaBuf
{
    std::uintptr_t ptr;
    std::size_t row_pitch_bytes;
};

inline CudaBuf extract_cuda_buffer(const py::object& obj, std::uint32_t expected_width, std::uint32_t expected_height)
{
    if (!py::hasattr(obj, "__cuda_array_interface__"))
    {
        throw std::runtime_error("expected object exposing __cuda_array_interface__");
    }
    py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
    if (!iface.contains("shape") || !iface.contains("data"))
    {
        throw std::runtime_error("__cuda_array_interface__ missing required keys (shape/data)");
    }

    if (!iface.contains("typestr") || iface["typestr"].cast<std::string>() != "|u1")
        throw std::runtime_error("output buffer must have uint8 dtype (|u1)");
    py::tuple output_data = iface["data"].cast<py::tuple>();
    if (output_data.size() != 2 || output_data[1].cast<bool>())
        throw std::runtime_error("output buffer must be writable");
    if (output_data[0].cast<std::uintptr_t>() == 0)
        throw std::runtime_error("output buffer pointer is null");

    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 3)
    {
        throw std::runtime_error("expected 3D array (H, W, 4); got rank " + std::to_string(shape.size()));
    }
    const auto h = shape[0].cast<std::uint32_t>();
    const auto w = shape[1].cast<std::uint32_t>();
    const auto c = shape[2].cast<std::size_t>();
    if (c != 4)
    {
        throw std::runtime_error("expected 4-channel RGBA; got " + std::to_string(c));
    }
    if (h != expected_height || w != expected_width)
    {
        throw std::runtime_error("frame size " + std::to_string(w) + "x" + std::to_string(h) +
                                 " does not match configured " + std::to_string(expected_width) + "x" +
                                 std::to_string(expected_height));
    }

    py::tuple data = iface["data"].cast<py::tuple>();
    const auto ptr = data[0].cast<std::uintptr_t>();

    // Default to tightly packed RGBA8.
    std::size_t row_pitch = static_cast<std::size_t>(w) * 4u;
    if (iface.contains("strides") && !iface["strides"].is_none())
    {
        py::tuple strides = iface["strides"].cast<py::tuple>();
        if (strides.size() != 3)
        {
            throw std::runtime_error("strides rank does not match shape rank");
        }
        const auto row_stride = strides[0].cast<std::ptrdiff_t>();
        const auto pixel_stride = strides[1].cast<std::ptrdiff_t>();
        const auto chan_stride = strides[2].cast<std::ptrdiff_t>();
        if (pixel_stride != 4 || chan_stride != 1)
        {
            throw std::runtime_error("non-contiguous-per-pixel RGBA layout not supported");
        }
        if (row_stride < static_cast<std::ptrdiff_t>(w * 4u))
        {
            throw std::runtime_error("row stride smaller than W*4");
        }
        row_pitch = static_cast<std::size_t>(row_stride);
    }

    return CudaBuf{ ptr, row_pitch };
}

inline CudaBuf extract_nv12_buffer(const py::object& obj, std::uint32_t expected_width, std::uint32_t expected_height)
{
    if (!py::hasattr(obj, "__cuda_array_interface__"))
    {
        throw std::runtime_error("expected object exposing __cuda_array_interface__");
    }
    py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
    if (!iface.contains("shape") || !iface.contains("data"))
    {
        throw std::runtime_error("__cuda_array_interface__ missing required keys (shape/data)");
    }

    if (!iface.contains("typestr") || iface["typestr"].cast<std::string>() != "|u1")
        throw std::runtime_error("output buffer must have uint8 dtype (|u1)");
    py::tuple output_data = iface["data"].cast<py::tuple>();
    if (output_data.size() != 2 || output_data[1].cast<bool>())
        throw std::runtime_error("output buffer must be writable");
    if (output_data[0].cast<std::uintptr_t>() == 0)
        throw std::runtime_error("output buffer pointer is null");

    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 2)
    {
        throw std::runtime_error("expected 2D NV12 array ((H*3)/2, W); got rank " + std::to_string(shape.size()));
    }
    const auto h = shape[0].cast<std::uint32_t>();
    const auto w = shape[1].cast<std::uint32_t>();
    const auto expected_rows = expected_height * 3u / 2u;
    if (h != expected_rows || w != expected_width)
    {
        throw std::runtime_error("NV12 frame size " + std::to_string(w) + "x" + std::to_string(h) +
                                 " does not match configured " + std::to_string(expected_width) + "x" +
                                 std::to_string(expected_rows));
    }

    py::tuple data = iface["data"].cast<py::tuple>();
    const auto ptr = data[0].cast<std::uintptr_t>();

    std::size_t row_pitch = expected_width;
    if (iface.contains("strides") && !iface["strides"].is_none())
    {
        py::tuple strides = iface["strides"].cast<py::tuple>();
        if (strides.size() != 2)
        {
            throw std::runtime_error("NV12 strides rank does not match shape rank");
        }
        const auto row_stride = strides[0].cast<std::ptrdiff_t>();
        const auto col_stride = strides[1].cast<std::ptrdiff_t>();
        if (col_stride != 1)
        {
            throw std::runtime_error("non-contiguous NV12 columns are not supported");
        }
        if (row_stride < static_cast<std::ptrdiff_t>(expected_width))
        {
            throw std::runtime_error("NV12 row stride smaller than width");
        }
        row_pitch = static_cast<std::size_t>(row_stride);
    }

    return CudaBuf{ ptr, row_pitch };
}

} // namespace
