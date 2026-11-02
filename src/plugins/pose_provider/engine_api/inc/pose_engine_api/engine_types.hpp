// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Value types shared by the pose engine records and the resolved setup.
#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace pose_provider
{

/** @brief Identifies one camera stream, independent of its position within an image group. */
struct StreamId
{
    uint32_t value = 0;

    friend auto operator<=>(const StreamId&, const StreamId&) = default;
};

enum class InputMode
{
    Stereo, //!< Two cameras.
    StereoImu, //!< Two cameras and one IMU.
};

/** @brief Where prepared input storage lives. This names a memory location, not where the engine computes. */
enum class MemoryKind
{
    Host,
};

/** @brief Canonical pixel formats. Row strides are in bytes; see canonical_size_bytes(). */
enum class PixelFormat
{
    Gray8, //!< One intensity byte per pixel.
    BGR888, //!< Packed B, G, R bytes per pixel.
    RGBA8, //!< Packed R, G, B, A bytes per pixel.
    NV12, //!< Y plane, then interleaved UV at offset stride * height with the same stride; even dimensions.
};

/** @brief Pinhole distortion models. Coefficients are stored in the order listed. */
enum class DistortionModel
{
    None, //!< No coefficients.
    RadTan5, //!< k1, k2, p1, p2, k3
    Rational8, //!< RadTan5, then k4, k5, k6
    ThinPrism12, //!< Rational8, then s1, s2, s3, s4
    Tilted14, //!< ThinPrism12, then tau_x, tau_y
};

constexpr std::size_t coefficient_count(DistortionModel model)
{
    switch (model)
    {
    case DistortionModel::None:
        return 0;
    case DistortionModel::RadTan5:
        return 5;
    case DistortionModel::Rational8:
        return 8;
    case DistortionModel::ThinPrism12:
        return 12;
    case DistortionModel::Tilted14:
        return 14;
    }
    return 0;
}

/**
 * @brief Rigid transform T_A_B, which maps coordinates in frame B into frame A.
 *
 * Right-handed; translation in meters; rotation as a unit Hamilton quaternion stored x, y, z, w.
 */
struct Transform
{
    std::array<double, 3> translation_m{ 0.0, 0.0, 0.0 };
    std::array<double, 4> rotation_xyzw{ 0.0, 0.0, 0.0, 1.0 };
};

/** @brief Dimensions, pixel format and row stride of a prepared image. */
struct ImageLayout
{
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::Gray8;
    uint32_t stride_bytes = 0;

    friend bool operator==(const ImageLayout&, const ImageLayout&) = default;
};

/**
 * @brief Minimum storage for a canonical layout, or nullopt if the layout is invalid.
 *
 * Invalid means a zero dimension, a stride below the packed row size, odd NV12 dimensions or a size that
 * overflows.
 */
inline std::optional<std::size_t> canonical_size_bytes(const ImageLayout& layout)
{
    if (layout.width == 0 || layout.height == 0)
    {
        return std::nullopt;
    }
    uint64_t bytes_per_pixel = 1;
    uint64_t rows = layout.height;
    switch (layout.format)
    {
    case PixelFormat::Gray8:
        break;
    case PixelFormat::BGR888:
        bytes_per_pixel = 3;
        break;
    case PixelFormat::RGBA8:
        bytes_per_pixel = 4;
        break;
    case PixelFormat::NV12:
        if (layout.width % 2 != 0 || layout.height % 2 != 0)
        {
            return std::nullopt;
        }
        rows += layout.height / 2;
        break;
    }
    if (layout.stride_bytes < layout.width * bytes_per_pixel)
    {
        return std::nullopt;
    }
    if (layout.stride_bytes > std::numeric_limits<std::size_t>::max() / rows)
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(layout.stride_bytes * rows);
}

} // namespace pose_provider
