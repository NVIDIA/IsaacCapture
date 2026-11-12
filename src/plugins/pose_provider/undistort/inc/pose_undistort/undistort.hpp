// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Undistortion of one camera to a distortion-free pinhole with the same optical frame and image size. The lens model
// is DistortionModel's, with the equations of OpenCV's calib3d documentation; tau_x and tau_y are in radians.
#pragma once

#include <pose_engine_api/resolved_setup.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pose_provider
{

/**
 * @brief The source position that each output pixel samples, row-major, width * height entries in each of x and y.
 *
 * The origin is the upper-left pixel center, as in cv::initUndistortRectifyMap's CV_32FC1 map pair.
 */
struct UndistortMap
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> x;
    std::vector<float> y;
};

/**
 * @brief The map from the delivered lens @p source to the distortion-free @p output: cv::initUndistortRectifyMap's
 * map with R = I. Reads only the pinhole and distortion fields.
 *
 * @throws std::invalid_argument for a zero dimension, an @p output with distortion, a focal length that is not
 * positive and finite, a principal point or coefficient that is not finite, an unknown model, or a coefficient count
 * other than coefficient_count() of its model.
 */
UndistortMap make_undistort_map(const CameraCalibration& source,
                                const CameraCalibration& output,
                                uint32_t width,
                                uint32_t height);

/**
 * @brief The applied calibration for the delivered lens @p source: distortion None, the source principal point and
 * T_body_camera, and the source focal lengths times the smallest factor, to a relative 1e-6, whose map is valid:
 * within [0, width - 1] x [0, height - 1] and not folded, so every triangle of adjacent output pixels keeps its
 * orientation.
 *
 * @throws std::invalid_argument as make_undistort_map() does, or if that factor is not in (1/16, 16].
 */
CameraCalibration widest_undistorted_calibration(const CameraCalibration& source, uint32_t width, uint32_t height);

/**
 * @brief Writes each output pixel as the bilinear sample of @p source at its map position, rounded to the nearest
 * level, with neighbors outside the source counted as 0: cv::remap with INTER_LINEAR and BORDER_CONSTANT 0, up to
 * OpenCV 4's rounding of positions to 1/32 px. Does not allocate or block.
 *
 * @p source is map.width x map.height Gray8 with rows @p source_stride_bytes >= map.width apart; @p output receives
 * the same size tightly packed and must not overlap @p source.
 */
void remap_gray8(const UndistortMap& map, const std::byte* source, uint32_t source_stride_bytes, std::byte* output) noexcept;

} // namespace pose_provider
