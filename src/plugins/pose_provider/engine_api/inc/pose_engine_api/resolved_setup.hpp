// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_records.hpp"
#include "engine_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pose_provider
{

/** @brief Pinhole calibration of the prepared pixel grid. The pixel origin is the upper-left pixel center. */
struct CameraCalibration
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    DistortionModel distortion = DistortionModel::None;
    std::vector<double> coefficients; //!< coefficient_count(distortion) values.
    Transform T_body_camera; //!< Camera optical frame into Body.
};

/** @brief Range of BT.709 NV12 samples. */
enum class YuvRange
{
    Limited,
    Full,
};

struct StreamSetup
{
    StreamId id;
    ImageLayout layout;
    YuvRange nv12_range = YuvRange::Limited; //!< Applies to NV12 only.
    double rate_hz = 0.0;
    CameraCalibration calibration;
};

/**
 * @brief Whether factory correction was applied upstream.
 *
 * The engine does its own online bias estimation in both cases.
 */
enum class ImuCorrection
{
    Uncorrected,
    FactoryCorrected,
};

struct ImuSetup
{
    Transform T_body_imu;
    double rate_hz = 0.0;
    ImuCorrection correction = ImuCorrection::Uncorrected;
    double accel_noise_density = 0.0; //!< m/s^2/sqrt(Hz)
    double gyro_noise_density = 0.0; //!< rad/s/sqrt(Hz)
    double accel_random_walk = 0.0; //!< m/s^3/sqrt(Hz)
    double gyro_random_walk = 0.0; //!< rad/s^2/sqrt(Hz)
    int64_t imu_offset_ns = 0; //!< t_camera = t_imu + imu_offset_ns. Fusion times include it; never apply it again.
};

/** @brief The run's immutable setup. Calibration and geometry here override anything in engine_settings. */
struct ResolvedPoseSetup
{
    InputMode input_mode = InputMode::Stereo;
    MemoryKind memory_kind = MemoryKind::Host; //!< The memory kind of all input in the run.
    std::string body_frame; //!< Names Body, the tracked rigid frame that the extrinsics map into.
    std::vector<StreamSetup> streams;
    std::optional<ImuSetup> imu; //!< Present in StereoImu only.
    std::string engine_settings; //!< Opaque to everything but the adapter.
    EngineIdentity engine_identity; //!< The engine this setup was resolved for.
};

} // namespace pose_provider
