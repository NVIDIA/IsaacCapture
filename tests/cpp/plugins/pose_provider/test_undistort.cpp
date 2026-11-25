// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The undistortion library on synthetic lenses. Reference values were generated offline for these lenses; remap
// positions lie on a 1/32 px grid, where an implementation that rounds positions to 1/32 px is exact.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <pose_undistort/undistort.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace pose_provider;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 400;

CameraCalibration pinhole(double fx, double fy, double cx, double cy)
{
    CameraCalibration calibration;
    calibration.fx = fx;
    calibration.fy = fy;
    calibration.cx = cx;
    calibration.cy = cy;
    return calibration;
}

// Barrel, tilted, every term nonzero.
CameraCalibration wide_tilted_lens()
{
    CameraCalibration lens = pinhole(268.0, 266.5, 307.5, 205.5);
    lens.distortion = DistortionModel::Tilted14;
    lens.coefficients = { -0.28, 0.09,   4.0e-4,  -6.0e-4, -0.012, 0.05,  0.012,
                          0.002, 1.5e-3, -4.0e-4, -1.2e-3, 3.0e-4, 0.009, -0.004 };
    lens.T_body_camera.translation_m = { 0.031, -0.004, 0.002 };
    lens.T_body_camera.rotation_xyzw = { 0.0, 0.0, 0.6, 0.8 };
    return lens;
}

// Pincushion.
CameraCalibration mild_lens()
{
    CameraCalibration lens = pinhole(452.0, 450.5, 306.25, 184.25);
    lens.distortion = DistortionModel::RadTan5;
    lens.coefficients = { 0.045, -0.012, -3.0e-4, 5.0e-4, 0.004 };
    return lens;
}

bool within_image(const UndistortMap& map)
{
    for (std::size_t i = 0; i < map.x.size(); ++i)
    {
        if (!(map.x[i] >= 0 && map.x[i] <= kWidth - 1 && map.y[i] >= 0 && map.y[i] <= kHeight - 1))
        {
            return false;
        }
    }
    return true;
}

// Checks what does not depend on the lens and returns the focal length factor.
double widest_factor(const CameraCalibration& source)
{
    const CameraCalibration widest = widest_undistorted_calibration(source, kWidth, kHeight);
    CHECK(widest.distortion == DistortionModel::None);
    CHECK(widest.coefficients.empty());
    CHECK(widest.cx == source.cx);
    CHECK(widest.cy == source.cy);
    CHECK(widest.T_body_camera.translation_m == source.T_body_camera.translation_m);
    CHECK(widest.T_body_camera.rotation_xyzw == source.T_body_camera.rotation_xyzw);
    CHECK_THAT(widest.fy / widest.fx, WithinRel(source.fy / source.fx, 1e-12));
    CHECK(within_image(make_undistort_map(source, widest, kWidth, kHeight)));
    return widest.fx / source.fx;
}

// A ramp of level 8 x + 16 y in padded rows.
constexpr uint32_t kRampWidth = 16;
constexpr uint32_t kRampHeight = 8;
constexpr uint32_t kRampStride = 19;

std::vector<std::byte> ramp()
{
    std::vector<std::byte> bytes(kRampStride * kRampHeight, std::byte{ 255 });
    for (uint32_t y = 0; y < kRampHeight; ++y)
    {
        for (uint32_t x = 0; x < kRampWidth; ++x)
        {
            bytes[y * kRampStride + x] = static_cast<std::byte>(8 * x + 16 * y);
        }
    }
    return bytes;
}

std::vector<int> remapped(const UndistortMap& map)
{
    const std::vector<std::byte> source = ramp();
    std::vector<std::byte> output(kRampWidth * kRampHeight, std::byte{ 7 });
    remap_gray8(map, source.data(), kRampStride, output.data());
    std::vector<int> levels;
    for (const std::byte b : output)
    {
        levels.push_back(std::to_integer<int>(b));
    }
    return levels;
}

// An output pixel and the source position the reference map gives it.
struct ReferenceSample
{
    uint32_t u;
    uint32_t v;
    double x;
    double y;
};

// A lens pair and its reference samples.
struct ReferenceCase
{
    CameraCalibration source;
    CameraCalibration output;
    std::vector<ReferenceSample> samples;
};

// A source position near or beyond the image edge, and the level remap_gray8 must produce there.
struct BorderPosition
{
    float x;
    float y;
    int expected;
};

} // namespace

TEST_CASE("The undistortion map matches reference values", "[unit][pose_provider]")
{
    const ReferenceCase cases[] = {
        { wide_tilted_lens(),
          pinhole(248.0, 246.5, 307.5, 205.5),
          { { 0, 0, 116.5386, 77.7332 },
            { 639, 0, 503.5078, 83.8659 },
            { 0, 399, 111.5314, 328.7954 },
            { 639, 399, 508.7408, 323.0029 },
            { 320, 0, 318.4942, 26.4571 },
            { 320, 399, 318.8763, 379.9041 },
            { 0, 200, 87.2771, 201.4108 },
            { 639, 200, 533.7136, 201.6285 },
            { 173, 291, 179.6628, 286.7767 } } },
        // The output principal point differs from the source's.
        { mild_lens(),
          pinhole(470.0, 468.5, 298.5, 187.75),
          { { 0, 0, 13.0271, -0.3137 },
            { 639, 0, 642.8566, -1.3368 },
            { 0, 399, 12.7534, 391.9476 },
            { 639, 399, 643.1750, 393.0549 },
            { 320, 0, 327.1144, 2.3723 },
            { 320, 399, 327.1497, 389.0921 },
            { 0, 200, 14.7272, 196.1613 },
            { 639, 200, 640.9094, 196.2130 },
            { 173, 291, 185.0006, 283.9964 } } },
    };
    for (const ReferenceCase& c : cases)
    {
        const UndistortMap map = make_undistort_map(c.source, c.output, kWidth, kHeight);
        REQUIRE(map.width == kWidth);
        REQUIRE(map.height == kHeight);
        REQUIRE(map.x.size() == std::size_t{ kWidth } * kHeight);
        REQUIRE(map.y.size() == map.x.size());
        for (const ReferenceSample& s : c.samples)
        {
            CAPTURE(c.source.distortion, s.u, s.v);
            CHECK_THAT(map.x[s.v * kWidth + s.u], WithinAbs(s.x, 1e-3));
            CHECK_THAT(map.y[s.v * kWidth + s.u], WithinAbs(s.y, 1e-3));
        }
    }
}

TEST_CASE("The widest undistorted calibration has the smallest focal length factor with a valid map",
          "[unit][pose_provider]")
{
    SECTION("A barrel lens stops where its map would fold")
    {
        CHECK_THAT(widest_factor(wide_tilted_lens()), WithinRel(0.9274621, 1e-5));
    }
    SECTION("A pincushion lens stops at the image edge")
    {
        CHECK_THAT(widest_factor(mild_lens()), WithinRel(1.0288439, 1e-5));
    }
    SECTION("A principal point off the vertical center stops at the nearer top or bottom edge")
    {
        CameraCalibration lens = pinhole(330.0, 330.0, 319.5, 179.5);
        lens.distortion = DistortionModel::RadTan5;
        lens.coefficients = { -0.12, 0.0, 0.0, 0.0, 0.0 };
        CHECK_THAT(widest_factor(lens), WithinRel(0.9616041, 1e-5));
        lens.cy = 219.5;
        CHECK_THAT(widest_factor(lens), WithinRel(0.9616041, 1e-5));
    }
    SECTION("A radial turnover inside the image bounds the factor")
    {
        // r (1 + k1 r^2) turns over at r = 1 / sqrt(-3 k1); the corner ray reaches it at this factor.
        const double k1 = -0.65;
        CameraCalibration lens = pinhole(400.0, 400.0, 319.5, 199.5);
        lens.distortion = DistortionModel::RadTan5;
        lens.coefficients = { k1, 0.0, 0.0, 0.0, 0.0 };
        const double turnover = std::hypot(319.5, 199.5) / 400.0 * std::sqrt(-3 * k1);
        CHECK_THAT(widest_factor(lens), WithinRel(turnover, 0.01));
    }
}

TEST_CASE("Calibrations that the model cannot use are rejected", "[unit][pose_provider]")
{
    const CameraCalibration output = pinhole(400.0, 400.0, 320.0, 200.0);
    CameraCalibration short_coefficients = mild_lens();
    short_coefficients.coefficients.pop_back();
    CameraCalibration not_finite = mild_lens();
    not_finite.coefficients[0] = std::numeric_limits<double>::quiet_NaN();
    CameraCalibration zero_focal = mild_lens();
    zero_focal.fy = 0.0;
    CameraCalibration unknown_model = mild_lens();
    unknown_model.distortion = static_cast<DistortionModel>(99);
    unknown_model.coefficients.clear();
    CameraCalibration long_coefficients = mild_lens();
    long_coefficients.coefficients.push_back(0.0);
    CameraCalibration no_valid_factor = mild_lens();
    no_valid_factor.cx = -1.0;

    CHECK_THROWS_AS(make_undistort_map(mild_lens(), output, 0, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(make_undistort_map(short_coefficients, output, kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(make_undistort_map(not_finite, output, kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(make_undistort_map(zero_focal, output, kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(make_undistort_map(unknown_model, output, kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(make_undistort_map(mild_lens(), mild_lens(), kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(widest_undistorted_calibration(long_coefficients, kWidth, kHeight), std::invalid_argument);
    CHECK_THROWS_AS(widest_undistorted_calibration(no_valid_factor, kWidth, kHeight), std::invalid_argument);
}

TEST_CASE("remap_gray8 interpolates a padded source bilinearly and rounds to the nearest level", "[unit][pose_provider]")
{
    // 0.75 u + 0.25 + 3/32 and 0.75 v + 0.5 sample 8 x + 16 y at 6 u + 12 v + 10.75.
    UndistortMap map{ kRampWidth, kRampHeight, {}, {} };
    for (uint32_t v = 0; v < kRampHeight; ++v)
    {
        for (uint32_t u = 0; u < kRampWidth; ++u)
        {
            map.x.push_back(0.75f * u + 0.25f + 3.0f / 32);
            map.y.push_back(0.75f * v + 0.5f);
        }
    }
    const std::vector<int> output = remapped(map);
    for (uint32_t v = 0; v < kRampHeight; ++v)
    {
        for (uint32_t u = 0; u < kRampWidth; ++u)
        {
            CAPTURE(u, v);
            CHECK(output[v * kRampWidth + u] == static_cast<int>(6 * u + 12 * v + 11));
        }
    }
}

TEST_CASE("remap_gray8 counts neighbors outside the source as zero", "[unit][pose_provider]")
{
    const BorderPosition positions[] = {
        { 15.0f, 7.0f, 232 }, // The last pixel center.
        { -0.5f, 2.5f, 20 }, // Half of 40.
        { 15.5f, 3.0f, 84 }, // Half of 168.
        { 4.0f, -0.25f, 24 }, // Three quarters of 32.
        { -1.0f, 2.0f, 0 }, // No neighbor inside.
        { 15.03125f, 0.0f, 116 }, // 31/32 of 120.
    };
    // Every other entry samples its own pixel center.
    UndistortMap map{ kRampWidth, kRampHeight, {}, {} };
    for (uint32_t v = 0; v < kRampHeight; ++v)
    {
        for (uint32_t u = 0; u < kRampWidth; ++u)
        {
            map.x.push_back(static_cast<float>(u));
            map.y.push_back(static_cast<float>(v));
        }
    }
    for (std::size_t i = 0; i < std::size(positions); ++i)
    {
        map.x[i] = positions[i].x;
        map.y[i] = positions[i].y;
    }

    const std::vector<int> output = remapped(map);
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        CAPTURE(i);
        const int expected = i < std::size(positions) ? positions[i].expected :
                                                        static_cast<int>(8 * (i % kRampWidth) + 16 * (i / kRampWidth));
        CHECK(output[i] == expected);
    }
}
