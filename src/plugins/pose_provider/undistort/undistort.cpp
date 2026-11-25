// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pose_undistort/undistort.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pose_provider
{

namespace
{

// The focal length factor is searched in (kMinFactor, kMaxFactor] to a relative kPrecision.
constexpr double kMinFactor = 1.0 / 16;
constexpr double kMaxFactor = 16.0;
constexpr double kPrecision = 1e-6;

using Matrix3 = std::array<double, 9>; // Row-major.

Matrix3 multiply(const Matrix3& a, const Matrix3& b)
{
    Matrix3 product{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int k = 0; k < 3; ++k)
            {
                product[row * 3 + col] += a[row * 3 + k] * b[k * 3 + col];
            }
        }
    }
    return product;
}

// Mtilt = [[R33, 0, -R13], [0, R33, -R23], [0, 0, 1]] * R, with R = Ry(tau_y) * Rx(tau_x).
Matrix3 tilt_projection(double tau_x, double tau_y)
{
    const double cos_x = std::cos(tau_x);
    const double sin_x = std::sin(tau_x);
    const double cos_y = std::cos(tau_y);
    const double sin_y = std::sin(tau_y);
    const Matrix3 rx{ 1, 0, 0, 0, cos_x, sin_x, 0, -sin_x, cos_x };
    const Matrix3 ry{ cos_y, 0, -sin_y, 0, 1, 0, sin_y, 0, cos_y };
    const Matrix3 r = multiply(ry, rx);
    return multiply({ r[8], 0, -r[2], 0, r[8], -r[5], 0, 0, 1 }, r);
}

// coefficient_count() cannot tell an unknown model from None.
bool known_model(DistortionModel model)
{
    switch (model)
    {
    case DistortionModel::None:
    case DistortionModel::RadTan5:
    case DistortionModel::Rational8:
    case DistortionModel::ThinPrism12:
    case DistortionModel::Tilted14:
        return true;
    }
    return false;
}

void check_calibration(const CameraCalibration& calibration, const std::string& role)
{
    const auto finite = [](double v) { return std::isfinite(v); };
    if (!(finite(calibration.fx) && finite(calibration.fy) && calibration.fx > 0 && calibration.fy > 0 &&
          finite(calibration.cx) && finite(calibration.cy)))
    {
        throw std::invalid_argument(role +
                                    " calibration needs positive finite focal lengths and a finite principal point");
    }
    if (!known_model(calibration.distortion))
    {
        throw std::invalid_argument(role + " calibration has an unknown distortion model");
    }
    if (calibration.coefficients.size() != coefficient_count(calibration.distortion) ||
        !std::all_of(calibration.coefficients.begin(), calibration.coefficients.end(), finite))
    {
        throw std::invalid_argument(role + " calibration needs one finite coefficient per term of its distortion model");
    }
}

void check_inputs(const CameraCalibration& source, const CameraCalibration& output, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0)
    {
        throw std::invalid_argument("undistortion needs a nonzero image size");
    }
    check_calibration(source, "source");
    check_calibration(output, "output");
    if (output.distortion != DistortionModel::None)
    {
        throw std::invalid_argument("undistortion output must have distortion None");
    }
}

// Fills the sized map from validated calibrations, in double; the map stores float.
void fill_map(const CameraCalibration& source, const CameraCalibration& output, UndistortMap& map)
{
    std::array<double, 14> d{}; // A shorter model's missing terms are zero.
    std::copy(source.coefficients.begin(), source.coefficients.end(), d.begin());
    const auto [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y] = d;
    const Matrix3 m = tilt_projection(tau_x, tau_y);

    std::size_t i = 0;
    for (uint32_t v = 0; v < map.height; ++v)
    {
        const double y = (v - output.cy) / output.fy;
        for (uint32_t u = 0; u < map.width; ++u, ++i)
        {
            const double x = (u - output.cx) / output.fx;
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6);
            const double xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4;
            const double yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4;
            const double a = m[0] * xd + m[1] * yd + m[2];
            const double b = m[3] * xd + m[4] * yd + m[5];
            const double c = m[6] * xd + m[7] * yd + m[8];
            map.x[i] = static_cast<float>(source.fx * a / c + source.cx);
            map.y[i] = static_cast<float>(source.fy * b / c + source.cy);
        }
    }
}

// Twice the signed area of triangle (o, a, b) in source pixels; positive for the output grid's own orientation.
double area(const UndistortMap& map, std::size_t o, std::size_t a, std::size_t b)
{
    const double ax = static_cast<double>(map.x[a]) - map.x[o];
    const double ay = static_cast<double>(map.y[a]) - map.y[o];
    const double bx = static_cast<double>(map.x[b]) - map.x[o];
    const double by = static_cast<double>(map.y[b]) - map.y[o];
    return ax * by - ay * bx;
}

// The header's validity rule, on the stored floats. The bounds are exact in float, and NaN fails them.
bool valid(const UndistortMap& map)
{
    const auto last_x = static_cast<float>(map.width - 1);
    const auto last_y = static_cast<float>(map.height - 1);
    for (std::size_t i = 0; i < map.x.size(); ++i)
    {
        if (!(map.x[i] >= 0.0f && map.x[i] <= last_x && map.y[i] >= 0.0f && map.y[i] <= last_y))
        {
            return false;
        }
    }
    for (uint32_t v = 0; v + 1 < map.height; ++v)
    {
        for (uint32_t u = 0; u + 1 < map.width; ++u)
        {
            // Cell p q over s t.
            const std::size_t p = std::size_t{ v } * map.width + u;
            const std::size_t q = p + 1;
            const std::size_t s = p + map.width;
            const std::size_t t = s + 1;
            if (!(area(map, p, q, s) > 0 && area(map, t, s, q) > 0))
            {
                return false;
            }
        }
    }
    return true;
}

float level(std::byte b)
{
    return static_cast<float>(std::to_integer<int>(b));
}

// The level at (x, y) of a source @p width x @p height, or 0 outside it.
float pixel(const std::byte* source, uint32_t stride_bytes, int width, int height, int x, int y)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return 0.0f;
    }
    return level(source[static_cast<std::size_t>(y) * stride_bytes + x]);
}

// Bilinear blend of p00, p10 (right), p01 (below) and p11 at fractions ax, ay.
float blend(float ax, float ay, float p00, float p10, float p01, float p11)
{
    const float top = p00 + ax * (p10 - p00);
    const float bottom = p01 + ax * (p11 - p01);
    return top + ay * (bottom - top);
}

// Sets @p output's focal lengths to @p factor times @p source's.
void scale_focal_lengths(const CameraCalibration& source, double factor, CameraCalibration& output)
{
    output.fx = source.fx * factor;
    output.fy = source.fy * factor;
}

// Whether the map at @p factor is valid; leaves @p output scaled to it and @p map filled for it.
bool valid_at(const CameraCalibration& source, double factor, CameraCalibration& output, UndistortMap& map)
{
    scale_focal_lengths(source, factor, output);
    fill_map(source, output, map);
    return valid(map);
}

UndistortMap sized_map(uint32_t width, uint32_t height)
{
    const std::size_t count = std::size_t{ width } * height;
    return UndistortMap{ width, height, std::vector<float>(count), std::vector<float>(count) };
}

} // namespace

UndistortMap make_undistort_map(const CameraCalibration& source,
                                const CameraCalibration& output,
                                uint32_t width,
                                uint32_t height)
{
    check_inputs(source, output, width, height);
    UndistortMap map = sized_map(width, height);
    fill_map(source, output, map);
    return map;
}

CameraCalibration widest_undistorted_calibration(const CameraCalibration& source, uint32_t width, uint32_t height)
{
    CameraCalibration output; // Copies the principal point and T_body_camera; scales the focal lengths.
    output.cx = source.cx;
    output.cy = source.cy;
    output.T_body_camera = source.T_body_camera;
    scale_focal_lengths(source, 1.0, output);
    check_inputs(source, output, width, height);

    // Defined over the map itself, so any map builder gives the same guarantee.
    UndistortMap map = sized_map(width, height);
    const auto fits = [&](double factor) { return valid_at(source, factor, output, map); };

    // Bracket the smallest valid factor between low, invalid, and high, valid, by halving or doubling from 1.
    double low = 1.0;
    double high = 1.0;
    if (fits(1.0))
    {
        do
        {
            high = low;
            low = high / 2;
            if (low < kMinFactor)
            {
                throw std::invalid_argument("the undistortion map stays valid below the searched focal length range");
            }
        } while (fits(low));
    }
    else
    {
        do
        {
            low = high;
            high = low * 2;
            if (high > kMaxFactor)
            {
                throw std::invalid_argument("no focal length in the searched range gives a valid undistortion map");
            }
        } while (!fits(high));
    }
    while (high - low > kPrecision * high)
    {
        const double middle = (low + high) / 2;
        (fits(middle) ? high : low) = middle;
    }
    scale_focal_lengths(source, high, output);
    return output;
}

void remap_gray8(const UndistortMap& map, const std::byte* source, uint32_t source_stride_bytes, std::byte* output) noexcept
{
    const auto width = static_cast<int>(map.width);
    const auto height = static_cast<int>(map.height);
    const auto last_x = static_cast<float>(width - 1);
    const auto last_y = static_cast<float>(height - 1);
    const auto at = [&](int x, int y) { return pixel(source, source_stride_bytes, width, height, x, y); };
    // Locals, because a std::byte store may alias the vectors' members, which g++ then reloads for every pixel.
    const float* xs = map.x.data();
    const float* ys = map.y.data();
    const std::size_t count = map.x.size();

    for (std::size_t i = 0; i < count; ++i)
    {
        const float x = xs[i];
        const float y = ys[i];
        // Comparisons are false for NaN, which gives 0; a position that passes either test converts to int safely.
        float value = 0.0f;
        if (x >= 0.0f && y >= 0.0f && x < last_x && y < last_y)
        {
            const int x0 = static_cast<int>(x);
            const int y0 = static_cast<int>(y);
            const std::byte* row = source + static_cast<std::size_t>(y0) * source_stride_bytes + x0;
            value = blend(x - static_cast<float>(x0), y - static_cast<float>(y0), level(row[0]), level(row[1]),
                          level(row[source_stride_bytes]), level(row[std::size_t{ source_stride_bytes } + 1]));
        }
        else if (x > -1.0f && y > -1.0f && x < last_x + 1 && y < last_y + 1)
        {
            // Truncating x + 1 floors x above -1.
            const int x0 = static_cast<int>(x + 1.0f) - 1;
            const int y0 = static_cast<int>(y + 1.0f) - 1;
            value = blend(x - static_cast<float>(x0), y - static_cast<float>(y0), at(x0, y0), at(x0 + 1, y0),
                          at(x0, y0 + 1), at(x0 + 1, y0 + 1));
        }
        output[i] = static_cast<std::byte>(static_cast<uint8_t>(value + 0.5f));
    }
}

} // namespace pose_provider
