// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "calibration.hpp"

#include "inc/ego_camera/raw_calibration.hpp"

#include <yaml-cpp/yaml.h>

// Separate std headers because clang-format's module category does not match yaml-cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plugins::ego
{

namespace
{

YAML::Node unique_yaml_field(const YAML::Node& mapping, const std::string& name)
{
    YAML::Node result(YAML::NodeType::Undefined);
    if (!mapping.IsMap())
        return result;
    for (const auto& field : mapping)
    {
        if (!field.first.IsScalar() || field.first.as<std::string>() != name)
            continue;
        if (result.IsDefined())
            return YAML::Node(YAML::NodeType::Undefined);
        result = field.second;
    }
    return result;
}

bool valid_imu_transform(const std::array<std::array<double, 4>, 4>& matrix)
{
    constexpr double tolerance = 1e-4;
    for (size_t column = 0; column < 4; ++column)
    {
        if (std::abs(matrix[3][column] - (column == 3 ? 1.0 : 0.0)) > 1e-6)
            return false;
    }
    for (size_t first = 0; first < 3; ++first)
    {
        for (size_t second = 0; second < 3; ++second)
        {
            double product = 0;
            for (size_t row = 0; row < 3; ++row)
                product += matrix[row][first] * matrix[row][second];
            if (std::abs(product - (first == second ? 1.0 : 0.0)) > tolerance)
                return false;
        }
    }
    const auto& a = matrix[0];
    const auto& b = matrix[1];
    const auto& c = matrix[2];
    const double determinant =
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    return std::abs(determinant - 1.0) <= tolerance;
}

} // namespace

bool populate_imu_calibration_from_yaml(const std::string& yaml, core::OrbbecEgoCalibrationT& calibration)
{
    core::OrbbecEgoExtrinsicsT extrinsics;
    try
    {
        std::string parse_input = yaml;
        if (parse_input.rfind("%YAML:1.0", 0) == 0)
            parse_input.replace(0, 9, "%YAML 1.1");
        const auto documents = YAML::LoadAll(parse_input);
        if (documents.size() != 1)
            return false;
        const auto camera = unique_yaml_field(documents.front(), "cam0");
        const auto transform = unique_yaml_field(camera, "T_cam_imu");
        if (!transform.IsSequence() || transform.size() != 4)
            return false;
        std::array<std::array<double, 4>, 4> matrix{};
        for (size_t row = 0; row < 4; ++row)
        {
            if (!transform[row].IsSequence() || transform[row].size() != 4)
                return false;
            for (size_t column = 0; column < 4; ++column)
            {
                const double value = transform[row][column].as<double>();
                if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                    return false;
                matrix[row][column] = value;
            }
        }
        if (!valid_imu_transform(matrix))
            return false;
        // EGO SDK d49af169 registers T_cam_imu as accel -> cam0, then declares
        // gyro/accel coincident. Its row-major rotation and mm translation copy directly.
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t column = 0; column < 3; ++column)
                extrinsics.rotation.push_back(static_cast<float>(matrix[row][column]));
            extrinsics.translation_mm.push_back(static_cast<float>(matrix[row][3]));
        }
    }
    catch (const YAML::Exception&)
    {
        return false;
    }

    auto accel = calibration.accel_to_left;
    auto gyro = calibration.gyro_to_left;
    if (!accel)
        accel = std::make_shared<core::OrbbecEgoExtrinsicsT>(extrinsics);
    if (!gyro)
        gyro = std::make_shared<core::OrbbecEgoExtrinsicsT>(extrinsics);
    calibration.accel_to_left = std::move(accel);
    calibration.gyro_to_left = std::move(gyro);
    return true;
}

std::vector<float> yaml_numbers(const std::string& text)
{
    std::vector<float> numbers;
    const char* cursor = text.c_str();
    while (*cursor != '\0')
    {
        char* end = nullptr;
        const float value = std::strtof(cursor, &end);
        if (end != cursor)
        {
            numbers.push_back(value);
            cursor = end;
        }
        else
            ++cursor;
    }
    return numbers;
}

bool populate_stereo_calibration_from_yaml(const std::string& yaml, core::OrbbecEgoCalibrationT& calibration)
{
    struct Camera
    {
        uint32_t width = 0;
        uint32_t height = 0;
        float fx = 0;
        float fy = 0;
        float cx = 0;
        float cy = 0;
        std::vector<float> distortion;
        std::vector<float> rotation;
        std::vector<float> translation;
    };
    std::vector<Camera> cameras;
    enum class Section
    {
        kNone,
        kIntrinsics,
        kDistortion,
        kRotation,
    };
    Section section = Section::kNone;
    std::istringstream input(yaml);
    std::string line;
    while (std::getline(input, line))
    {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos)
            continue;
        std::string key = line.substr(first);
        if (!key.empty() && key.back() == '\r')
            key.pop_back();
        if (key.rfind("- id:", 0) == 0)
        {
            cameras.emplace_back();
            section = Section::kNone;
            continue;
        }
        if (cameras.empty())
            continue;
        auto& camera = cameras.back();
        if (key == "intrinsics:")
        {
            section = Section::kIntrinsics;
            continue;
        }
        if (key == "distortion:")
        {
            section = Section::kDistortion;
            continue;
        }
        if (key == "rotation:")
        {
            section = Section::kRotation;
            continue;
        }
        if (key.rfind("translation:", 0) == 0)
        {
            camera.translation = yaml_numbers(key);
            section = Section::kNone;
            continue;
        }
        if (section == Section::kRotation && key.rfind("- [", 0) == 0)
        {
            const auto values = yaml_numbers(key);
            camera.rotation.insert(camera.rotation.end(), values.begin(), values.end());
            continue;
        }
        const auto colon = key.find(':');
        if (colon == std::string::npos)
            continue;
        const auto values = yaml_numbers(key.substr(colon + 1));
        if (values.empty())
            continue;
        if (key.rfind("image_width:", 0) == 0)
            camera.width = static_cast<uint32_t>(values.front());
        else if (key.rfind("image_height:", 0) == 0)
            camera.height = static_cast<uint32_t>(values.front());
        else if (section == Section::kIntrinsics)
        {
            if (key.rfind("fx:", 0) == 0)
                camera.fx = values.front();
            else if (key.rfind("fy:", 0) == 0)
                camera.fy = values.front();
            else if (key.rfind("cx:", 0) == 0)
                camera.cx = values.front();
            else if (key.rfind("cy:", 0) == 0)
                camera.cy = values.front();
        }
        else if (section == Section::kDistortion && (key.rfind("k", 0) == 0 || key.rfind("p", 0) == 0))
            camera.distortion.push_back(values.front());
    }
    if (cameras.size() < 2 || cameras[0].width == 0 || cameras[1].width == 0 || cameras[0].fx == 0 || cameras[1].fx == 0)
        return false;
    const auto intrinsics = [](const Camera& camera)
    {
        auto result = std::make_shared<core::OrbbecEgoCameraIntrinsicsT>();
        result->width = camera.width;
        result->height = camera.height;
        result->fx = camera.fx;
        result->fy = camera.fy;
        result->cx = camera.cx;
        result->cy = camera.cy;
        result->distortion_model = OB_DISTORTION_KANNALA_BRANDT4;
        result->distortion = camera.distortion;
        return result;
    };
    calibration.color_left = intrinsics(cameras[0]);
    calibration.color_right = intrinsics(cameras[1]);
    if (cameras[1].rotation.size() == 9 && cameras[1].translation.size() == 3)
    {
        calibration.left_to_right = std::make_shared<core::OrbbecEgoExtrinsicsT>();
        calibration.left_to_right->rotation = cameras[1].rotation;
        calibration.left_to_right->translation_mm = cameras[1].translation;
    }
    return true;
}

void detail::RawCalibrationTransfer::append(size_t total_size, size_t offset, const uint8_t* data, size_t size) noexcept
{
    if (failed_)
        return;
    // The SDK delivers contiguous chunks. Reject gaps and overlaps before copying.
    if (completed_ || !data || !size || !total_size || offset > total_size || size > total_size - offset ||
        offset != data_.size() || (total_size_ && total_size_ != total_size))
    {
        fail();
        return;
    }
    total_size_ = total_size;
    try
    {
        // Allocate only received bytes, not the device's untrusted total length.
        data_.append(reinterpret_cast<const char*>(data), size);
    }
    catch (...)
    {
        exception_ = std::current_exception();
        fail();
    }
}

void detail::RawCalibrationTransfer::complete() noexcept
{
    if (completed_)
        fail();
    completed_ = true;
}

void detail::RawCalibrationTransfer::fail() noexcept
{
    failed_ = true;
}

std::string detail::RawCalibrationTransfer::finish() const
{
    if (exception_)
        std::rethrow_exception(exception_);
    if (failed_ || !completed_ || !total_size_ || data_.size() != total_size_)
        throw std::runtime_error("Ego raw calibration transfer failed");
    return data_;
}

std::string raw_data(const std::shared_ptr<ob::Device>& device, OBPropertyID id)
{
    detail::RawCalibrationTransfer transfer;
    device->getRawData(id,
                       [&transfer](OBDataTranState state, OBDataChunk* chunk) noexcept
                       {
                           if (state == DATA_TRAN_STAT_TRANSFERRING && chunk)
                               transfer.append(chunk->fullDataSize, chunk->offset, chunk->data, chunk->size);
                           // DONE carries no payload, so handle it independently of chunk data.
                           else if (state == DATA_TRAN_STAT_DONE)
                               transfer.complete();
                           else
                               transfer.fail();
                       });
    return transfer.finish();
}

std::string json_escape(const std::string& input)
{
    std::string output;
    output.reserve(input.size());
    for (const char character : input)
    {
        switch (character)
        {
        case '\\':
            output += "\\\\";
            break;
        case '"':
            output += "\\\"";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            output += character;
            break;
        }
    }
    return output;
}

} // namespace plugins::ego
