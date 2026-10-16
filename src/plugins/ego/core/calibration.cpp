// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "calibration.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace plugins::ego
{

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

bool populate_stereo_calibration_from_yaml(const std::string& yaml, core::EgoCalibrationT& calibration)
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
        auto result = std::make_shared<core::EgoCameraIntrinsicsT>();
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
        calibration.left_to_right = std::make_shared<core::EgoExtrinsicsT>();
        calibration.left_to_right->rotation = cameras[1].rotation;
        calibration.left_to_right->translation_mm = cameras[1].translation;
    }
    return true;
}

std::string raw_data(const std::shared_ptr<ob::Device>& device, OBPropertyID id)
{
    std::vector<uint8_t> result;
    bool failed = false;
    device->getRawData(id,
                       [&result, &failed](OBDataTranState state, OBDataChunk* chunk)
                       {
                           if (state == DATA_TRAN_STAT_TRANSFERRING && chunk && chunk->data && chunk->size)
                           {
                               if (result.size() < chunk->fullDataSize)
                                   result.resize(chunk->fullDataSize);
                               std::copy(chunk->data, chunk->data + chunk->size, result.begin() + chunk->offset);
                           }
                           else if (state < 0)
                               failed = true;
                       });
    if (failed)
        throw std::runtime_error("Ego raw calibration transfer failed");
    return std::string(result.begin(), result.end());
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
