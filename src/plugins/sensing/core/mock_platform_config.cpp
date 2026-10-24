// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_platform_config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace plugins
{
namespace sensing
{

namespace
{

uint32_t scale_divisor()
{
    const char* value = std::getenv("SENSING_MOCK_SCALE_DIV");
    if (!value)
        return 1;
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 1 ? static_cast<uint32_t>(parsed) : 1;
}

uint32_t scaled(uint32_t extent, uint32_t divisor)
{
    return std::max<uint32_t>(2, (extent / divisor) & ~1u);
}

} // namespace

std::vector<SensorInfo> mock_query_platform_config(const std::string& platform_config_json,
                                                   const std::string& platform_config_name,
                                                   const std::vector<uint32_t>& link_masks)
{
    std::ifstream file(platform_config_json);
    if (!file)
        throw std::runtime_error("ParseJsonFile failed: cannot open " + platform_config_json);

    nlohmann::json root;
    try
    {
        root = nlohmann::json::parse(file);
    }
    catch (const nlohmann::json::exception& e)
    {
        throw std::runtime_error("ParseJsonFile failed: " + std::string(e.what()));
    }

    const auto& configs = root.at("cameraConfigs");
    auto find_config = [&](const std::string& name) -> const nlohmann::json*
    {
        for (const auto& config : configs)
        {
            if (config.value("name", "") == name)
                return &config;
        }
        return nullptr;
    };

    const auto* platform = find_config(platform_config_name);
    if (!platform)
        throw std::runtime_error("GetSensorSystemConfig failed: no camera config named '" + platform_config_name + "'");

    const uint32_t divisor = scale_divisor();
    std::vector<SensorInfo> sensors;
    for (const auto& module : platform->at("cameraModules"))
    {
        const uint32_t link = module.at("linkIndex");
        const uint32_t transport = module.value("transportIndex", 0u);
        if (!link_masks.empty())
        {
            const bool enabled = transport < link_masks.size() && ((link_masks[transport] >> (4 * link)) & 0xF) != 0;
            if (!enabled)
                continue;
        }

        const std::string module_name = module.at("name");
        const auto* module_config = find_config(module_name);
        if (!module_config)
            throw std::runtime_error("GetSensorSystemConfig failed: no camera config named '" + module_name + "'");

        for (const auto& sensor : module_config->at("sensorInfo"))
        {
            SensorInfo info;
            info.id = static_cast<uint32_t>(sensors.size());
            info.name = module_name;
            const auto& channels = sensor.at("virtualChannels");
            if (!channels.empty())
            {
                info.width = scaled(channels.front().at("width"), divisor);
                info.height = scaled(channels.front().at("height"), divisor);
                info.fps = channels.front().at("fps");
            }
            sensors.push_back(std::move(info));
        }
    }
    return sensors;
}

} // namespace sensing
} // namespace plugins
