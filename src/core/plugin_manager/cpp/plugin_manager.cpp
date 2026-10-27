// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_manager/plugin_manager.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace core
{

namespace fs = std::filesystem;

// Parse plugin metadata from YAML file using yaml-cpp
static PluginInfo parse_plugin_yaml(const std::string& path)
{
    PluginInfo info;

    try
    {
        YAML::Node config = YAML::LoadFile(path);

        // Parse root-level fields
        if (config["name"])
            info.name = config["name"].as<std::string>();
        if (config["description"])
            info.description = config["description"].as<std::string>();
        if (config["command"])
            info.command = config["command"].as<std::string>();
        if (config["version"])
            info.version = config["version"].as<std::string>();
        if (config["args"] && config["args"].IsSequence())
        {
            for (const auto& arg_node : config["args"])
            {
                info.args.push_back(arg_node.as<std::string>());
            }
        }

        // Parse devices list
        if (config["devices"] && config["devices"].IsSequence())
        {
            for (const auto& device_node : config["devices"])
            {
                DeviceInfo device;

                if (device_node["path"])
                    device.path = device_node["path"].as<std::string>();
                if (device_node["type"])
                    device.type = device_node["type"].as<std::string>();
                if (device_node["description"])
                    device.description = device_node["description"].as<std::string>();

                if (!device.path.empty())
                {
                    info.devices.push_back(device);
                }
            }
        }
    }
    catch (const YAML::Exception& e)
    {
        throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
    }

    return info;
}

PluginManager::PluginManager(const std::vector<std::string>& search_paths)
{
    for (const auto& path : search_paths)
    {
        if (fs::exists(path) && fs::is_directory(path))
        {
            m_search_paths.push_back(path);
        }
    }
    discover_plugins();
}

std::vector<std::string> PluginManager::get_plugin_names() const
{
    std::vector<std::string> names;
    names.reserve(m_discovered_plugins.size());
    for (const auto& [name, info] : m_discovered_plugins)
    {
        names.push_back(name);
    }
    return names;
}

PluginInfo PluginManager::get_plugin_info(const std::string& plugin_name) const
{
    const auto it = m_discovered_plugins.find(plugin_name);
    if (it == m_discovered_plugins.end())
    {
        throw std::runtime_error("Plugin not found: " + plugin_name);
    }
    return it->second;
}

void PluginManager::discover_plugins()
{
    m_discovered_plugins.clear();

    for (const auto& base_path : m_search_paths)
    {
        try
        {
            // Iterate over subdirectories in search path
            for (const auto& entry : fs::directory_iterator(base_path))
            {
                if (entry.is_directory())
                {
                    fs::path plugin_dir = entry.path();
                    fs::path yaml_path = plugin_dir / "plugin.yaml";

                    if (fs::exists(yaml_path) && fs::is_regular_file(yaml_path))
                    {
                        try
                        {
                            PluginInfo info = parse_plugin_yaml(yaml_path.string());

                            if (!info.name.empty() && !info.command.empty())
                            {
                                info.working_dir = plugin_dir.string();

                                m_discovered_plugins[info.name] = info;
                            }
                        }
                        catch (const std::exception& e)
                        {
                            m_logger->error("Error parsing metadata for {}: {}", plugin_dir.string(), e.what());
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            m_logger->error("Error scanning directory {}: {}", base_path, e.what());
        }
    }
}

std::vector<std::string> PluginManager::query_devices(const std::string& plugin_name) const
{
    const PluginInfo info = get_plugin_info(plugin_name);
    std::vector<std::string> result;
    result.reserve(info.devices.size());
    for (const auto& device : info.devices)
    {
        result.push_back(device.path);
    }
    return result;
}

std::unique_ptr<Plugin> PluginManager::start(const std::string& plugin_name,
                                             const std::string& plugin_root_id,
                                             const std::vector<std::string>& plugin_args)
{
    auto it = m_discovered_plugins.find(plugin_name);
    if (it == m_discovered_plugins.end())
    {
        throw std::runtime_error("Plugin not found: " + plugin_name);
    }

    const auto& info = it->second;
    std::vector<std::string> args = info.args;
    args.insert(args.end(), plugin_args.begin(), plugin_args.end());
    return std::make_unique<Plugin>(info.command, info.working_dir, plugin_root_id, args);
}

} // namespace core
