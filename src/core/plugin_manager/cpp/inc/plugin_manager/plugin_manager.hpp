// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "plugin.hpp"

#include <log_bridge/logger.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace core
{

struct DeviceInfo
{
    std::string path;
    std::string type;
    std::string description;
};

struct PluginInfo
{
    std::string name;
    std::string description;
    std::string command;
    std::string version;
    std::string working_dir;
    std::vector<std::string> args;
    std::vector<DeviceInfo> devices;
};

class PluginManager
{
public:
    /**
     * @brief Construct a PluginManager and discover plugins in the given search paths.
     * @param search_paths List of directories to search for plugins.
     */
    PluginManager(const std::vector<std::string>& search_paths);

    /**
     * @brief Get the list of discovered plugin names.
     * @return List of discovered plugin names.
     */
    std::vector<std::string> get_plugin_names() const;

    /**
     * @brief Get the complete immutable descriptor for a discovered plugin.
     * @param plugin_name The name of the plugin to query.
     * @return A copy of the plugin metadata and device descriptors.
     */
    PluginInfo get_plugin_info(const std::string& plugin_name) const;

    /**
     * @brief Query available devices from a plugin.
     * @param plugin_name The name of the plugin to query.
     * @return List of device paths from the plugin metadata.
     */
    std::vector<std::string> query_devices(const std::string& plugin_name) const;

    /**
     * @brief Start a plugin and return a RAII handle.
     * @param plugin_name The name of the plugin to start.
     * @param plugin_root_id The root ID for the plugin.
     * @param plugin_args Optional arguments appended after plugin.yaml args.
     * @return A unique pointer to the Plugin instance (stops on destruction).
     */
    std::unique_ptr<Plugin> start(const std::string& plugin_name,
                                  const std::string& plugin_root_id,
                                  const std::vector<std::string>& plugin_args = {});

private:
    void discover_plugins();

    std::vector<std::string> m_search_paths;
    std::map<std::string, PluginInfo> m_discovered_plugins;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.core.PluginManager");
};

} // namespace core
