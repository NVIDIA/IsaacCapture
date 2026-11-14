// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_plugin/config.hpp"

#include <cstdlib>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace plugins::camera
{

namespace
{

[[noreturn]] void fail(const std::string& path, const YAML::Mark& mark, const std::string& what)
{
    const std::string line = mark.is_null() ? "" : ":" + std::to_string(mark.line + 1);
    throw std::invalid_argument(path + line + ": " + what);
}

CameraConfig read_camera(const std::string& path, const YAML::Node& id, const YAML::Node& settings)
{
    if (!id.IsScalar() || id.Scalar().empty())
    {
        fail(path, id.Mark(), "a camera needs an id");
    }
    CameraConfig camera;
    camera.id = id.Scalar();
    if (!settings.IsMap())
    {
        fail(path, settings.Mark(), "camera '" + camera.id + "' needs a backend, as in `backend: synthetic`");
    }
    for (const auto& setting : settings)
    {
        if (!setting.first.IsScalar() || !setting.second.IsScalar())
        {
            fail(path, setting.first.Mark(), "camera '" + camera.id + "' takes settings of the form `name: value`");
        }
        if (setting.first.Scalar() == "backend")
        {
            camera.backend = setting.second.Scalar();
        }
        else
        {
            camera.options[setting.first.Scalar()] = setting.second.Scalar();
        }
    }
    if (camera.backend.empty())
    {
        fail(path, settings.Mark(), "camera '" + camera.id + "' needs a backend, as in `backend: synthetic`");
    }
    return camera;
}

} // namespace

PluginConfig load_config_file(const std::string& path)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const YAML::BadFile&)
    {
        throw std::invalid_argument(path + ": the configuration file cannot be read");
    }
    catch (const YAML::ParserException& error)
    {
        fail(path, error.mark, "not valid YAML: " + error.msg);
    }
    if (!root.IsMap())
    {
        fail(path, root.Mark(), "a configuration starts with `cameras:`");
    }

    PluginConfig config;
    for (const auto& entry : root)
    {
        const std::string key = entry.first.IsScalar() ? entry.first.Scalar() : "";
        if (key == "cameras")
        {
            if (!entry.second.IsMap())
            {
                fail(path, entry.second.Mark(), "`cameras:` lists cameras by id, each with a backend");
            }
            for (const auto& camera : entry.second)
            {
                config.cameras.push_back(read_camera(path, camera.first, camera.second));
            }
        }
        else if (key == "slots")
        {
            try
            {
                config.slot_count = entry.second.as<std::uint32_t>();
            }
            catch (const YAML::Exception&)
            {
                fail(path, entry.second.Mark(), "`slots:` takes a whole number");
            }
        }
        else
        {
            fail(path, entry.first.Mark(),
                 "there is no setting '" + key + "'; a configuration has `cameras:` and `slots:`");
        }
    }
    return config;
}

std::string default_config_path()
{
    const char* config_home = std::getenv("XDG_CONFIG_HOME");
    if (config_home != nullptr && config_home[0] != '\0')
    {
        return std::string(config_home) + "/isaaccapture/cameras.yaml";
    }
    const char* home = std::getenv("HOME");
    return std::string(home != nullptr ? home : "") + "/.config/isaaccapture/cameras.yaml";
}

std::vector<CameraConfig> discover_cameras(std::span<const Backend> backends)
{
    std::vector<CameraConfig> cameras;
    for (const Backend& backend : backends)
    {
        if (backend.discover)
        {
            for (CameraConfig& camera : backend.discover())
            {
                cameras.push_back(std::move(camera));
            }
        }
    }
    return cameras;
}

} // namespace plugins::camera
