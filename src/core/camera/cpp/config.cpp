// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera/config.hpp"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace core::camera
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

PluginConfig load_config_file(const std::string& path, std::string_view backend)
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
    std::vector<CameraConfig> cameras; // of every backend
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
                cameras.push_back(read_camera(path, camera.first, camera.second));
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

    std::copy_if(cameras.begin(), cameras.end(), std::back_inserter(config.cameras),
                 [&](const CameraConfig& camera) { return camera.backend == backend; });
    if (config.cameras.empty() && !cameras.empty())
    {
        // Most often a misspelt backend, so the message shows what the file says.
        std::string named;
        for (const CameraConfig& camera : cameras)
        {
            named += (named.empty() ? "" : ", ") + camera.id + " (backend: " + camera.backend + ")";
        }
        throw std::invalid_argument(path + ": none of its cameras has `backend: " + std::string(backend) +
                                    "`, the backend of this plugin. It names: " + named);
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

} // namespace core::camera
