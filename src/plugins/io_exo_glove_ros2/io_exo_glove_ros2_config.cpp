// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace fs = std::filesystem;

namespace
{

//! Every key the config file may define. Anything else is rejected, so a typo (say `joint_topics`)
//! fails loudly instead of silently leaving the built-in default in place.
constexpr std::string_view kKnownKeys[] = { "joints_topic",
                                            "links_topic",
                                            "joints_left_collection_id",
                                            "joints_right_collection_id",
                                            "tips_left_collection_id",
                                            "tips_right_collection_id",
                                            "tips_reference_frame",
                                            "wrist_source",
                                            "left_aim_to_wrist",
                                            "right_aim_to_wrist",
                                            "left_joint_zero_offsets",
                                            "right_joint_zero_offsets",
                                            "wrist_smoothing_ms" };

std::string known_keys_as_text()
{
    std::string text;
    for (const std::string_view key : kKnownKeys)
    {
        if (!text.empty())
        {
            text += ", ";
        }
        text += key;
    }
    return text;
}

//! Rejects a root node that is not a mapping of known keys.
void validate_keys(const YAML::Node& root, const std::string& path)
{
    if (!root.IsMap())
    {
        throw std::runtime_error("config '" + path + "': expected a YAML mapping of keys (" + known_keys_as_text() + ")");
    }

    for (const auto& entry : root)
    {
        const std::string key = entry.first.as<std::string>();
        const bool known = std::any_of(std::begin(kKnownKeys), std::end(kKnownKeys),
                                       [&key](const std::string_view candidate) { return candidate == key; });
        if (!known)
        {
            throw std::runtime_error("config '" + path + "': unknown key '" + key +
                                     "' (valid keys: " + known_keys_as_text() + ")");
        }
    }
}

//! Applies `key` to @p target when the file defines it; a missing key leaves @p target untouched.
void assign(const YAML::Node& root, const char* key, std::string& target, const std::string& path)
{
    const YAML::Node node = root[key];
    if (!node)
    {
        return;
    }

    if (!node.IsScalar())
    {
        throw std::runtime_error("config '" + path + "': '" + key + "' must be a single scalar string");
    }

    const std::string value = node.as<std::string>();
    if (value.empty())
    {
        throw std::runtime_error("config '" + path + "': '" + key + "' must not be empty");
    }

    target = value;
}

//! Applies one per-hand `*_joint_zero_offsets` string. Unlike the other string keys an empty value is
//! meaningful -- it keeps the baked table -- but a non-empty one is parsed here, so a wrong count
//! fails at config load (and therefore in --print-config) instead of after the session exists.
void assign_zero_offsets(const YAML::Node& root, const char* key, std::string& target, const std::string& path)
{
    const YAML::Node node = root[key];
    if (!node)
    {
        return;
    }
    // `key:` with no scalar is the same as an empty string: keep the baked table.
    if (node.IsNull())
    {
        target.clear();
        return;
    }

    if (!node.IsScalar())
    {
        throw std::runtime_error("config '" + path + "': '" + key + "' must be a single scalar string");
    }

    const std::string value = node.as<std::string>();
    try
    {
        (void)parse_joint_zero_offsets(value, key);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("config '" + path + "': " + e.what());
    }
    target = value;
}

//! Applies a numeric `key` to @p target; a non-numeric value is rejected rather than truncated.
void assign_number(const YAML::Node& root, const char* key, double& target, const std::string& path)
{
    const YAML::Node node = root[key];
    if (!node)
    {
        return;
    }
    if (!node.IsScalar())
    {
        throw std::runtime_error("config '" + path + "': '" + key + "' must be a single number");
    }
    try
    {
        const double value = node.as<double>();
        if (!(value >= 0.0))
        {
            throw std::runtime_error("config '" + path + "': '" + key + "' must not be negative");
        }
        target = value;
    }
    catch (const YAML::Exception& e)
    {
        throw std::runtime_error("config '" + path + "': '" + key + "' must be a number: " + std::string(e.what()));
    }
}

//! Implicit candidates, used only when no file was named explicitly.
std::vector<std::string> implicit_candidates()
{
    // The plugin launcher chdir()s into the plugin directory, so this resolves to the copy installed
    // next to plugin.yaml. Running by hand from a directory holding a config/ subdirectory works too.
    return { (fs::current_path() / "config" / kConfigFileName).string() };
}

} // namespace

ConfigResolution load_options_from_config(const std::string& explicit_path, IoExoGloveRos2Plugin::Options& options)
{
    ConfigResolution resolution;

    // An explicit request must be honoured exactly: no fallback to the implicit candidates, and a
    // missing file is an error rather than a silent return to the built-in defaults.
    std::string explicit_file = explicit_path;
    if (explicit_file.empty())
    {
        const char* from_env = std::getenv(kConfigEnvVar);
        if (from_env != nullptr && *from_env != '\0')
        {
            explicit_file = from_env;
        }
    }

    resolution.searched_paths = explicit_file.empty() ? implicit_candidates() : std::vector<std::string>{ explicit_file };

    std::string path;
    for (const auto& candidate : resolution.searched_paths)
    {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec))
        {
            path = candidate;
            break;
        }
    }

    if (path.empty())
    {
        if (!explicit_file.empty())
        {
            throw std::runtime_error("config file not found: " + explicit_file);
        }
        return resolution; // Nothing to load: the caller keeps the built-in Options defaults.
    }

    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const YAML::Exception& e)
    {
        throw std::runtime_error("config '" + path + "': YAML parse error: " + std::string(e.what()));
    }

    // An empty file carries no overrides, and is not an error.
    if (root.IsNull())
    {
        resolution.loaded_path = path;
        return resolution;
    }

    validate_keys(root, path);
    assign(root, "joints_topic", options.joints_topic, path);
    assign(root, "links_topic", options.links_topic, path);
    assign(root, "joints_left_collection_id", options.joints_left_collection_id, path);
    assign(root, "joints_right_collection_id", options.joints_right_collection_id, path);
    assign(root, "tips_left_collection_id", options.tips_left_collection_id, path);
    assign(root, "tips_right_collection_id", options.tips_right_collection_id, path);
    assign(root, "tips_reference_frame", options.tips_reference_frame, path);
    assign(root, "wrist_source", options.wrist_source, path);
    assign(root, "left_aim_to_wrist", options.left_aim_to_wrist, path);
    assign(root, "right_aim_to_wrist", options.right_aim_to_wrist, path);
    assign_zero_offsets(root, "left_joint_zero_offsets", options.left_joint_zero_offsets, path);
    assign_zero_offsets(root, "right_joint_zero_offsets", options.right_joint_zero_offsets, path);
    assign_number(root, "wrist_smoothing_ms", options.wrist_smoothing_ms, path);

    resolution.loaded_path = path;
    return resolution;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
