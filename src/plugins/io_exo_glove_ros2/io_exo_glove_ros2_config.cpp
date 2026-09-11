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

//! Every key the config file may define. Anything else is rejected, so a typo (say `left_topics`)
//! fails loudly instead of silently leaving the built-in default in place.
constexpr std::string_view kKnownKeys[] = { "left_topic", "right_topic", "left_collection_id", "right_collection_id" };

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
    assign(root, "left_topic", options.left_topic, path);
    assign(root, "right_topic", options.right_topic, path);
    assign(root, "left_collection_id", options.left_collection_id, path);
    assign(root, "right_collection_id", options.right_collection_id, path);

    resolution.loaded_path = path;
    return resolution;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
