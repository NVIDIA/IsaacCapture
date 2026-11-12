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
                                            "joints_collection_id",
                                            "tips_left_collection_id",
                                            "tips_right_collection_id",
                                            "tips_reference_frame",
                                            "wrist_source",
                                            "wrist_alignment",
                                            "reference_left",
                                            "reference_right",
                                            "export_session_reference" };

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
    assign(root, "joints_topic", options.joints_topic, path);
    assign(root, "links_topic", options.links_topic, path);
    assign(root, "joints_collection_id", options.joints_collection_id, path);
    assign(root, "tips_left_collection_id", options.tips_left_collection_id, path);
    assign(root, "tips_right_collection_id", options.tips_right_collection_id, path);
    assign(root, "tips_reference_frame", options.tips_reference_frame, path);
    assign(root, "wrist_source", options.wrist_source, path);
    assign(root, "wrist_alignment", options.wrist_alignment, path);
    assign(root, "reference_left", options.reference_left, path);
    assign(root, "reference_right", options.reference_right, path);
    assign(root, "export_session_reference", options.export_session_reference, path);

    resolution.loaded_path = path;
    return resolution;
}

SessionReferenceFile load_session_reference_file(const std::string& path)
{
    if (path.empty())
    {
        throw std::runtime_error("--session-reference= needs a file path");
    }
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const YAML::Exception& e)
    {
        throw std::runtime_error("session reference '" + path + "': YAML parse error: " + std::string(e.what()));
    }
    if (!root || !root.IsMap())
    {
        throw std::runtime_error("session reference '" + path +
                                 "': expected a YAML mapping with reference_left / reference_right");
    }

    SessionReferenceFile reference;
    for (const auto& entry : root)
    {
        const std::string key = entry.first.as<std::string>();
        std::string* target =
            key == "reference_left" ? &reference.left : (key == "reference_right" ? &reference.right : nullptr);
        if (target == nullptr)
        {
            continue; // Other keys belong to the main config; only the reference keys are read here.
        }
        if (!entry.second.IsScalar())
        {
            throw std::runtime_error("session reference '" + path + "': '" + key + "' must be a single scalar string");
        }
        *target = entry.second.as<std::string>();
        if (target->empty())
        {
            throw std::runtime_error("session reference '" + path + "': '" + key + "' must not be empty");
        }
    }
    return reference;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
