// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Resolve bundled sdk_config.json / hand_fk / wave-sdk next to the plugin
// binary so a vendored install does not require /opt/avatar-sdk at runtime.
// Relative hand_fk_config_root is resolved by the SDK against the config
// file directory; this helper rewrites it to an absolute path so a copied
// config (for example under /tmp) still finds the bundled hand_fk tree.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace plugins
{
namespace avatar
{

inline std::filesystem::path executable_dir(const char* argv0)
{
    if (argv0 == nullptr || argv0[0] == '\0')
    {
        return {};
    }
    std::error_code ec;
    const auto p = std::filesystem::absolute(argv0, ec);
    if (ec)
    {
        return {};
    }
    return p.parent_path();
}

inline std::string default_sdk_config_path(const char* argv0, const std::string& fallback = "sdk_config.json")
{
    const auto dir = executable_dir(argv0);
    if (!dir.empty())
    {
        const auto bundled = dir / "sdk_config.json";
        std::error_code ec;
        if (std::filesystem::is_regular_file(bundled, ec))
        {
            return bundled.string();
        }
    }
    return fallback;
}

inline void ensure_vendored_lib_path(const char* argv0)
{
    const auto exe_dir = executable_dir(argv0);
    if (exe_dir.empty())
    {
        return;
    }

    const std::filesystem::path candidates[] = {
        (exe_dir / ".." / ".." / "lib").lexically_normal(),
        (exe_dir / "lib").lexically_normal(),
    };

    std::string prepend;
    for (const auto& candidate : candidates)
    {
        std::error_code ec;
        const bool has_sdk = std::filesystem::is_regular_file(candidate / "libavatar_sdk.so", ec);
        const bool has_mumps = std::filesystem::is_regular_file(candidate / "libcoinmumps.so.3", ec) ||
                               std::filesystem::is_symlink(candidate / "libcoinmumps.so.3", ec);
        if (!has_sdk && !has_mumps)
        {
            continue;
        }
        if (!prepend.empty())
        {
            prepend.push_back(':');
        }
        prepend += candidate.string();
    }
    if (prepend.empty())
    {
        return;
    }

    if (const char* existing = std::getenv("LD_LIBRARY_PATH"); existing != nullptr && existing[0] != '\0')
    {
        prepend.push_back(':');
        prepend += existing;
    }
    ::setenv("LD_LIBRARY_PATH", prepend.c_str(), 1);
}

inline std::string normalize_transport_link(const std::string& value)
{
    if (value == "wired" || value == "ethernet_udp")
    {
        return "ethernet_udp";
    }
    if (value == "wireless" || value == "usb_serial")
    {
        return "usb_serial";
    }
    return value;
}

inline bool is_existing_dir(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

inline bool replace_json_string_field(std::string& json, const std::string& key, const std::string& value)
{
    const std::string needle = "\"" + key + "\"";
    const auto pos = json.find(needle);
    if (pos == std::string::npos)
    {
        return false;
    }
    const auto colon = json.find(':', pos + needle.size());
    const auto q1 = json.find('"', colon);
    if (colon == std::string::npos || q1 == std::string::npos)
    {
        return false;
    }
    const auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos)
    {
        return false;
    }
    json.replace(q1, q2 - q1 + 1, "\"" + value + "\"");
    return true;
}

inline void upsert_json_string_field(std::string& json, const std::string& key, const std::string& value)
{
    if (replace_json_string_field(json, key, value))
    {
        return;
    }
    const auto brace = json.find('{');
    if (brace == std::string::npos)
    {
        return;
    }
    json.insert(brace + 1, "\n  \"" + key + "\": \"" + value + "\",");
}

inline bool replace_json_array_field(std::string& json, const std::string& key, const std::string& array_literal)
{
    const std::string needle = "\"" + key + "\"";
    const auto pos = json.find(needle);
    if (pos == std::string::npos)
    {
        return false;
    }
    const auto colon = json.find(':', pos + needle.size());
    const auto lb = json.find('[', colon);
    if (colon == std::string::npos || lb == std::string::npos)
    {
        return false;
    }
    int depth = 0;
    size_t rb = lb;
    for (; rb < json.size(); ++rb)
    {
        if (json[rb] == '[')
        {
            ++depth;
        }
        else if (json[rb] == ']')
        {
            --depth;
            if (depth == 0)
            {
                break;
            }
        }
    }
    if (rb >= json.size())
    {
        return false;
    }
    json.replace(lb, rb - lb + 1, array_literal);
    return true;
}

inline std::string json_string_array(const std::vector<std::filesystem::path>& paths)
{
    std::string out = "[";
    for (size_t i = 0; i < paths.size(); ++i)
    {
        if (i != 0)
        {
            out += ", ";
        }
        out += "\"" + paths[i].string() + "\"";
    }
    out += "]";
    return out;
}

struct PreparedSdkConfig
{
    std::string json;
    std::string path;
};

inline PreparedSdkConfig prepare_sdk_config(const std::string& config_path,
                                            const std::filesystem::path& plugin_dir,
                                            const std::string& transport_override)
{
    PreparedSdkConfig prepared;
    prepared.path = config_path;

    std::ifstream in(config_path);
    if (!in)
    {
        return prepared;
    }
    prepared.json.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (prepared.json.find('{') == std::string::npos)
    {
        prepared.json.clear();
        return prepared;
    }

    const auto config_dir = std::filesystem::path(config_path).parent_path();
    const auto hand_fk = is_existing_dir(config_dir / "hand_fk") ? (config_dir / "hand_fk") :
                         is_existing_dir(plugin_dir / "hand_fk") ? (plugin_dir / "hand_fk") :
                                                                   std::filesystem::path{};
    if (!hand_fk.empty())
    {
        upsert_json_string_field(prepared.json, "hand_fk_config_root", hand_fk.lexically_normal().string() + "/");
    }

    std::vector<std::filesystem::path> wave_roots;
    for (const auto& candidate : { plugin_dir / "wave-sdk", config_dir / "wave-sdk" })
    {
        if (is_existing_dir(candidate))
        {
            wave_roots.push_back(candidate.lexically_normal());
        }
    }
    if (wave_roots.empty())
    {
        for (const char* fallback : { "/opt/avatar-sdk/share/wave-sdk", "/opt/sharpa-wave-sdk" })
        {
            if (is_existing_dir(fallback))
            {
                wave_roots.emplace_back(fallback);
            }
        }
    }
    if (!wave_roots.empty())
    {
        const auto literal = json_string_array(wave_roots);
        if (!replace_json_array_field(prepared.json, "wave_sdk_root", literal))
        {
            const auto brace = prepared.json.find('{');
            prepared.json.insert(brace + 1, "\n  \"wave_sdk_root\": " + literal + ",");
        }
    }

    if (!transport_override.empty())
    {
        upsert_json_string_field(prepared.json, "transport_link", normalize_transport_link(transport_override));
    }
    else if (prepared.json.find("\"transport_link\"") == std::string::npos)
    {
        upsert_json_string_field(prepared.json, "transport_link", "usb_serial");
    }

    // Keep the file the SDK treats as "config location" next to the plugin so
    // any leftover relative keys (hand_fk, wave-sdk) do not resolve under /tmp.
    if (!plugin_dir.empty())
    {
        const auto runtime = plugin_dir / ".sdk_config.runtime.json";
        std::ofstream out(runtime);
        if (out)
        {
            out << prepared.json;
            if (out)
            {
                prepared.path = runtime.string();
            }
        }
    }

    return prepared;
}

} // namespace avatar
} // namespace plugins
