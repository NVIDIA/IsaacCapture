// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <plugin_manager/plugin_manager.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int suffix = 0; suffix < 100; ++suffix)
        {
            m_path = std::filesystem::temp_directory_path() /
                     ("isaacteleop_plugin_manager_" + std::to_string(seed) + "_" + std::to_string(suffix));
            if (std::filesystem::create_directory(m_path))
            {
                return;
            }
        }
        throw std::runtime_error("Failed to create temporary test directory");
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    const std::filesystem::path& path() const
    {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

} // namespace

TEST_CASE("plugin manager returns complete descriptors", "[plugin_manager][metadata]")
{
    TemporaryDirectory search_directory;
    const std::filesystem::path plugin_directory = search_directory.path() / "sample";
    std::filesystem::create_directory(plugin_directory);

    std::ofstream metadata(plugin_directory / "plugin.yaml");
    metadata << "name: sample_plugin\n"
                "description: Sample plugin\n"
                "command: /bin/true\n"
                "version: 1.2.3\n"
                "args:\n"
                "  - --alpha\n"
                "  - beta\n"
                "devices:\n"
                "  - path: /hands/left\n"
                "    type: hand\n"
                "    description: Left hand\n"
                "  - path: /camera\n"
                "    type: camera\n"
                "    description: Front camera\n";
    metadata.close();

    const core::PluginManager manager({ search_directory.path().string() });
    const core::PluginInfo info = manager.get_plugin_info("sample_plugin");

    REQUIRE(info.name == "sample_plugin");
    REQUIRE(info.description == "Sample plugin");
    REQUIRE(info.command == "/bin/true");
    REQUIRE(info.version == "1.2.3");
    REQUIRE(info.working_dir == plugin_directory.string());
    REQUIRE((info.args == std::vector<std::string>{ "--alpha", "beta" }));
    REQUIRE(info.devices.size() == 2);
    REQUIRE(info.devices[0].path == "/hands/left");
    REQUIRE(info.devices[0].type == "hand");
    REQUIRE(info.devices[0].description == "Left hand");
    REQUIRE(info.devices[1].path == "/camera");
    REQUIRE((manager.query_devices("sample_plugin") == std::vector<std::string>{ "/hands/left", "/camera" }));
    REQUIRE_THROWS_AS(manager.get_plugin_info("missing"), std::runtime_error);
}
