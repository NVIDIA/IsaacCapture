// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "temporary_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <plugin_manager/plugin_manager.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void write_metadata(const std::filesystem::path& search_directory, const std::string& extra_lines)
{
    const std::filesystem::path plugin_directory = search_directory / "sample";
    std::filesystem::create_directory(plugin_directory);
    std::ofstream metadata(plugin_directory / "plugin.yaml");
    metadata << "name: sample_plugin\n"
                "command: /bin/true\n"
             << extra_lines;
}

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
    REQUIRE(info.shutdown_timeout_seconds == core::DEFAULT_PLUGIN_SHUTDOWN_TIMEOUT_SECONDS);
    REQUIRE_THROWS_AS(manager.get_plugin_info("missing"), std::runtime_error);
}

TEST_CASE("plugin.yaml sets the plugin's shutdown timeout", "[plugin_manager][metadata]")
{
    SECTION("a valid value is reported")
    {
        TemporaryDirectory search_directory;
        write_metadata(search_directory.path(), "shutdown_timeout_seconds: 15.5\n");

        const core::PluginManager manager({ search_directory.path().string() });
        REQUIRE(manager.get_plugin_info("sample_plugin").shutdown_timeout_seconds == 15.5);
    }

    SECTION("an invalid value skips the plugin")
    {
        for (const std::string value : { "0", "-3", ".nan", ".inf", "3601", "soon" })
        {
            CAPTURE(value);
            TemporaryDirectory search_directory;
            write_metadata(search_directory.path(), "shutdown_timeout_seconds: " + value + "\n");

            const core::PluginManager manager({ search_directory.path().string() });
            REQUIRE(manager.get_plugin_names().empty());
        }
    }
}

TEST_CASE("plugin rejects an out-of-range shutdown timeout before launching", "[plugin_manager]")
{
    for (const double seconds : { 0.0, -1.0, std::nan(""), std::numeric_limits<double>::infinity(),
                                  core::MAX_PLUGIN_SHUTDOWN_TIMEOUT_SECONDS + 1.0 })
    {
        CAPTURE(seconds);
        REQUIRE_THROWS_AS(core::Plugin("never-launched", "", "test-root", {}, seconds), std::invalid_argument);
    }
}
