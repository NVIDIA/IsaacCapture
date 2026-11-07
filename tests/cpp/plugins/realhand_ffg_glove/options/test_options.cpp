// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <realhand_ffg_glove/realhand_ffg_glove_options.hpp>

#include <cstdlib>
#include <optional>
#include <string>

using namespace plugins::realhand_ffg_glove;

namespace
{
struct Environment
{
    std::string name;
    std::optional<std::string> old;
    Environment(const char* key, const char* value) : name(key)
    {
        if (const char* previous = std::getenv(key))
            old = previous;
        ::setenv(key, value, 1);
    }
    ~Environment()
    {
        if (old)
            ::setenv(name.c_str(), old->c_str(), 1);
        else
            ::unsetenv(name.c_str());
    }
};
}

TEST_CASE("CTS launches with environment defaults and no plugin arguments", "[realhand_ffg_options]")
{
    Environment calibration("REALHAND_FFG_GLOVE_CALIBRATION", "/tmp/human.yml");
    Environment ports("REALHAND_FFG_GLOVE_PORTS", "/dev/ttyUSB0,/dev/ttyUSB1");
    Environment source("REALHAND_FFG_GLOVE_WRIST_SOURCE", "hand_tracking");
    const auto options = parse_options({});
    REQUIRE(options.config.calibration_path == "/tmp/human.yml");
    REQUIRE(options.ports == std::vector<std::string>{ "/dev/ttyUSB0", "/dev/ttyUSB1" });
    REQUIRE(options.config.wrist.mode == plugin_utils::WristSourceMode::HandTracking);
    const auto overridden =
        parse_options({ "--calibration=/tmp/other.yml", "--ports=/dev/ttyUSB9", "--wrist-source=controller" });
    REQUIRE(overridden.config.calibration_path == "/tmp/other.yml");
    REQUIRE(overridden.ports == std::vector<std::string>{ "/dev/ttyUSB9" });
    REQUIRE(overridden.config.wrist.mode == plugin_utils::WristSourceMode::Controller);
}

TEST_CASE("Baud-rate parsing rejects malformed and unsupported values", "[realhand_ffg_options]")
{
    for (const auto* value : { "abc", "99999999999", "460800abc", "460800.5", "0", "-1", "115200", "" })
    {
        CAPTURE(value);
        REQUIRE_THROWS_AS(parse_baudrates(value), std::invalid_argument);
    }
    REQUIRE(parse_baudrates("460800,921600") == std::vector<int>{ 460800, 921600 });
    REQUIRE(parse_wrist_source("handtracking") == plugin_utils::WristSourceMode::HandTracking);
}
