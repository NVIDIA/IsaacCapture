// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the plugin's command line.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <jmex/plugin_args.hpp>

#include <optional>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using namespace plugins::jmex;

namespace
{

//! Parse as the plugin would, with the root id the PluginManager passes ahead of the arguments.
std::optional<PluginArgs> parse_command_line(const std::vector<std::string>& args)
{
    std::vector<const char*> argv{ "jmex_plugin", "--plugin-root-id=jmex" };
    for (const std::string& arg : args)
    {
        argv.push_back(arg.c_str());
    }
    return parse_plugin_args(static_cast<int>(argv.size()), argv.data());
}

//! The arguments of a command line that does not ask for --help.
PluginArgs parse(const std::vector<std::string>& args)
{
    return parse_command_line(args).value();
}

} // namespace

TEST_CASE("plugin args: every flag sets its field, and the last occurrence wins", "[unit][jmex][args]")
{
    const PluginArgs args =
        parse({ "--channel=2", "--channel=255", "--tcp-port=7001", "--udp-start-port=20000", "--bind-ip=192.0.2.10",
                "--broadcast-ip=192.0.2.10", "--collection-id=suit_b", "--device-clock=local" });
    REQUIRE(args.receiver.channel == 255);
    REQUIRE(args.receiver.tcp_port == 7001);
    REQUIRE(args.receiver.udp_start_port == 20000);
    REQUIRE(args.receiver.bind_ip == "192.0.2.10");
    REQUIRE(args.receiver.broadcast_ip == "192.0.2.10");
    REQUIRE(args.collection_id == "suit_b");
    REQUIRE(args.device_clock == DeviceClock::LOCAL);
}

TEST_CASE("plugin args: an argument it does not know stops the plugin", "[unit][jmex][args]")
{
    // A space instead of '=': "--channel" is unknown and "2" is positional, so the default channel would stay.
    REQUIRE_THROWS_WITH(parse({ "--channel", "2" }), ContainsSubstring("'--channel'"));
    REQUIRE_THROWS_WITH(parse({ "--chanel=2" }), ContainsSubstring("'--chanel=2'"));
    REQUIRE_THROWS_WITH(parse({ "2" }), ContainsSubstring("'2'"));
}

TEST_CASE("plugin args: a device clock it cannot read stops the plugin", "[unit][jmex][args]")
{
    REQUIRE_THROWS_WITH(parse({ "--device-clock=Device" }), ContainsSubstring("--device-clock="));
    REQUIRE_THROWS_WITH(parse({ "--device-clock=" }), ContainsSubstring("--device-clock="));
    REQUIRE_THROWS_WITH(parse({ "--device-clock=sequence" }), ContainsSubstring("--device-clock="));
}

TEST_CASE("plugin args: the device clock the startup line names can be passed back in", "[unit][jmex][args]")
{
    for (DeviceClock clock : { DeviceClock::DEVICE, DeviceClock::LOCAL })
    {
        REQUIRE(parse({ std::string("--device-clock=") + device_clock_name(clock) }).device_clock == clock);
    }
}

TEST_CASE("plugin args: --help or -h asks for the usage whatever else is given", "[unit][jmex][args]")
{
    REQUIRE_FALSE(parse_command_line({ "--help" }).has_value());
    REQUIRE_FALSE(parse_command_line({ "-h" }).has_value());
    REQUIRE_FALSE(parse_command_line({ "--chanel=2", "--help", "--channel=0" }).has_value());
}

TEST_CASE("plugin args: the usage names every flag", "[unit][jmex][args]")
{
    const std::string usage = plugin_usage();
    for (const char* flag : { "--channel=", "--collection-id=", "--bind-ip=", "--broadcast-ip=", "--tcp-port=",
                              "--udp-start-port=", "--device-clock=", "--help" })
    {
        CAPTURE(flag);
        REQUIRE_THAT(usage, ContainsSubstring(flag));
    }
}

TEST_CASE("plugin args: an empty collection id or address stops the plugin", "[unit][jmex][args]")
{
    REQUIRE_THROWS_WITH(parse({ "--collection-id=" }), ContainsSubstring("--collection-id="));
    REQUIRE_THROWS_WITH(parse({ "--bind-ip=" }), ContainsSubstring("--bind-ip="));
    REQUIRE_THROWS_WITH(parse({ "--broadcast-ip=" }), ContainsSubstring("--broadcast-ip="));
}

TEST_CASE("plugin args: a value that is not a whole number stops the plugin", "[unit][jmex][args]")
{
    REQUIRE_THROWS_WITH(parse({ "--channel=2.9" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel=" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel=+5" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel= 5" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--tcp-port=7000x" }), ContainsSubstring("--tcp-port="));
    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=port" }), ContainsSubstring("--udp-start-port="));
}

TEST_CASE("plugin args: a channel outside 0..255 stops the plugin", "[unit][jmex][args]")
{
    REQUIRE(parse({ "--channel=0" }).receiver.channel == 0);
    REQUIRE(parse({ "--channel=255" }).receiver.channel == 255);

    REQUIRE_THROWS_WITH(parse({ "--channel=-1" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel=256" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel=300" }), ContainsSubstring("--channel="));
}

TEST_CASE("plugin args: a port outside 1..65535 stops the plugin", "[unit][jmex][args]")
{
    REQUIRE(parse({ "--tcp-port=65535" }).receiver.tcp_port == 65535);

    REQUIRE_THROWS_WITH(parse({ "--tcp-port=0" }), ContainsSubstring("--tcp-port="));
    REQUIRE_THROWS_WITH(parse({ "--tcp-port=70000" }), ContainsSubstring("--tcp-port="));
    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=0" }), ContainsSubstring("--udp-start-port="));
    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=99999999999" }), ContainsSubstring("--udp-start-port="));
}
