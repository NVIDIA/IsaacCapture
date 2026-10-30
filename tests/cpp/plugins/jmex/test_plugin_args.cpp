// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the plugin's command line.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <jmex/plugin_args.hpp>

#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using namespace plugins::jmex;

namespace
{

//! Parse as the plugin would, with the PluginManager's injected root id ahead of the arguments.
PluginArgs parse(const std::vector<std::string>& args, const char* device_clock_env = nullptr)
{
    std::vector<const char*> argv{ "jmex_plugin", "--plugin-root-id=jmex" };
    for (const std::string& arg : args)
    {
        argv.push_back(arg.c_str());
    }
    return parse_plugin_args(static_cast<int>(argv.size()), argv.data(), device_clock_env);
}

} // namespace

TEST_CASE("plugin args: no flags give the defaults", "[unit][jmex][args]")
{
    const PluginArgs args = parse({});
    const ReceiverConfig defaults;
    REQUIRE(args.receiver.channel == defaults.channel);
    REQUIRE(args.receiver.tcp_port == defaults.tcp_port);
    REQUIRE(args.receiver.udp_start_port == defaults.udp_start_port);
    REQUIRE(args.receiver.bind_ip.empty());
    REQUIRE(args.receiver.broadcast_ip.empty());
    REQUIRE(args.collection_id == "jmex_full_body");
    REQUIRE(args.device_clock == DeviceClock::DEVICE);
}

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

TEST_CASE("plugin args: --device-clock= overrides JMEX_DEVICE_CLOCK", "[unit][jmex][args]")
{
    REQUIRE(parse({}, "local").device_clock == DeviceClock::LOCAL);
    REQUIRE(parse({ "--device-clock=device" }, "local").device_clock == DeviceClock::DEVICE);
}

TEST_CASE("plugin args: an argument it does not know stops the plugin", "[unit][jmex][args]")
{
    // A space instead of '=': "--channel" is unknown and "2" is positional, so channel 1 would stay.
    REQUIRE_THROWS_WITH(parse({ "--channel", "2" }), ContainsSubstring("'--channel'"));
    REQUIRE_THROWS_WITH(parse({ "--chanel=2" }), ContainsSubstring("'--chanel=2'"));
    REQUIRE_THROWS_WITH(parse({ "2" }), ContainsSubstring("'2'"));
}

TEST_CASE("plugin args: a value that is not a whole number stops the plugin", "[unit][jmex][args]")
{
    REQUIRE_THROWS_WITH(parse({ "--channel=2.9" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--channel=" }), ContainsSubstring("--channel="));
    REQUIRE_THROWS_WITH(parse({ "--tcp-port=7000x" }), ContainsSubstring("--tcp-port="));
    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=port" }), ContainsSubstring("--udp-start-port="));
}

TEST_CASE("plugin args: a channel outside 1..255 stops the plugin", "[unit][jmex][args]")
{
    REQUIRE(parse({ "--channel=1" }).receiver.channel == 1);
    REQUIRE(parse({ "--channel=11" }).receiver.channel == 11);
    REQUIRE(parse({ "--channel=255" }).receiver.channel == 255);

    REQUIRE_THROWS_WITH(parse({ "--channel=0" }), ContainsSubstring("--channel="));
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

TEST_CASE("plugin args: a channel data port past 65535 stops the plugin", "[unit][jmex][args]")
{
    // 65279 + 1 + 255 is the last port there is.
    REQUIRE(parse({ "--udp-start-port=65279", "--channel=255" }).receiver.udp_start_port == 65279);

    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=65280", "--channel=255" }), ContainsSubstring("65536"));
    REQUIRE_THROWS_WITH(parse({ "--channel=255", "--udp-start-port=65280" }), ContainsSubstring("65536"));
    REQUIRE_THROWS_WITH(parse({ "--udp-start-port=65535" }), ContainsSubstring("--udp-start-port="));
}
