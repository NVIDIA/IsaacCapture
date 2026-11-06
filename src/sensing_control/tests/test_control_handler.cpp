// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <sensing_control/control_handler.hpp>

using namespace plugins::sensing;
using Json = nlohmann::json;

namespace
{

class FakeActions : public ControlActions
{
public:
    ActionResult set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) override
    {
        ++calls;
        last_track = track;
        last_host = host;
        last_port = port;
        return result;
    }

    int calls = 0;
    std::string last_track;
    std::string last_host;
    std::uint16_t last_port = 0;
    ActionResult result;
};

struct Fixture
{
    Fixture() : status("test-release"), handler(status, actions)
    {
        StreamStatus left;
        left.track = "sensing-left";
        left.sensor_id = 3;
        StreamStatus right;
        right.track = "sensing-right";
        status.set_streams({ left, right });
    }

    Json call(const Json& request)
    {
        return Json::parse(handler.handle(request.dump()));
    }

    ProducerStatus status;
    FakeActions actions;
    ControlHandler handler;
};

} // namespace

TEST_CASE("ping answers pong and echoes the id", "[control_handler]")
{
    Fixture f;
    const auto reply = f.call({ { "id", 7 }, { "type", "ping" } });
    CHECK(reply["ok"] == true);
    CHECK(reply["type"] == "pong");
    CHECK(reply["id"] == 7);
}

TEST_CASE("a string id is echoed and a missing id is null", "[control_handler]")
{
    Fixture f;
    CHECK(f.call({ { "id", "abc" }, { "type", "ping" } })["id"] == "abc");
    CHECK(f.call({ { "type", "ping" } })["id"].is_null());
}

TEST_CASE("get_status reports the state, release and streams", "[control_handler]")
{
    Fixture f;
    f.status.set_state(ProducerState::CameraFault, Fault{ "camera_error", "check the power" });
    const auto reply = f.call({ { "id", 1 }, { "type", "get_status" } });
    REQUIRE(reply["ok"] == true);
    CHECK(reply["type"] == "status");
    const auto& status = reply["status"];
    CHECK(status["state"] == "camera_fault");
    CHECK(status["release"] == "test-release");
    CHECK(status["fault"]["code"] == "camera_error");
    CHECK(status["fault"]["message"] == "check the power");
    REQUIRE(status["streams"].size() == 2);
    CHECK(status["streams"][0]["stream"] == "sensing-left");
    CHECK(status["streams"][0]["sensor"] == 3);
    CHECK(status["streams"][0]["direct"]["connected"] == false);
}

TEST_CASE("the status has no fault key when healthy", "[control_handler]")
{
    Fixture f;
    f.status.set_state(ProducerState::Streaming);
    CHECK_FALSE(f.call({ { "type", "get_status" } })["status"].contains("fault"));
}

TEST_CASE("malformed requests are answered with a code, never dropped", "[control_handler]")
{
    Fixture f;

    SECTION("bad_json")
    {
        for (const char* text : { "not json", "", "[1,2]", "42", "\"str\"" })
        {
            const auto reply = Json::parse(f.handler.handle(text));
            CHECK(reply["ok"] == false);
            CHECK(reply["error"]["code"] == "bad_json");
            CHECK_FALSE(reply["error"]["message"].get<std::string>().empty());
        }
    }
    SECTION("missing_type")
    {
        CHECK(f.call({ { "id", 4 } })["error"]["code"] == "missing_type");
        const auto reply = f.call({ { "id", 5 }, { "type", 3 } });
        CHECK(reply["error"]["code"] == "missing_type");
        CHECK(reply["id"] == 5);
    }
    SECTION("unknown_type")
    {
        const auto reply = f.call({ { "id", 6 }, { "type", "reboot" } });
        CHECK(reply["ok"] == false);
        CHECK(reply["error"]["code"] == "unknown_type");
        CHECK(reply["error"]["message"].get<std::string>().find("reboot") != std::string::npos);
        CHECK(reply["id"] == 6);
    }
    CHECK(f.actions.calls == 0);
}

TEST_CASE("set_direct_target validation", "[control_handler]")
{
    Fixture f;
    const Json good = { { "id", 9 },
                        { "type", "set_direct_target" },
                        { "stream", "sensing-left" },
                        { "host", "192.168.1.205" },
                        { "port", 5600 } };

    const auto code_of = [&](Json request) { return f.call(request)["error"]["code"]; };

    SECTION("missing_stream")
    {
        Json r = good;
        r.erase("stream");
        CHECK(code_of(r) == "missing_stream");
        r["stream"] = 5;
        CHECK(code_of(r) == "missing_stream");
    }
    SECTION("missing_host")
    {
        Json r = good;
        r.erase("host");
        CHECK(code_of(r) == "missing_host");
        r["host"] = 5;
        CHECK(code_of(r) == "missing_host");
    }
    SECTION("invalid_host")
    {
        for (const char* host : { "http://x", "a b", "a/b", "-bad", ".bad", "host:5600", "a_b" })
        {
            Json r = good;
            r["host"] = host;
            CHECK(code_of(r) == "invalid_host");
        }
        Json r = good;
        r["host"] = std::string(254, 'a');
        CHECK(code_of(r) == "invalid_host");
    }
    SECTION("invalid_port")
    {
        Json r = good;
        r.erase("port");
        CHECK(code_of(r) == "invalid_port");
        for (const Json& port : { Json(0), Json(-1), Json(65536), Json("5600"), Json(5600.5), Json(nullptr) })
        {
            r["port"] = port;
            CHECK(code_of(r) == "invalid_port");
        }
    }
    SECTION("unknown_stream lists the real ones")
    {
        Json r = good;
        r["stream"] = "sensing-up";
        const auto reply = f.call(r);
        CHECK(reply["error"]["code"] == "unknown_stream");
        const std::string message = reply["error"]["message"];
        CHECK(message.find("sensing-up") != std::string::npos);
        CHECK(message.find("sensing-left") != std::string::npos);
        CHECK(message.find("sensing-right") != std::string::npos);
    }
    CHECK(f.actions.calls == 0);
}

TEST_CASE("set_direct_target forwards to the action", "[control_handler]")
{
    Fixture f;

    SECTION("success returns the status and echoes the id")
    {
        const auto reply = f.call({ { "id", 11 },
                                    { "type", "set_direct_target" },
                                    { "stream", "sensing-right" },
                                    { "host", "pc.example" },
                                    { "port", 65535 } });
        CHECK(reply["ok"] == true);
        CHECK(reply["id"] == 11);
        CHECK(reply["type"] == "status");
        CHECK(f.actions.calls == 1);
        CHECK(f.actions.last_track == "sensing-right");
        CHECK(f.actions.last_host == "pc.example");
        CHECK(f.actions.last_port == 65535);
    }
    SECTION("IPv4 and IPv6 literals are accepted")
    {
        for (const char* host : { "127.0.0.1", "::1", "fe80::1" })
        {
            const auto reply = f.call(
                { { "type", "set_direct_target" }, { "stream", "sensing-left" }, { "host", host }, { "port", 1 } });
            CHECK(reply["ok"] == true);
            CHECK(f.actions.last_host == host);
        }
    }
    SECTION("an empty host disables and needs no port")
    {
        const auto reply =
            f.call({ { "id", 12 }, { "type", "set_direct_target" }, { "stream", "sensing-left" }, { "host", "" } });
        CHECK(reply["ok"] == true);
        CHECK(f.actions.calls == 1);
        CHECK(f.actions.last_host.empty());
        CHECK(f.actions.last_port == 0);
    }
    SECTION("an action failure is passed through with its own code")
    {
        f.actions.result = { false, "busy", "the sender is restarting" };
        const auto reply = f.call({ { "id", 13 },
                                    { "type", "set_direct_target" },
                                    { "stream", "sensing-left" },
                                    { "host", "10.1.1.1" },
                                    { "port", 5600 } });
        CHECK(reply["ok"] == false);
        CHECK(reply["id"] == 13);
        CHECK(reply["error"]["code"] == "busy");
        CHECK(reply["error"]["message"] == "the sender is restarting");
    }
}

TEST_CASE("events", "[control_handler]")
{
    ProducerStatus status("r");
    status.set_state(ProducerState::Error, Fault{ "stream_failed", "gone" });
    const auto state = Json::parse(state_event(status.snapshot()));
    CHECK(state["type"] == "event");
    CHECK(state["event"] == "state_changed");
    CHECK(state["status"]["state"] == "error");
    CHECK(state["status"]["fault"]["code"] == "stream_failed");

    const auto error = Json::parse(error_event(Fault{ "c", "m" }));
    CHECK(error["event"] == "error");
    CHECK(error["error"]["code"] == "c");
    CHECK(error["error"]["message"] == "m");
}
