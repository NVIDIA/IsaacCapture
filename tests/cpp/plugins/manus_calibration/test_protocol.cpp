// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <manus_calibration/protocol.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

using namespace plugins::manus::calibration;
using Catch::Matchers::ContainsSubstring;

const StartCommand* as_start(const ParsedCommand& parsed)
{
    return parsed.command ? std::get_if<StartCommand>(&*parsed.command) : nullptr;
}

const ConfirmCommand* as_confirm(const ParsedCommand& parsed)
{
    return parsed.command ? std::get_if<ConfirmCommand>(&*parsed.command) : nullptr;
}

const CancelCommand* as_cancel(const ParsedCommand& parsed)
{
    return parsed.command ? std::get_if<CancelCommand>(&*parsed.command) : nullptr;
}

// Every outcome is exactly one of a command or a rejection.
void require_rejected(const std::string& line, const std::string& expected_command = "unknown")
{
    INFO("line: " << line.substr(0, 200));
    const ParsedCommand parsed = parse_command(line);
    REQUIRE_FALSE(parsed.command.has_value());
    REQUIRE(parsed.rejection.has_value());
    CHECK(parsed.rejection->reason == RejectReason::Invalid);
    CHECK(parsed.rejection->command == expected_command);
    CHECK_FALSE(parsed.rejection->message.empty());
}

std::string start_line(const std::string& attempt = "att1",
                       const std::string& sides = "[\"left\",\"right\"]",
                       const std::string& staging = "/stage")
{
    return "{\"v\":1,\"type\":\"start\",\"attempt\":\"" + attempt + "\",\"sides\":" + sides + ",\"staging_dir\":\"" +
           staging + "\"}";
}

std::string confirm_line(const std::string& token = "att1-1")
{
    return "{\"v\":1,\"type\":\"confirm\",\"attempt\":\"att1\",\"token\":\"" + token + "\"}";
}

} // namespace

TEST_CASE("valid commands parse into typed commands", "[unit][protocol][parse]")
{
    const ParsedCommand start = parse_command(start_line());
    REQUIRE(as_start(start));
    CHECK(start.command.has_value());
    CHECK_FALSE(start.rejection.has_value());
    CHECK(as_start(start)->attempt == "att1");
    CHECK(as_start(start)->sides == std::vector<Side>{ Side::Left, Side::Right });
    CHECK(as_start(start)->staging_dir == "/stage");

    const ParsedCommand confirm = parse_command(confirm_line("att1-7"));
    REQUIRE(as_confirm(confirm));
    CHECK(as_confirm(confirm)->attempt == "att1");
    CHECK(as_confirm(confirm)->token == "att1-7");

    const ParsedCommand cancel = parse_command("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"att1\"}");
    REQUIRE(as_cancel(cancel));
    CHECK(as_cancel(cancel)->attempt == "att1");
}

TEST_CASE("start sides are normalized to left then right", "[unit][protocol][parse]")
{
    CHECK(as_start(parse_command(start_line("a", "[\"right\",\"left\"]")))->sides ==
          std::vector<Side>{ Side::Left, Side::Right });
    CHECK(as_start(parse_command(start_line("a", "[\"right\"]")))->sides == std::vector<Side>{ Side::Right });
    CHECK(as_start(parse_command(start_line("a", "[\"left\"]")))->sides == std::vector<Side>{ Side::Left });
}

TEST_CASE("whitespace and a trailing carriage return around the object are tolerated", "[unit][protocol][parse]")
{
    CHECK(as_cancel(parse_command("  \t{ \"v\" : 1 , \"type\" : \"cancel\" , \"attempt\" : \"a\" }\r\n")));
    CHECK(as_cancel(parse_command("{\"attempt\":\"a\",\"type\":\"cancel\",\"v\":1}")));
}

TEST_CASE("start rejects bad sides", "[unit][protocol][parse]")
{
    require_rejected(start_line("a", "[]"), "start");
    require_rejected(start_line("a", "[\"left\",\"left\"]"), "start");
    require_rejected(start_line("a", "[\"left\",\"right\",\"left\"]"), "start");
    require_rejected(start_line("a", "[\"both\"]"), "start");
    require_rejected(start_line("a", "[\"Left\"]"), "start");
    require_rejected(start_line("a", "\"left\""), "start");
    // An array element that is not a string is a JSON-level failure, so no command name is known.
    require_rejected(start_line("a", "[1]"));
    require_rejected(start_line("a", "7"), "start");
    require_rejected("{\"v\":1,\"type\":\"start\",\"attempt\":\"a\",\"staging_dir\":\"/s\"}", "start");
}

TEST_CASE("start rejects a bad staging_dir", "[unit][protocol][parse]")
{
    require_rejected(start_line("a", "[\"left\"]", ""), "start");
    require_rejected(start_line("a", "[\"left\"]", "relative/dir"), "start");
    require_rejected(start_line("a", "[\"left\"]", "./dir"), "start");
    // An escaped NUL would truncate the path in C APIs.
    require_rejected(start_line("a", "[\"left\"]", "/stage\\u0000/x"), "start");
    require_rejected("{\"v\":1,\"type\":\"start\",\"attempt\":\"a\",\"sides\":[\"left\"]}", "start");
    require_rejected("{\"v\":1,\"type\":\"start\",\"attempt\":\"a\",\"sides\":[\"left\"],\"staging_dir\":5}", "start");
    require_rejected(start_line("a", "[\"left\"]", "/" + std::string(4096, 'x')), "start");

    CHECK(as_start(parse_command(start_line("a", "[\"left\"]", "/" + std::string(4095, 'x')))));
}

TEST_CASE("attempt ids are 1-64 characters of letters, digits, underscore and dash", "[unit][protocol][parse]")
{
    CHECK(as_cancel(parse_command("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"Aa09_-\"}")));
    CHECK(as_cancel(parse_command("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"" + std::string(64, 'a') + "\"}")));

    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"" + std::string(65, 'a') + "\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a b\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a/b\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a.b\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\\u0000\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\\u00e9\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\xC3\xA9\"}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":7}", "cancel");
    require_rejected("{\"v\":1,\"type\":\"cancel\"}", "cancel");
}

TEST_CASE("confirm tokens are 1-128 printable ASCII characters", "[unit][protocol][parse]")
{
    CHECK(as_confirm(parse_command(confirm_line("a"))));
    CHECK(as_confirm(parse_command(confirm_line(std::string(128, 'z')))));
    CHECK(as_confirm(parse_command(confirm_line("~!#$%"))));

    require_rejected(confirm_line(""), "confirm");
    require_rejected(confirm_line(std::string(129, 'z')), "confirm");
    require_rejected(confirm_line("with space"), "confirm");
    require_rejected(confirm_line("tab\\t"), "confirm");
    require_rejected(confirm_line("nul\\u0000"), "confirm");
    require_rejected(confirm_line("del\\u007f"), "confirm");
    require_rejected(confirm_line("caf\\u00e9"), "confirm");
    require_rejected("{\"v\":1,\"type\":\"confirm\",\"attempt\":\"a\"}", "confirm");
    require_rejected("{\"v\":1,\"type\":\"confirm\",\"attempt\":\"a\",\"token\":3}", "confirm");
}

TEST_CASE("a rejection for a known command echoes the attempt once it is valid", "[unit][protocol][parse]")
{
    const ParsedCommand bad_sides = parse_command(start_line("att9", "[]"));
    REQUIRE(bad_sides.rejection);
    CHECK(bad_sides.rejection->attempt == std::optional<std::string>("att9"));

    const ParsedCommand bad_token = parse_command(confirm_line("with space"));
    REQUIRE(bad_token.rejection);
    CHECK(bad_token.rejection->attempt == std::optional<std::string>("att1"));

    const ParsedCommand bad_attempt = parse_command(start_line("a b"));
    REQUIRE(bad_attempt.rejection);
    CHECK_FALSE(bad_attempt.rejection->attempt.has_value());
}

TEST_CASE("unknown keys are rejected for every command", "[unit][protocol][parse]")
{
    require_rejected(
        "{\"v\":1,\"type\":\"start\",\"attempt\":\"a\",\"sides\":[\"left\"],\"staging_dir\":\"/s\","
        "\"extra\":1}",
        "start");
    require_rejected("{\"v\":1,\"type\":\"confirm\",\"attempt\":\"a\",\"token\":\"t\",\"extra\":\"x\"}", "confirm");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\",\"token\":\"t\"}", "cancel");
}

TEST_CASE("the protocol version is required and must be the integer 1", "[unit][protocol][parse]")
{
    require_rejected("{\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    require_rejected("{\"v\":2,\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    require_rejected("{\"v\":0,\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    require_rejected("{\"v\":\"1\",\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    // Values the reader does not support at all fail before the command name is known.
    require_rejected("{\"v\":1.0,\"type\":\"cancel\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":true,\"type\":\"cancel\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":null,\"type\":\"cancel\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":[1],\"type\":\"cancel\",\"attempt\":\"a\"}");
}

TEST_CASE("the command type must be one of the three known strings", "[unit][protocol][parse]")
{
    require_rejected("{\"v\":1,\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":1,\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":null,\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":\"Start\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":\"start \",\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":\"ready\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":\"\",\"attempt\":\"a\"}");
}

TEST_CASE("malformed JSON is rejected, never crashes", "[unit][protocol][json]")
{
    const std::vector<std::string> lines = {
        "",
        " ",
        "\n",
        "null",
        "[]",
        "[{\"v\":1}]",
        "\"cancel\"",
        "42",
        "{",
        "}",
        "{\"v\":1",
        "{\"v\":1,}",
        "{,\"v\":1}",
        "{\"v\":1 \"type\":\"cancel\"}",
        "{'v':1,'type':'cancel','attempt':'a'}",
        "{v:1}",
        "{\"v\"}",
        "{\"v\":}",
        "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"} trailing",
        "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"}{\"v\":1}",
        "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"}}",
        "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a",
        "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\\",
        "\xEF\xBB\xBF{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"}",
    };
    for (const std::string& line : lines)
    {
        require_rejected(line);
    }
}

TEST_CASE("duplicate keys are rejected even when the values agree", "[unit][protocol][json]")
{
    require_rejected("{\"v\":1,\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\",\"attempt\":\"b\"}");
    // A key written with an escape is still the same key.
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\",\"att\\u0065mpt\":\"b\"}");
}

TEST_CASE("only strings, plain integers and string arrays are accepted as values", "[unit][protocol][json]")
{
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":{\"x\":1}}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":[1,2]}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":[[\"a\"]]}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":[\"a\",]}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":[\"a\" \"b\"]}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":true}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":false}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":null}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":NaN}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":Infinity}");
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":-Infinity}");
}

TEST_CASE("numbers must be plain integers", "[unit][protocol][json]")
{
    require_rejected("{\"v\":1e400,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":1E0,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":1.5,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":1.,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":01,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":-,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":+1,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":0x1,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    // Nineteen digits would not fit the reader's int64 guard.
    require_rejected("{\"v\":1000000000000000000,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    require_rejected("{\"v\":99999999999999999999999999,\"type\":\"cancel\",\"attempt\":\"a\"}", "unknown");
    // Any accepted integer that is not 1 is still the wrong version.
    require_rejected("{\"v\":-1,\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    require_rejected("{\"v\":-0,\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
    require_rejected("{\"v\":999999999999999999,\"type\":\"cancel\",\"attempt\":\"a\"}", "cancel");
}

TEST_CASE("string escapes decode and invalid ones are rejected", "[unit][protocol][json]")
{
    // All the standard escapes decode inside a path.
    const ParsedCommand escaped = parse_command(start_line("a", "[\"left\"]", "/a\\/b\\\\c\\\"d\\be\\ff\\ng\\rh\\ti"));
    REQUIRE(as_start(escaped));
    CHECK(as_start(escaped)->staging_dir == std::string("/a/b\\c\"d\be\ff\ng\rh\ti"));

    require_rejected(start_line("a", "[\"left\"]", "/bad\\q"));
    require_rejected(start_line("a", "[\"left\"]", "/bad\\x41"));
    require_rejected(start_line("a", "[\"left\"]", "/bad\\u12"));
    require_rejected(start_line("a", "[\"left\"]", "/bad\\u12G4"));
    require_rejected(start_line("a", "[\"left\"]", "/bad\\U0041"));
}

TEST_CASE("unicode escapes decode to UTF-8 including surrogate pairs", "[unit][protocol][json]")
{
    const ParsedCommand two_byte = parse_command(start_line("a", "[\"left\"]", "/\\u00e9"));
    REQUIRE(as_start(two_byte));
    CHECK(as_start(two_byte)->staging_dir == "/\xC3\xA9");

    const ParsedCommand three_byte = parse_command(start_line("a", "[\"left\"]", "/\\u20AC"));
    REQUIRE(as_start(three_byte));
    CHECK(as_start(three_byte)->staging_dir == "/\xE2\x82\xAC");

    const ParsedCommand pair = parse_command(start_line("a", "[\"left\"]", "/\\ud83d\\ude00"));
    REQUIRE(as_start(pair));
    CHECK(as_start(pair)->staging_dir == "/\xF0\x9F\x98\x80");

    const ParsedCommand upper = parse_command(start_line("a", "[\"left\"]", "/\\uD83D\\uDE00"));
    REQUIRE(as_start(upper));
    CHECK(as_start(upper)->staging_dir == "/\xF0\x9F\x98\x80");

    // Raw multi-byte UTF-8 passes through unchanged.
    const ParsedCommand raw = parse_command(start_line("a", "[\"left\"]", "/caf\xC3\xA9"));
    REQUIRE(as_start(raw));
    CHECK(as_start(raw)->staging_dir == "/caf\xC3\xA9");
}

TEST_CASE("unpaired surrogates are rejected", "[unit][protocol][json]")
{
    require_rejected(start_line("a", "[\"left\"]", "/\\ud83d"));
    require_rejected(start_line("a", "[\"left\"]", "/\\ud83dx"));
    require_rejected(start_line("a", "[\"left\"]", "/\\ud83d\\u0041"));
    require_rejected(start_line("a", "[\"left\"]", "/\\ud83d\\ud83d"));
    require_rejected(start_line("a", "[\"left\"]", "/\\ude00"));
    require_rejected(start_line("a", "[\"left\"]", "/\\ude00\\ud83d"));
}

TEST_CASE("raw control characters inside a string are rejected", "[unit][protocol][json]")
{
    require_rejected(start_line("a", "[\"left\"]", std::string("/bad") + '\x01' + "x"));
    require_rejected(start_line("a", "[\"left\"]", std::string("/bad") + '\n' + "x"));
    require_rejected(start_line("a", "[\"left\"]", std::string("/bad") + '\t' + "x"));
    require_rejected(start_line("a", "[\"left\"]", std::string("/bad") + '\0' + "x"));
}

TEST_CASE("deeply nested input is rejected without recursion problems", "[unit][protocol][json]")
{
    const std::string prefix = "{\"v\":1,\"type\":\"cancel\",\"attempt\":";
    std::string nested_objects;
    for (size_t i = 0; i < 10000; ++i)
    {
        nested_objects += "{\"a\":";
    }

    require_rejected(prefix + std::string(60000, '[') + "}");
    require_rejected(prefix + nested_objects);
    require_rejected(std::string(60000, '{'));
    // Past the size limit the line is refused before any parsing.
    require_rejected(prefix + std::string(200000, '[') + std::string(200000, ']') + "}");
}

TEST_CASE("the size limit applies to the whole line", "[unit][protocol][limits]")
{
    const std::string base = "{\"v\":1,\"type\":\"cancel\",\"attempt\":\"a\"}";

    // Whitespace padding keeps the line valid, so the limit is the only thing under test.
    std::string at_limit = base + std::string(kMaxMessageBytes - base.size(), ' ');
    REQUIRE(at_limit.size() == kMaxMessageBytes);
    CHECK(as_cancel(parse_command(at_limit)));

    std::string over = at_limit + " ";
    const ParsedCommand parsed = parse_command(over);
    REQUIRE(parsed.rejection);
    CHECK_FALSE(parsed.command.has_value());
    CHECK_THAT(parsed.rejection->message, ContainsSubstring("size limit"));
    CHECK(parsed.rejection->reason == RejectReason::Invalid);

    require_rejected(std::string(10 * kMaxMessageBytes, 'x'));
}

TEST_CASE("a very long string value is rejected by the field limits, not accepted", "[unit][protocol][limits]")
{
    require_rejected("{\"v\":1,\"type\":\"cancel\",\"attempt\":\"" + std::string(60000, 'a') + "\"}", "cancel");
    require_rejected(confirm_line(std::string(60000, 't')), "confirm");
}

// --- serialization -------------------------------------------------------------------------------------------

TEST_CASE("ready serializes with and without an SDK version", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(ReadyEvent{ "1.0", std::nullopt }) ==
          "{\"v\":1,\"type\":\"ready\",\"worker_version\":\"1.0\",\"sdk_version\":null}");
    CHECK(serialize_event(ReadyEvent{ "1.0", std::string("5.0.1") }) ==
          "{\"v\":1,\"type\":\"ready\",\"worker_version\":\"1.0\",\"sdk_version\":\"5.0.1\"}");
}

TEST_CASE("status serializes phases and the discovered gloves", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(StatusEvent{ "a1", Phase::AwaitingStep, { { Side::Left, 10 }, { Side::Right, 20 } } }) ==
          "{\"v\":1,\"type\":\"status\",\"attempt\":\"a1\",\"phase\":\"awaiting_step\",\"gloves\":"
          "[{\"side\":\"left\",\"glove_id\":10},{\"side\":\"right\",\"glove_id\":20}]}");
    CHECK(serialize_event(StatusEvent{ "a1", Phase::Discovering, {} }) ==
          "{\"v\":1,\"type\":\"status\",\"attempt\":\"a1\",\"phase\":\"discovering\",\"gloves\":[]}");

    const std::vector<std::pair<Phase, std::string>> phases = {
        { Phase::Discovering, "discovering" },
        { Phase::AwaitingStep, "awaiting_step" },
        { Phase::CapturingStep, "capturing_step" },
        { Phase::Saving, "saving" },
        { Phase::Stopping, "stopping" },
    };
    for (const auto& [phase, name] : phases)
    {
        CHECK_THAT(serialize_event(StatusEvent{ "a", phase, {} }), ContainsSubstring("\"phase\":\"" + name + "\""));
    }
}

TEST_CASE("prompt serializes every field", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(PromptEvent{ "a1", "a1-2", Side::Right, 20, 1, 3, "Pose", "Hold flat", StepMode::Motion, 2.5 }) ==
          "{\"v\":1,\"type\":\"prompt\",\"attempt\":\"a1\",\"token\":\"a1-2\",\"side\":\"right\",\"glove_id\":20,"
          "\"index\":1,\"count\":3,\"title\":\"Pose\",\"instruction\":\"Hold flat\",\"mode\":\"motion\","
          "\"duration_s\":2.500,\"continuous\":false,\"estimated_s\":0.000}");
    CHECK(serialize_event(PromptEvent{ "a1", "a1-1", Side::Left, 4294967295U, 0, 1, "", "", StepMode::Hold, 0.0 }) ==
          "{\"v\":1,\"type\":\"prompt\",\"attempt\":\"a1\",\"token\":\"a1-1\",\"side\":\"left\","
          "\"glove_id\":4294967295,\"index\":0,\"count\":1,\"title\":\"\",\"instruction\":\"\",\"mode\":\"hold\","
          "\"duration_s\":0.000,\"continuous\":false,\"estimated_s\":0.000}");
}

TEST_CASE("a continuous prompt serializes the flag and the estimate", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(PromptEvent{ "a1", "a1-3", Side::Left, 7, 2, 8, "Open and close", "Repeat", StepMode::Hold,
                                       0.0, true, 10.0 }) ==
          "{\"v\":1,\"type\":\"prompt\",\"attempt\":\"a1\",\"token\":\"a1-3\",\"side\":\"left\",\"glove_id\":7,"
          "\"index\":2,\"count\":8,\"title\":\"Open and close\",\"instruction\":\"Repeat\",\"mode\":\"hold\","
          "\"duration_s\":0.000,\"continuous\":true,\"estimated_s\":10.000}");
    // The estimate keeps three decimals, so a sub-second value is not rounded away.
    CHECK_THAT(serialize_event(PromptEvent{ "a", "t", Side::Right, 1, 0, 1, "", "", StepMode::Hold, 0.0, true, 0.25 }),
               ContainsSubstring("\"continuous\":true,\"estimated_s\":0.250}"));
}

TEST_CASE("step_accepted and step_progress serialize", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(StepAcceptedEvent{ "a1", "a1-1" }) ==
          "{\"v\":1,\"type\":\"step_accepted\",\"attempt\":\"a1\",\"token\":\"a1-1\"}");
    CHECK(serialize_event(StepProgressEvent{ "a1", "a1-2", 1.2344 }) ==
          "{\"v\":1,\"type\":\"step_progress\",\"attempt\":\"a1\",\"token\":\"a1-2\",\"elapsed_s\":1.234}");
    CHECK_THAT(serialize_event(StepProgressEvent{ "a1", "t", 0.0004 }), ContainsSubstring("\"elapsed_s\":0.000"));
    CHECK_THAT(serialize_event(StepProgressEvent{ "a1", "t", 2.0 }), ContainsSubstring("\"elapsed_s\":2.000"));
}

TEST_CASE("rejected serializes a null attempt and an optional token", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(RejectedEvent{ std::nullopt, "unknown", RejectReason::Invalid, "bad", std::nullopt }) ==
          "{\"v\":1,\"type\":\"rejected\",\"attempt\":null,\"command\":\"unknown\",\"reason\":\"invalid\","
          "\"message\":\"bad\"}");
    CHECK(serialize_event(
              RejectedEvent{ std::string("a1"), "confirm", RejectReason::StaleToken, "old", std::string("a1-9") }) ==
          "{\"v\":1,\"type\":\"rejected\",\"attempt\":\"a1\",\"command\":\"confirm\",\"reason\":\"stale_token\","
          "\"message\":\"old\",\"token\":\"a1-9\"}");

    const std::vector<std::pair<RejectReason, std::string>> reasons = {
        { RejectReason::Invalid, "invalid" },
        { RejectReason::UnknownAttempt, "unknown_attempt" },
        { RejectReason::StaleToken, "stale_token" },
        { RejectReason::Unexpected, "unexpected" },
        { RejectReason::Busy, "busy" },
    };
    for (const auto& [reason, name] : reasons)
    {
        CHECK_THAT(serialize_event(RejectedEvent{ std::nullopt, "start", reason, "m", std::nullopt }),
                   ContainsSubstring("\"reason\":\"" + name + "\""));
    }
}

TEST_CASE("exported serializes the file facts", "[unit][protocol][serialize]")
{
    CHECK(serialize_event(ExportedEvent{ "a1", Side::Left, 10, "a1_left.mcal", 5, std::string(64, 'a') }) ==
          "{\"v\":1,\"type\":\"exported\",\"attempt\":\"a1\",\"side\":\"left\",\"glove_id\":10,"
          "\"file\":\"a1_left.mcal\",\"bytes\":5,\"sha256\":\"" +
              std::string(64, 'a') + "\"}");
    CHECK_THAT(serialize_event(ExportedEvent{ "a1", Side::Right, 1, "f", 18446744073709551615ULL, "h" }),
               ContainsSubstring("\"bytes\":18446744073709551615"));
}

TEST_CASE("result lists files only for a completed outcome", "[unit][protocol][serialize]")
{
    const std::vector<ResultFile> files = { { Side::Left, "a1_left.mcal", 5, "h1" },
                                            { Side::Right, "a1_right.mcal", 7, "h2" } };

    ResultEvent completed;
    completed.attempt = "a1";
    completed.outcome = Outcome::Completed;
    completed.files = files;
    CHECK(serialize_event(completed) ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":\"a1\",\"outcome\":\"completed\",\"files\":"
          "[{\"side\":\"left\",\"file\":\"a1_left.mcal\",\"bytes\":5,\"sha256\":\"h1\"},"
          "{\"side\":\"right\",\"file\":\"a1_right.mcal\",\"bytes\":7,\"sha256\":\"h2\"}],\"released\":true}");

    ResultEvent empty_completed = completed;
    empty_completed.files.clear();
    CHECK_THAT(serialize_event(empty_completed), ContainsSubstring("\"files\":[]"));

    // A cancelled or failed result never advertises files, even if the struct still holds some.
    ResultEvent cancelled = completed;
    cancelled.outcome = Outcome::Cancelled;
    CHECK(serialize_event(cancelled) ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":\"a1\",\"outcome\":\"cancelled\",\"released\":true}");

    ResultEvent failed = completed;
    failed.outcome = Outcome::Failed;
    failed.error = ErrorInfo{ ErrorCode::ExportFailed, "disk full", Side::Right, 2U };
    const std::string failed_line = serialize_event(failed);
    CHECK(failed_line ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":\"a1\",\"outcome\":\"failed\",\"error\":{\"code\":"
          "\"export_failed\",\"message\":\"disk full\",\"side\":\"right\",\"step\":2},\"released\":true}");
    CHECK_THAT(failed_line, !ContainsSubstring("files"));
}

TEST_CASE("result always states whether the gloves were released, last in the object", "[unit][protocol][serialize]")
{
    ResultEvent fresh;
    CHECK(fresh.released);

    ResultEvent failed;
    failed.attempt = "a1";
    failed.outcome = Outcome::Failed;
    failed.error = ErrorInfo{ ErrorCode::SdkError, "cannot shut down the SDK: x", std::nullopt, std::nullopt };
    failed.released = false;
    CHECK(serialize_event(failed) ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":\"a1\",\"outcome\":\"failed\",\"error\":{\"code\":"
          "\"sdk_error\",\"message\":\"cannot shut down the SDK: x\"},\"released\":false}");

    // The flag is independent of the outcome: a cancelled result can also report an unreleased SDK.
    ResultEvent cancelled;
    cancelled.attempt = std::nullopt;
    cancelled.outcome = Outcome::Cancelled;
    cancelled.released = false;
    CHECK(serialize_event(cancelled) ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":null,\"outcome\":\"cancelled\",\"released\":false}");

    ResultEvent completed;
    completed.attempt = "a1";
    completed.outcome = Outcome::Completed;
    completed.released = false;
    CHECK_THAT(serialize_event(completed), ContainsSubstring("\"released\":false}"));
}

TEST_CASE("result serializes a missing attempt and error without side or step", "[unit][protocol][serialize]")
{
    ResultEvent result;
    result.attempt = std::nullopt;
    result.outcome = Outcome::Failed;
    result.error = ErrorInfo{ ErrorCode::SdkInitFailed, "no sdk", std::nullopt, std::nullopt };
    CHECK(serialize_event(result) ==
          "{\"v\":1,\"type\":\"result\",\"attempt\":null,\"outcome\":\"failed\",\"error\":{\"code\":"
          "\"sdk_init_failed\",\"message\":\"no sdk\"},\"released\":true}");

    const std::vector<std::pair<ErrorCode, std::string>> codes = {
        { ErrorCode::SdkInitFailed, "sdk_init_failed" },
        { ErrorCode::NoGloves, "no_gloves" },
        { ErrorCode::GloveMissing, "glove_missing" },
        { ErrorCode::GloveReplaced, "glove_replaced" },
        { ErrorCode::SdkError, "sdk_error" },
        { ErrorCode::StepFailed, "step_failed" },
        { ErrorCode::ExportFailed, "export_failed" },
        { ErrorCode::ProtocolError, "protocol_error" },
        { ErrorCode::Internal, "internal" },
    };
    for (const auto& [code, name] : codes)
    {
        result.error = ErrorInfo{ code, "m", std::nullopt, std::nullopt };
        CHECK_THAT(serialize_event(result), ContainsSubstring("\"code\":\"" + name + "\""));
    }
}

TEST_CASE("serialized strings escape quotes, backslashes and control characters", "[unit][protocol][serialize]")
{
    const auto line_for = [](const std::string& text) {
        return serialize_event(RejectedEvent{ std::nullopt, "start", RejectReason::Invalid, text, std::nullopt });
    };

    CHECK_THAT(line_for("a\"b"), ContainsSubstring("\"message\":\"a\\\"b\""));
    CHECK_THAT(line_for("a\\b"), ContainsSubstring("\"message\":\"a\\\\b\""));
    CHECK_THAT(line_for("a\nb"), ContainsSubstring("\"message\":\"a\\nb\""));
    CHECK_THAT(line_for("a\rb"), ContainsSubstring("\"message\":\"a\\rb\""));
    CHECK_THAT(line_for("a\tb"), ContainsSubstring("\"message\":\"a\\tb\""));
    CHECK_THAT(line_for(std::string("a") + '\x01' + "b"), ContainsSubstring("\"message\":\"a\\u0001b\""));
    CHECK_THAT(line_for(std::string("a") + '\x1f' + "b"), ContainsSubstring("\"message\":\"a\\u001fb\""));
    CHECK_THAT(line_for(std::string("a") + '\b' + "b"), ContainsSubstring("\"message\":\"a\\u0008b\""));
    CHECK_THAT(line_for(std::string("a") + '\f' + "b"), ContainsSubstring("\"message\":\"a\\u000cb\""));
    CHECK_THAT(line_for(std::string("a") + '\0' + "b"), ContainsSubstring("\"message\":\"a\\u0000b\""));
    // DEL is not a JSON control character and passes through.
    CHECK_THAT(
        line_for(std::string("a") + '\x7f' + "b"), ContainsSubstring(std::string("\"message\":\"a") + '\x7f' + "b\""));
}

TEST_CASE("a serialized event is always one line", "[unit][protocol][serialize]")
{
    const std::string nasty = std::string("line1\nline2\r\n") + '\0' + "\x01\x02 \"q\" \\ \xE2\x80\xA8 end";
    const std::vector<Event> events = {
        PromptEvent{ "a", nasty, Side::Left, 1, 0, 1, nasty, nasty, StepMode::Hold, 0.0 },
        RejectedEvent{ nasty, nasty, RejectReason::Invalid, nasty, nasty },
        StatusEvent{ nasty, Phase::Saving, {} },
    };
    for (const Event& event : events)
    {
        const std::string line = serialize_event(event);
        CHECK(line.find('\n') == std::string::npos);
        CHECK(line.find('\r') == std::string::npos);
        CHECK(line.find('\0') == std::string::npos);
        CHECK(line.front() == '{');
        CHECK(line.back() == '}');
    }
}

TEST_CASE("valid multi-byte UTF-8 is preserved when serialized", "[unit][protocol][serialize]")
{
    const auto message_of = [](const std::string& text) {
        return serialize_event(RejectedEvent{ std::nullopt, "start", RejectReason::Invalid, text, std::nullopt });
    };

    CHECK_THAT(message_of("caf\xC3\xA9"), ContainsSubstring("\"message\":\"caf\xC3\xA9\""));
    CHECK_THAT(message_of("\xE2\x82\xAC"), ContainsSubstring("\"message\":\"\xE2\x82\xAC\""));
    CHECK_THAT(message_of("\xF0\x9F\x98\x80"), ContainsSubstring("\"message\":\"\xF0\x9F\x98\x80\""));
    // The boundary code points of each sequence length.
    CHECK_THAT(message_of("\xC2\x80"), ContainsSubstring("\"message\":\"\xC2\x80\""));
    CHECK_THAT(message_of("\xDF\xBF"), ContainsSubstring("\"message\":\"\xDF\xBF\""));
    CHECK_THAT(message_of("\xE0\xA0\x80"), ContainsSubstring("\"message\":\"\xE0\xA0\x80\""));
    CHECK_THAT(message_of("\xEF\xBF\xBF"), ContainsSubstring("\"message\":\"\xEF\xBF\xBF\""));
    CHECK_THAT(message_of("\xF0\x90\x80\x80"), ContainsSubstring("\"message\":\"\xF0\x90\x80\x80\""));
    CHECK_THAT(message_of("\xF4\x8F\xBF\xBF"), ContainsSubstring("\"message\":\"\xF4\x8F\xBF\xBF\""));
}

TEST_CASE("invalid UTF-8 becomes the replacement character when serialized", "[unit][protocol][serialize]")
{
    const auto message_of = [](const std::string& text) {
        return serialize_event(RejectedEvent{ std::nullopt, "start", RejectReason::Invalid, text, std::nullopt });
    };
    const std::string replacement = "\xEF\xBF\xBD";

    // A lone continuation byte, a bare lead byte, and a truncated sequence.
    CHECK_THAT(message_of("a\x80z"), ContainsSubstring("\"message\":\"a" + replacement + "z\""));
    CHECK_THAT(message_of("a\xC3z"), ContainsSubstring("\"message\":\"a" + replacement + "z\""));
    CHECK_THAT(message_of("a\xE2\x82"), ContainsSubstring("\"message\":\"a" + replacement + replacement + "\""));
    // Overlong forms, surrogates and code points past U+10FFFF are not valid UTF-8.
    CHECK_THAT(message_of("\xC0\x80"), ContainsSubstring("\"message\":\"" + replacement + replacement + "\""));
    CHECK_THAT(message_of("\xC1\xBF"), ContainsSubstring("\"message\":\"" + replacement + replacement + "\""));
    CHECK_THAT(message_of("\xE0\x80\x80"),
               ContainsSubstring("\"message\":\"" + replacement + replacement + replacement + "\""));
    CHECK_THAT(message_of("\xED\xA0\x80"),
               ContainsSubstring("\"message\":\"" + replacement + replacement + replacement + "\""));
    CHECK_THAT(message_of("\xF4\x90\x80\x80"),
               ContainsSubstring("\"message\":\"" + replacement + replacement + replacement + replacement + "\""));
    CHECK_THAT(message_of("\xF8\x88\x80\x80\x80"), ContainsSubstring(replacement));
    CHECK_THAT(message_of("\xFF\xFE"), ContainsSubstring("\"message\":\"" + replacement + replacement + "\""));
}

TEST_CASE("non-finite seconds never serialize as bare nan or inf", "[unit][protocol][serialize]")
{
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<std::string> lines = {
        serialize_event(StepProgressEvent{ "a", "t", std::nan("") }),
        serialize_event(StepProgressEvent{ "a", "t", inf }),
        serialize_event(StepProgressEvent{ "a", "t", -inf }),
        serialize_event(PromptEvent{ "a", "t", Side::Left, 1, 0, 1, "", "", StepMode::Motion, inf }),
    };
    for (const std::string& line : lines)
    {
        INFO("line: " << line);
        CHECK_THAT(line, !ContainsSubstring("nan"));
        CHECK_THAT(line, !ContainsSubstring("inf"));
    }
}
