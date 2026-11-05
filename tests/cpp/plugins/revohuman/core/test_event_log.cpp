// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/event_log.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using plugins::revohuman::EventLog;

namespace
{

std::vector<std::string> lines_of(const std::filesystem::path& path)
{
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

} // namespace

TEST_CASE("event log is off for an empty path", "[revohuman][event_log]")
{
    EventLog log("");
    REQUIRE_FALSE(log.enabled());
    log.write("ignored", { { "a", std::int64_t{ 1 } } });
}

TEST_CASE("event log writes one JSON object per line and appends", "[revohuman][event_log]")
{
    const auto path = std::filesystem::temp_directory_path() / "revohuman_event_log_test.jsonl";
    std::filesystem::remove(path);
    {
        EventLog log(path.string());
        REQUIRE(log.enabled());
        log.write("injector_created", { { "hand", std::string("left") },
                                        { "epoch", std::uint64_t{ 7 } },
                                        { "dt", std::int64_t{ -3 } },
                                        { "tracked", true } });
        log.write("odd", { { "s", std::string("a\"b\\c\nd\x01") },
                           { "nan", std::numeric_limits<double>::quiet_NaN() },
                           { "x", 0.5 } });
    }
    {
        EventLog log(path.string());
        log.write("again", {});
    }
    const auto lines = lines_of(path);
    std::filesystem::remove(path);
    REQUIRE(lines.size() == 3);
    REQUIRE(lines[0].find("\"event\":\"injector_created\"") != std::string::npos);
    REQUIRE(lines[0].find("\"hand\":\"left\"") != std::string::npos);
    REQUIRE(lines[0].find("\"epoch\":7") != std::string::npos);
    REQUIRE(lines[0].find("\"dt\":-3") != std::string::npos);
    REQUIRE(lines[0].find("\"tracked\":true") != std::string::npos);
    REQUIRE(lines[1].find(R"("s":"a\"b\\c\nd\u0001")") != std::string::npos);
    REQUIRE(lines[1].find("\"nan\":null") != std::string::npos);
    REQUIRE(lines[1].find("\"x\":0.5") != std::string::npos);
    REQUIRE(lines[2].find("\"event\":\"again\"") != std::string::npos);
}
