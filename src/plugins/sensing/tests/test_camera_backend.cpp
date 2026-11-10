// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "core/inc/sensing_core/camera_backend.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using plugins::sensing::BackendProbe;
using plugins::sensing::CameraBackend;
using plugins::sensing::FrameLease;
using plugins::sensing::parse_camera_backend;
using plugins::sensing::select_camera_backend;

namespace
{

BackendProbe probe(CameraBackend backend, bool available, bool matches, std::string detail = "")
{
    BackendProbe p;
    p.backend = backend;
    p.available = available;
    p.matches = matches;
    p.detail = std::move(detail);
    return p;
}

std::string error_of(CameraBackend requested, const std::vector<BackendProbe>& probes)
{
    try
    {
        select_camera_backend(requested, probes);
    }
    catch (const std::runtime_error& e)
    {
        return e.what();
    }
    FAIL("select_camera_backend did not throw");
    return {};
}

} // namespace

TEST_CASE("parse_camera_backend accepts the three names and nothing else", "[sensing][backend]")
{
    CHECK(parse_camera_backend("auto") == CameraBackend::Auto);
    CHECK(parse_camera_backend("argus") == CameraBackend::Argus);
    CHECK(parse_camera_backend("sipl") == CameraBackend::Sipl);
    CHECK_FALSE(parse_camera_backend("garbage"));
    CHECK_FALSE(parse_camera_backend(""));
    CHECK_FALSE(parse_camera_backend("SIPL"));
}

TEST_CASE("Auto picks the only matching backend", "[sensing][backend]")
{
    CHECK(select_camera_backend(CameraBackend::Auto, { probe(CameraBackend::Argus, true, true),
                                                       probe(CameraBackend::Sipl, true, false, "no pipelines") }) ==
          CameraBackend::Argus);
    CHECK(select_camera_backend(CameraBackend::Auto, { probe(CameraBackend::Argus, false, false, "no libargus"),
                                                       probe(CameraBackend::Sipl, true, true) }) == CameraBackend::Sipl);
}

TEST_CASE("Auto with no match throws and names the override", "[sensing][backend]")
{
    const auto message = error_of(CameraBackend::Auto, { probe(CameraBackend::Argus, false, false, "no libargus"),
                                                         probe(CameraBackend::Sipl, true, false, "no pipelines") });
    CHECK_THAT(message, ContainsSubstring("--camera-backend"));
    CHECK_THAT(message, ContainsSubstring("no libargus"));
    CHECK_THAT(message, ContainsSubstring("no pipelines"));
}

TEST_CASE("Auto with both matching throws and names the override", "[sensing][backend]")
{
    const auto message = error_of(
        CameraBackend::Auto, { probe(CameraBackend::Argus, true, true), probe(CameraBackend::Sipl, true, true) });
    CHECK_THAT(message, ContainsSubstring("both"));
    CHECK_THAT(message, ContainsSubstring("--camera-backend"));
}

TEST_CASE("An explicit request that matches is returned", "[sensing][backend]")
{
    const std::vector<BackendProbe> both = { probe(CameraBackend::Argus, true, true),
                                             probe(CameraBackend::Sipl, true, true) };
    CHECK(select_camera_backend(CameraBackend::Argus, both) == CameraBackend::Argus);
    CHECK(select_camera_backend(CameraBackend::Sipl, both) == CameraBackend::Sipl);
}

TEST_CASE("An explicit request that does not match throws with its own detail and never falls back", "[sensing][backend]")
{
    const auto message = error_of(CameraBackend::Sipl, { probe(CameraBackend::Argus, true, true),
                                                         probe(CameraBackend::Sipl, true, false, "sipl found nothing") });
    CHECK_THAT(message, ContainsSubstring("sipl found nothing"));
    CHECK_THAT(message, ContainsSubstring("--camera-backend=sipl"));
}

TEST_CASE("A probe that matches but is not available is never selected", "[sensing][backend]")
{
    const std::vector<BackendProbe> probes = { probe(CameraBackend::Argus, false, true, "argus unavailable") };
    CHECK_THROWS_AS(select_camera_backend(CameraBackend::Auto, probes), std::runtime_error);
    CHECK_THROWS_AS(select_camera_backend(CameraBackend::Argus, probes), std::runtime_error);
}

TEST_CASE("An explicit request for a backend missing from the probes throws", "[sensing][backend]")
{
    const auto message = error_of(CameraBackend::Argus, { probe(CameraBackend::Sipl, true, true) });
    CHECK_THAT(message, ContainsSubstring("--camera-backend=argus"));
}

TEST_CASE("FrameLease releases once on destruction", "[sensing][lease]")
{
    int released = 0;
    {
        FrameLease lease([&] { ++released; });
        CHECK(released == 0);
    }
    CHECK(released == 1);
}

TEST_CASE("A moved-from FrameLease does not release", "[sensing][lease]")
{
    int released = 0;
    {
        FrameLease a([&] { ++released; });
        {
            FrameLease b(std::move(a));
        }
        CHECK(released == 1);
    }
    CHECK(released == 1);
}

TEST_CASE("FrameLease reset releases once, then not again", "[sensing][lease]")
{
    int released = 0;
    {
        FrameLease lease([&] { ++released; });
        lease.reset();
        CHECK(released == 1);
        lease.reset();
        CHECK(released == 1);
    }
    CHECK(released == 1);
}

TEST_CASE("FrameLease move-assign releases the previous target first", "[sensing][lease]")
{
    std::vector<int> order;
    {
        FrameLease a([&] { order.push_back(1); });
        FrameLease b([&] { order.push_back(2); });
        a = std::move(b);
        REQUIRE(order == std::vector<int>{ 1 });
    }
    CHECK(order == std::vector<int>{ 1, 2 });
}

TEST_CASE("A default FrameLease releases nothing", "[sensing][lease]")
{
    FrameLease lease;
    lease.reset();
}
