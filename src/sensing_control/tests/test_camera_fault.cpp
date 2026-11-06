// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <sensing_control/camera_fault.hpp>

#include <string>

using plugins::sensing::diagnose_camera_failure;

namespace
{
bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}
} // namespace

TEST_CASE("known capture failures are explained and keep the original text", "[camera_fault]")
{
    const std::string sipl = "INvSIPLCamera::Init failed with SIPLStatus 10";
    const auto a = diagnose_camera_failure(sipl);
    CHECK(a.code == "camera_start_failed");
    CHECK(contains(a.message, "power cable"));
    CHECK(contains(a.message, "retries automatically"));
    CHECK(contains(a.message, sipl));

    const auto b = diagnose_camera_failure("GMSL link lock timed out");
    CHECK(b.code == "camera_link_not_locked");
    CHECK(contains(b.message, "GMSL"));
    CHECK(contains(b.message, "GMSL link lock timed out"));

    const auto c = diagnose_camera_failure("Failed to detect camera on link 2");
    CHECK(c.code == "camera_not_detected");
    CHECK(contains(c.message, "Failed to detect camera on link 2"));

    const auto d = diagnose_camera_failure("INvSIPLCamera::Init failed");
    CHECK(d.code == "camera_start_failed");
}

TEST_CASE("the most specific signature wins", "[camera_fault]")
{
    // Matches both "link lock" and "SIPLStatus 10"; the earlier table entry decides.
    CHECK(diagnose_camera_failure("link lock failed, SIPLStatus 10").code == "camera_link_not_locked");
}

TEST_CASE("an unknown failure gets the generic message with the original text", "[camera_fault]")
{
    const auto f = diagnose_camera_failure("something entirely new (errno 5)");
    CHECK(f.code == "camera_error");
    CHECK(contains(f.message, "could not be started"));
    CHECK(contains(f.message, "retries automatically"));
    CHECK(contains(f.message, "something entirely new (errno 5)"));

    const auto empty = diagnose_camera_failure("");
    CHECK(empty.code == "camera_error");
}
