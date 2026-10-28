// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <ego_camera/raw_calibration.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
constexpr std::array<uint8_t, 7> kCalibration{ 'c', 'a', 'm', 0, 'c', 'a', 'l' };
} // namespace

TEST_CASE("EGO raw calibration preserves every byte until DONE", "[ego][calibration][unit]")
{
    plugins::ego::detail::RawCalibrationTransfer transfer;
    SECTION("a single chunk")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
    }
    SECTION("multiple chunks with a short final chunk")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
        transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, 3);
        transfer.append(kCalibration.size(), 6, kCalibration.data() + 6, 1);
    }
    CHECK_THROWS_AS(transfer.finish(), std::runtime_error);
    transfer.complete();
    CHECK(transfer.finish() == std::string(kCalibration.begin(), kCalibration.end()));
}

TEST_CASE("EGO raw calibration requires DONE after a complete nonempty payload", "[ego][calibration][unit]")
{
    plugins::ego::detail::RawCalibrationTransfer transfer;
    SECTION("no callbacks")
    {
    }
    SECTION("DONE without data")
    {
        transfer.complete();
    }
    SECTION("missing DONE after all bytes")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
    }
    SECTION("DONE before the final bytes")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
        transfer.complete();
    }
    CHECK_THROWS_AS(transfer.finish(), std::runtime_error);
}

TEST_CASE("EGO raw calibration rejects invalid chunks without reading their payload", "[ego][calibration][unit]")
{
    plugins::ego::detail::RawCalibrationTransfer transfer;
    SECTION("empty transfer")
    {
        transfer.append(0, 0, nullptr, 0);
    }
    SECTION("empty chunk in a nonempty transfer")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 0);
    }
    SECTION("null payload with a positive size")
    {
        transfer.append(kCalibration.size(), 0, nullptr, 1);
    }
    SECTION("offset beyond the advertised total")
    {
        transfer.append(kCalibration.size(), kCalibration.size() + 1, kCalibration.data(), 1);
    }
    SECTION("size beyond the remaining bytes")
    {
        transfer.append(3, 0, kCalibration.data(), 4);
    }
    SECTION("maximum offset")
    {
        transfer.append(kCalibration.size(), std::numeric_limits<size_t>::max(), kCalibration.data(), 1);
    }
    SECTION("maximum size cannot wrap the bounds check")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), std::numeric_limits<size_t>::max());
    }
    transfer.complete();
    CHECK_THROWS_AS(transfer.finish(), std::runtime_error);
}

TEST_CASE("EGO raw calibration requires a stable length and contiguous chunks", "[ego][calibration][unit]")
{
    plugins::ego::detail::RawCalibrationTransfer transfer;
    transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
    SECTION("total length grows")
    {
        transfer.append(kCalibration.size() + 1, 3, kCalibration.data() + 3, 4);
    }
    SECTION("total length shrinks")
    {
        transfer.append(kCalibration.size() - 1, 3, kCalibration.data() + 3, 3);
    }
    SECTION("duplicate chunk")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
    }
    SECTION("overlapping chunk")
    {
        transfer.append(kCalibration.size(), 2, kCalibration.data() + 2, 3);
    }
    SECTION("a gap before the next chunk")
    {
        transfer.append(kCalibration.size(), 4, kCalibration.data() + 4, 3);
    }
    SECTION("size plus offset would overflow")
    {
        transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, std::numeric_limits<size_t>::max());
    }
    transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, 4);
    transfer.complete();
    CHECK_THROWS_AS(transfer.finish(), std::runtime_error);
}

TEST_CASE("EGO raw calibration cannot recover from failure or accept data after DONE", "[ego][calibration][unit]")
{
    plugins::ego::detail::RawCalibrationTransfer transfer;
    SECTION("SDK failure before data")
    {
        transfer.fail();
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
        transfer.complete();
    }
    SECTION("SDK failure after all data")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
        transfer.fail();
        transfer.complete();
    }
    SECTION("SDK failure after DONE")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
        transfer.complete();
        transfer.fail();
    }
    SECTION("chunks arrive out of order")
    {
        transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, 4);
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
        transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, 4);
        transfer.complete();
    }
    SECTION("data after early DONE")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), 3);
        transfer.complete();
        transfer.append(kCalibration.size(), 3, kCalibration.data() + 3, 4);
    }
    SECTION("data after a valid DONE")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
        transfer.complete();
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
    }
    SECTION("repeated DONE")
    {
        transfer.append(kCalibration.size(), 0, kCalibration.data(), kCalibration.size());
        transfer.complete();
        transfer.complete();
    }
    CHECK_THROWS_AS(transfer.finish(), std::runtime_error);
}
