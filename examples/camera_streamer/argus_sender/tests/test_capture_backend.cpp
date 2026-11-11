/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureBackend.h"
#include "CaptureFrame.h"
#include "TestHarness.h"

#include <string>

using namespace argus_sender;

int main()
{
    TestContext test("capture_backend");

    CaptureBackendKind kind = CAPTURE_BACKEND_AUTO;
    TEST_CHECK(test, parseCaptureBackend("sipl", &kind));
    TEST_CHECK_EQ(test, static_cast<int>(kind), static_cast<int>(CAPTURE_BACKEND_SIPL));
    TEST_CHECK(test, !parseCaptureBackend("SIPL", &kind));
    TEST_CHECK(test, !parseCaptureBackend("", &kind));

    BackendAvailability argusOnly;
    argusOnly.argus = true;
    BackendAvailability both;
    both.argus = both.sipl = true;
    BackendAvailability none;
    std::string error;
    CaptureBackendKind selected = CAPTURE_BACKEND_SIPL;

    TEST_CHECK(test, selectCaptureBackend(CAPTURE_BACKEND_AUTO, argusOnly, &selected, &error));
    TEST_CHECK_EQ(test, static_cast<int>(selected), static_cast<int>(CAPTURE_BACKEND_ARGUS));

    TEST_CHECK(test, !selectCaptureBackend(CAPTURE_BACKEND_SIPL, argusOnly, &selected, &error));
    TEST_CHECK(test, error.find("sipl") != std::string::npos && error.find("does not include") != std::string::npos);

    error.clear();
    TEST_CHECK(test, !selectCaptureBackend(CAPTURE_BACKEND_AUTO, both, &selected, &error));
    TEST_CHECK(test, error.find("capture_backend") != std::string::npos);

    TEST_CHECK(test, selectCaptureBackend(CAPTURE_BACKEND_SIPL, both, &selected, &error));
    TEST_CHECK_EQ(test, static_cast<int>(selected), static_cast<int>(CAPTURE_BACKEND_SIPL));

    TEST_CHECK(test, !selectCaptureBackend(CAPTURE_BACKEND_AUTO, none, &selected, &error));

    // A lease releases exactly once, on reset, destruction or after a move.
    int released = 0;
    {
        FrameLease a([&released]() { ++released; });
        TEST_CHECK(test, a.held());
        FrameLease b(std::move(a));
        TEST_CHECK(test, !a.held() && b.held());
        TEST_CHECK_EQ(test, released, 0);
        b.reset();
        TEST_CHECK_EQ(test, released, 1);
        b.reset();
        TEST_CHECK_EQ(test, released, 1);
        FrameLease c([&released]() { ++released; });
        FrameLease d([&released]() { released += 10; });
        d = std::move(c); // the lease d held is released by the assignment
        TEST_CHECK_EQ(test, released, 11);
    }
    TEST_CHECK_EQ(test, released, 12);

    TEST_CHECK_EQ(test, std::string(sensorClockDomainName(SENSOR_CLOCK_RAW_COUNTER)), std::string("raw_counter"));
    return test.finish();
}
