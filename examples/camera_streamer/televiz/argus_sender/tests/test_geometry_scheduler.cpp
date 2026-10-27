/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MixedRateScheduler.h"
#include "OutputGeometry.h"
#include "TestHarness.h"

#include <cmath>
#include <string>
#include <vector>

int main()
{
    TestContext test("geometry_scheduler");
    argus_sender::OutputConfig output;
    argus_sender::ResolvedOutputGeometry geometry;
    std::string error;
    std::vector<argus_sender::ImageSize> sources(1);
    sources[0].width = 2560;
    sources[0].height = 1984;

    TEST_CHECK(test, argus_sender::resolveOutputGeometry(output, sources, &geometry, &error));
    TEST_CHECK_EQ(test, geometry.width, 2560U);
    TEST_CHECK_EQ(test, geometry.height, 1984U);

    output.width = 1280;
    TEST_CHECK(test, argus_sender::resolveOutputGeometry(output, sources, &geometry, &error));
    TEST_CHECK_EQ(test, geometry.height, 992U);

    output.height = 720;
    TEST_CHECK(test, !argus_sender::resolveOutputGeometry(output, sources, &geometry, &error));
    output.resizeMode = argus_sender::RESIZE_CENTER_CROP;
    TEST_CHECK(test, argus_sender::resolveOutputGeometry(output, sources, &geometry, &error));
    TEST_CHECK(test, geometry.clips[0].top > 0.0F);
    TEST_CHECK(test, geometry.clips[0].bottom < 1.0F);

    output.resizeMode = argus_sender::RESIZE_STRETCH;
    TEST_CHECK(test, argus_sender::resolveOutputGeometry(output, sources, &geometry, &error));
    TEST_CHECK(test, std::fabs(geometry.clips[0].top) < 0.0001F);

    argus_sender::MixedRateScheduler scheduler({ 60, 30, 45 });
    TEST_CHECK_EQ(test, scheduler.baseFps(), 60U);
    uint32_t counts[3] = { 0, 0, 0 };
    for (uint32_t tick = 0; tick < 120; ++tick)
    {
        const std::vector<bool> due = scheduler.next();
        for (size_t i = 0; i < due.size(); ++i)
            counts[i] += due[i] ? 1 : 0;
    }
    TEST_CHECK_EQ(test, counts[0], 120U);
    TEST_CHECK_EQ(test, counts[1], 60U);
    TEST_CHECK_EQ(test, counts[2], 90U);
    return test.finish();
}
