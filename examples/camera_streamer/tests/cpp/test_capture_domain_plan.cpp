/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureDomainPlan.h"
#include "TestHarness.h"

#include <string>
#include <vector>

namespace
{

argus_sender::CaptureEndpointDescriptor endpoint(size_t stream,
                                                 size_t camera,
                                                 argus_sender::StreamType type,
                                                 uint32_t mode,
                                                 uint32_t width,
                                                 uint32_t height,
                                                 uint64_t periodNs)
{
    argus_sender::CaptureEndpointDescriptor value;
    value.streamIndex = stream;
    value.cameraIndex = camera;
    value.streamType = type;
    value.modeIndex = mode;
    value.sourceWidth = width;
    value.sourceHeight = height;
    value.framePeriodNs = periodNs;
    return value;
}

} // namespace

int main()
{
    TestContext test("capture_domain_plan");
    const uint64_t periodNs = 16666667ULL;
    std::vector<argus_sender::CaptureEndpointDescriptor> endpoints;
    endpoints.push_back(endpoint(0, 0, argus_sender::STREAM_TYPE_STEREO, 0, 2560, 1984, periodNs));
    endpoints.push_back(endpoint(0, 1, argus_sender::STREAM_TYPE_STEREO, 0, 2560, 1984, periodNs));
    endpoints.push_back(endpoint(1, 0, argus_sender::STREAM_TYPE_MONO, 0, 2560, 1984, periodNs));
    endpoints.push_back(endpoint(2, 0, argus_sender::STREAM_TYPE_MONO, 0, 2560, 1984, periodNs));

    argus_sender::CaptureDomainPlan plan;
    std::string error;
    TEST_CHECK(test, argus_sender::buildCaptureDomainPlan(endpoints, &plan, &error));
    TEST_CHECK_EQ(test, plan.domains.size(), static_cast<size_t>(3));
    TEST_CHECK_EQ(test, plan.domains[0].size(), static_cast<size_t>(2));
    TEST_CHECK_EQ(test, plan.domains[1].size(), static_cast<size_t>(1));
    TEST_CHECK_EQ(test, plan.domains[2].size(), static_cast<size_t>(1));
    TEST_CHECK_EQ(test, plan.endpointDomains[0], static_cast<size_t>(0));
    TEST_CHECK_EQ(test, plan.endpointDomains[1], static_cast<size_t>(0));
    TEST_CHECK_EQ(test, plan.endpointDomains[2], static_cast<size_t>(1));
    TEST_CHECK_EQ(test, plan.endpointDomains[3], static_cast<size_t>(2));

    endpoints[3].sourceWidth = 1920;
    TEST_CHECK(test, argus_sender::buildCaptureDomainPlan(endpoints, &plan, &error));
    TEST_CHECK_EQ(test, plan.domains.size(), static_cast<size_t>(3));

    endpoints[1].framePeriodNs = 33333333ULL;
    TEST_CHECK(test, !argus_sender::buildCaptureDomainPlan(endpoints, &plan, &error));
    TEST_CHECK(test, !error.empty());
    return test.finish();
}
