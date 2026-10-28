/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureDomainPlan.h"

#include <limits>
#include <sstream>

namespace argus_sender
{
namespace
{

bool compatible(const CaptureEndpointDescriptor& left, const CaptureEndpointDescriptor& right)
{
    return left.modeIndex == right.modeIndex && left.sourceWidth == right.sourceWidth &&
           left.sourceHeight == right.sourceHeight && left.framePeriodNs == right.framePeriodNs;
}

bool failPlan(const std::string& message, CaptureDomainPlan* plan, std::string* error)
{
    if (plan)
        *plan = CaptureDomainPlan();
    if (error)
        *error = message;
    return false;
}

} // namespace

bool buildCaptureDomainPlan(const std::vector<CaptureEndpointDescriptor>& endpoints,
                            CaptureDomainPlan* plan,
                            std::string* error)
{
    if (!plan)
        return false;
    *plan = CaptureDomainPlan();
    plan->endpointDomains.assign(endpoints.size(), std::numeric_limits<size_t>::max());

    for (size_t endpointIndex = 0; endpointIndex < endpoints.size(); ++endpointIndex)
    {
        if (plan->endpointDomains[endpointIndex] != std::numeric_limits<size_t>::max())
        {
            continue;
        }

        const CaptureEndpointDescriptor& endpoint = endpoints[endpointIndex];
        if (endpoint.streamType == STREAM_TYPE_STEREO)
        {
            size_t peerIndex = std::numeric_limits<size_t>::max();
            for (size_t candidate = endpointIndex + 1; candidate < endpoints.size(); ++candidate)
            {
                if (endpoints[candidate].streamIndex == endpoint.streamIndex)
                {
                    peerIndex = candidate;
                    break;
                }
            }
            if (peerIndex == std::numeric_limits<size_t>::max())
            {
                std::ostringstream stream;
                stream << "stereo stream " << endpoint.streamIndex << " does not have two camera endpoints";
                return failPlan(stream.str(), plan, error);
            }
            if (!compatible(endpoint, endpoints[peerIndex]))
            {
                std::ostringstream stream;
                stream << "stereo stream " << endpoint.streamIndex << " uses capture-incompatible sensor modes";
                return failPlan(stream.str(), plan, error);
            }
            const size_t domainIndex = plan->domains.size();
            plan->domains.push_back({ endpointIndex, peerIndex });
            plan->endpointDomains[endpointIndex] = domainIndex;
            plan->endpointDomains[peerIndex] = domainIndex;
            continue;
        }

        const size_t domainIndex = plan->domains.size();
        plan->domains.push_back(std::vector<size_t>(1, endpointIndex));
        plan->endpointDomains[endpointIndex] = domainIndex;
    }
    return true;
}

} // namespace argus_sender
