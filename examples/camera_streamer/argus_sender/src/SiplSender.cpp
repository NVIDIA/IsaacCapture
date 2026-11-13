/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SiplSender.h"

#include "CaptureSender.h"
#include "Config.h"
#include "SiplCapture.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace argus_sender
{

bool runSiplSender(const AppConfig& config, bool* receiverDisconnected, bool appendDump)
{
    if (receiverDisconnected)
        *receiverDisconnected = false;

    SiplConfig siplConfig;
    siplConfig.platformConfigJson = config.sipl.platformConfigJson;
    siplConfig.platformConfigName = config.sipl.platformConfigName;
    siplConfig.linkMasks = config.sipl.linkMasks;
    siplConfig.nitoDir = config.sipl.nitoDir;
    siplConfig.isp0Buffers = config.sipl.isp0Buffers;
    const uint64_t sessionId = createCaptureSessionId();
    siplConfig.sessionEpoch = sessionId;

    std::unique_ptr<CameraCapture> capture;
    try
    {
        capture.reset(new SiplCapture(siplConfig));
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "SIPL capture setup failed: %s\n", e.what());
        return false;
    }

    // Streams name SIPL pipeline indices; geometry and rate come from the platform configuration, never from the
    // command line.
    std::vector<CaptureSenderStream> streams(config.streams.size());
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const StreamConfig& configStream = config.streams[i];
        CaptureSenderStream& stream = streams[i];
        const char* label = configStream.id.c_str();
        stream.cameraId = configStream.cameras[0].device;
        const CameraDescriptor* found = NULL;
        for (size_t c = 0; c < capture->cameras().size(); ++c)
        {
            if (capture->cameras()[c].id == stream.cameraId)
                found = &capture->cameras()[c];
        }
        if (!found)
        {
            std::string available;
            for (size_t c = 0; c < capture->cameras().size(); ++c)
                available += " " + std::to_string(capture->cameras()[c].id);
            std::fprintf(stderr,
                         "Stream %s: device %u is not a SIPL pipeline index in '%s'. Available:%s (a pipeline index is "
                         "not the GMSL link index)\n",
                         label, stream.cameraId, config.sipl.platformConfigName.c_str(), available.c_str());
            return false;
        }
        stream.width = found->width;
        stream.height = found->height;
        if ((configStream.output.width != 0 && configStream.output.width != stream.width) ||
            (configStream.output.height != 0 && configStream.output.height != stream.height))
        {
            std::fprintf(stderr,
                         "Stream %s: the sipl backend encodes the sensor's %ux%u and does not scale; remove "
                         "output.width/height or set them to the sensor size\n",
                         label, stream.width, stream.height);
            return false;
        }
        stream.fps = configStream.output.fps ? configStream.output.fps : static_cast<uint32_t>(std::lround(found->fps));
        if (stream.fps == 0)
        {
            std::fprintf(stderr, "Stream %s resolved to zero FPS\n", label);
            return false;
        }
        std::printf("Stream %s: SIPL pipeline %u (%s) %ux%u @ %u FPS\n", label, stream.cameraId, found->name.c_str(),
                    stream.width, stream.height, stream.fps);
    }

    return runCaptureSender(config, std::move(capture), streams, sessionId, "SIPL", receiverDisconnected, appendDump);
}

} // namespace argus_sender
