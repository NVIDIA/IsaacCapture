/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_ARGUS_FRAME_ADAPTER_H
#define ARGUS_SENDER_ARGUS_FRAME_ADAPTER_H

#include "CaptureFrame.h"
#include "FrameTiming.h"

#include <cstdint>
#include <functional>

namespace argus_sender
{

class DmaBuffer;

// Describes one Argus capture as a CaptureFrame: the client-owned DMA-BUF's NV12 layout, the VI start-of-frame TSC as
// the sensor time (SENSOR_CLOCK_NONE when Argus reported none) and the acquire time as the ready time. `release` runs
// when the lease is dropped and returns the buffer to Argus. Returns false, leaving `frame` untouched, if the buffer is
// not a two-plane NV12 surface.
bool describeArgusFrame(DmaBuffer* buffer,
                        const FrameTiming& timing,
                        uint64_t sessionEpoch,
                        uint32_t cameraId,
                        std::function<void()> release,
                        CaptureFrame* frame);

} // namespace argus_sender

#endif // ARGUS_SENDER_ARGUS_FRAME_ADAPTER_H
