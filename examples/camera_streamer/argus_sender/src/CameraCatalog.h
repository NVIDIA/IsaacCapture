/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAMERA_CATALOG_H
#define ARGUS_SENDER_CAMERA_CATALOG_H

#include <Argus/Argus.h>

#include <cstdint>

namespace argus_sender
{

double frameRateFromDurationNs(uint64_t durationNs);
uint32_t roundedFrameRateFromDurationNs(uint64_t durationNs);

bool listCameraModes(Argus::CameraProvider* provider);
Argus::CameraDevice* selectCameraDevice(Argus::CameraProvider* provider, uint32_t index);
Argus::SensorMode* selectSensorMode(Argus::CameraDevice* device, uint32_t index);

} // namespace argus_sender

#endif // ARGUS_SENDER_CAMERA_CATALOG_H
