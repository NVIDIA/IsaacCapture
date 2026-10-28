// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <libobsensor/ObSensor.hpp>

#if __has_include(<libobsensor/hpp/DataAcquisitionDevice.hpp>)
using EgoSdkDevice = ob::DataAcquisitionDevice;
#else
using EgoSdkDevice = ob::Device;
#endif

// Compile and link the required extensions without opening a camera.
void check_ego_sdk(ob::Device& device, EgoSdkDevice& ego, const ob::AudioStreamProfile& audio)
{
    device.getRawData(static_cast<OBPropertyID>(0), [](OBDataTranState, OBDataChunk*) {});
    ego.setEgoStateCallback([](const OBEgoStateReport&) {});
    (void)audio.getSampleRate();
}

int main()
{
    return 0;
}
