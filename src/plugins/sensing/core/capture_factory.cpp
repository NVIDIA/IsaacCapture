// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_core/capture_factory.hpp"

#ifdef SENSING_HAS_SIPL
#    include "sipl_capture.hpp"
#endif

#ifdef SENSING_HAS_ARGUS
#    include "inc/sensing_core/argus_capture.hpp"
#endif

#include <stdexcept>

namespace plugins
{
namespace sensing
{

std::unique_ptr<CameraCapture> make_capture(CameraBackend backend,
                                            const SensingConfig& config,
                                            const std::vector<uint32_t>& camera_ids)
{
#ifdef SENSING_HAS_SIPL
    if (backend == CameraBackend::Sipl)
    {
        SiplConfig sipl{};
        sipl.platform_config_json = config.platform_config_json;
        sipl.platform_config_name = config.platform_config_name;
        sipl.link_masks = config.link_masks;
        sipl.nito_dir = config.nito_dir;
        sipl.gpu_id = config.gpu_id;
        sipl.full_range = config.full_range;
        sipl.swap_uv = config.swap_uv;
        sipl.isp0_buffers = config.isp0_buffers;
        return std::make_unique<SiplCapture>(sipl);
    }
#endif
#ifdef SENSING_HAS_ARGUS
    if (backend == CameraBackend::Argus)
    {
        ArgusConfig argus;
        argus.devices = camera_ids;
        argus.sensor_mode = config.argus_sensor_mode;
        return std::make_unique<ArgusCapture>(argus);
    }
#endif
    (void)camera_ids;
    throw std::runtime_error("this build does not include the requested camera backend");
}

} // namespace sensing
} // namespace plugins
