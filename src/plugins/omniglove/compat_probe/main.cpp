// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <omniglove/omniglove.h>

#include <cstdint>
#include <iostream>

namespace
{

void callback(const omniglove_stream_sample_t*, void*)
{
}

} // namespace

int main()
{
    omniglove_stream_config_t config{};
    config.struct_size = sizeof(config);
    config.kind = OMNIGLOVE_STREAM_POST_DATA;
    config.stream_mode = OMNIGLOVE_STREAM_MODE_DEFAULT;
    config.output_mode = OMNIGLOVE_STREAM_OUTPUT_DIRECT;
    config.source_interval_ms = 8;
    config.source_timeout_ms = 200;
    config.callback = callback;
    config.callback_queue_capacity = 64;

    omniglove_post_data_t post{};
    post.thumb.landmarks.cmc.position_m[0] = 0.0F;
    post.index.landmarks.mcp.quaternion_xyzw[3] = 1.0F;
    post.wrist.landmark.quaternion_xyzw[3] = 1.0F;

    static_cast<void>(config);
    static_cast<void>(post);

    const uint32_t runtime = omniglove_abi_version();
    std::cout << "Omniglove header ABI " << OMNIGLOVE_ABI_VERSION_MAJOR << '.' << OMNIGLOVE_ABI_VERSION_MINOR << '.'
              << OMNIGLOVE_ABI_VERSION_PATCH << ", runtime ABI " << ((runtime >> 16U) & 0xffU) << '.'
              << ((runtime >> 8U) & 0xffU) << '.' << (runtime & 0xffU) << '\n';
    return runtime == OMNIGLOVE_ABI_VERSION ? 0 : 1;
}
