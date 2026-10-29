<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone GMSL/Argus streamer

- Keep the wire/configuration tests runnable without Jetson or CUDA dependencies.
- Preserve upstream notices when adding SPDX headers; missing SPDX does not mean
  a file lacks a license. Keep third-party exceptions separate from Apache-2.0 code.
- Keep each review independently buildable: add tests and documentation with their dependencies.
- Normalize generated build fragments to one final newline before staging.
- Keep argus_sender, camera_receiver and cuda_ipc_consumer directly under camera_streamer;
  do not group standalone transport components under the TeleViz visualization name.
