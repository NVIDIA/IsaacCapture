<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CloudXR process status

- Probe process existence through `background.process_exists`: reject booleans,
  nonpositive IDs, and out-of-range values. POSIX signal zero is a liveness check;
  Windows `os.kill` would terminate the process instead.
- A liveness probe does not establish process ownership. Preserve the separate
  identity checks before stopping a service or signaling its process group.
