# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Values the feeder and the printer must agree on."""

DEFAULT_COLLECTION_ID = "tactile"

# TactileTracker's default max_flatbuffer_size (trackers.toml); a larger collection is rejected.
TACTILE_MAX_PAYLOAD_SIZE = 32768
