# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wheel contents SBOM and license evidence collector."""

TOOL_NAME = "isaaccapture-sbom"
# Bump on any change that alters the shape of an emitted document or report.
TOOL_VERSION = "1.0.0"
SPDX_VERSION = "SPDX-2.3"


def stamped_now():
    """The time to publish, honouring SOURCE_DATE_EPOCH.

    Every artifact a release advertises uses this; the manifest binds them as one
    chain, so they must agree on the clock.
    """
    import os
    from datetime import datetime, timezone

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.isdigit():
        return datetime.fromtimestamp(int(epoch), timezone.utc)
    return datetime.now(timezone.utc)
