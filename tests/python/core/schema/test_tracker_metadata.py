# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tracker metadata helpers in isaaccapture.schema."""

from isaaccapture.schema import (
    TRACKER_METADATA_MAX_SIZE,
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    tracker_metadata_collection_id,
)


def test_metadata_collection_is_named_after_the_data_collection():
    assert tracker_metadata_collection_id("tactile") == "tactile/metadata"


def test_wire_contract():
    assert TRACKER_METADATA_TENSOR_IDENTIFIER == "metadata"
    assert TRACKER_METADATA_MAX_SIZE == 16384
