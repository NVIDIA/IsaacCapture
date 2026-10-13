# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tracker metadata helpers in isaaccapture.schema."""

from isaaccapture.schema import (
    TRACKER_METADATA_SAMPLE_SIZE,
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    tracker_metadata_collection_id,
)


def test_collection_id_sits_next_to_the_data_collection():
    assert tracker_metadata_collection_id("tactile/right") == "tactile/right/metadata"
    assert TRACKER_METADATA_TENSOR_IDENTIFIER == "metadata"


def test_sample_size_is_what_consumers_accept():
    assert TRACKER_METADATA_SAMPLE_SIZE == 16384
