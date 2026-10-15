# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tracker metadata chunking helpers in isaaccapture.schema."""

import pytest

from isaaccapture.schema import (
    TRACKER_METADATA_CHUNK_SIZE,
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    split_tracker_metadata,
    tracker_metadata_collection_id,
)


def test_collection_id_sits_next_to_the_data_collection():
    assert tracker_metadata_collection_id("tactile/right") == "tactile/right/metadata"
    assert TRACKER_METADATA_TENSOR_IDENTIFIER == "metadata"


def test_split_respects_the_chunk_size():
    metadata = bytes(range(256)) * 200
    chunks = split_tracker_metadata(metadata, 1024)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1024 for chunk in chunks)


def test_small_metadata_is_one_chunk():
    assert len(split_tracker_metadata(b"\x01" * 100)) == 1
    assert TRACKER_METADATA_CHUNK_SIZE == 16384


def test_empty_metadata_raises():
    with pytest.raises(ValueError, match="metadata size"):
        split_tracker_metadata(b"")


def test_chunk_size_without_room_for_payload_raises():
    with pytest.raises(ValueError, match="max_chunk_size"):
        split_tracker_metadata(b"\x01", 64)


def test_chunk_size_above_the_reader_limit_raises():
    with pytest.raises(ValueError, match="max_chunk_size"):
        split_tracker_metadata(b"\x01", TRACKER_METADATA_CHUNK_SIZE + 1)
