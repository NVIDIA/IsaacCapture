# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaccapture.schema import (
    ContentDescriptor,
    ProfileHandling,
    SemanticProfile,
)


def test_profile_round_trip_preserves_unknown_payload_and_resource() -> None:
    resource = ContentDescriptor(
        "robot",
        media_type="application/xml",
        version="7",
        uris=["package://robot/model.urdf"],
        sha256=b"x" * 32,
        byte_size=3,
        inline_data=b"</>",
    )
    profile = SemanticProfile(
        "org.example.robot.binding",
        version_major=2,
        version_minor=3,
        handling=ProfileHandling.REQUIRED,
        payload_schema="example.RobotBinding",
        payload=b"opaque",
        resources=[resource],
    )

    decoded = SemanticProfile.from_bytes(profile.to_bytes())
    assert decoded.profile_id == "org.example.robot.binding"
    assert (decoded.version_major, decoded.version_minor) == (2, 3)
    assert decoded.handling == ProfileHandling.REQUIRED
    assert decoded.payload_schema == "example.RobotBinding"
    assert decoded.payload == b"opaque"
    assert decoded.resources[0].logical_id == "robot"
    assert decoded.resources[0].inline_data == b"</>"


def test_profile_rejects_ambiguous_payload_and_invalid_digest() -> None:
    with pytest.raises(ValueError, match="payload_schema"):
        SemanticProfile("org.example.profile", payload=b"opaque")
    with pytest.raises(ValueError, match="32 bytes"):
        ContentDescriptor("robot", sha256=b"short")


def test_schema_views_reject_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="invalid FlatBuffer"):
        SemanticProfile.from_bytes(b"not a flatbuffer")
