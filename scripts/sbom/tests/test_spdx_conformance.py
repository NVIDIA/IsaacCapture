# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Third-party conformance check, so the document is valid SPDX and not just ours."""

from __future__ import annotations

import pytest

pytest.importorskip("spdx_tools", reason="spdx-tools is a dev dependency of this suite")

from spdx_tools.spdx.parser.parse_anything import parse_file  # noqa: E402
from spdx_tools.spdx.validation.document_validator import (  # noqa: E402
    validate_full_spdx_document,
)


def test_emitted_document_is_valid_spdx_2_3(built):
    document = parse_file(str(built["spdx_path"]))

    messages = validate_full_spdx_document(document)

    assert [message.validation_message for message in messages] == []


def test_relationships_survive_the_round_trip(built):
    document = parse_file(str(built["spdx_path"]))

    kinds = {
        relationship.relationship_type.name for relationship in document.relationships
    }

    assert {"DESCRIBES", "STATIC_LINK", "CONTAINS", "DEPENDS_ON"} <= kinds
    assert {"GENERATED_FROM", "COPY_OF"} & kinds


def test_an_identifier_newer_than_the_grammar_still_validates(workspace):
    """A vendored file may name an exception the pinned list has and the
    grammar consumers validate with does not; it travels as a LicenseRef with
    its texts, and the report lists it as an evidence gap."""
    import synth

    synth.write(
        workspace.root / "src/vendor/rsync_helper.h",
        "// Copyright 2024 Rsync Folk\n// SPDX-"
        + "License-Identifier: BSL-1.0 WITH rsync-linking-exception\n",
    )
    synth.track(workspace.root)
    synth.write_wheel(workspace.wheel, synth.wheel_payload(workspace))
    collected = synth.collect(workspace)

    document = parse_file(str(collected["out_dir"] / f"{synth.STEM}.spdx.json"))
    assert validate_full_spdx_document(document) == []

    report = collected["licenses"]
    component = next(
        item
        for item in report["components"]
        if item["component"] == "vendored:rsync-folk"
    )
    assert component["license_concluded"] == "LicenseRef-vendored-rsync-folk"
    gaps = {
        item["component"]: item
        for item in report["components_with_unidentified_license"]
    }
    # The id that caused the fallback is named, and its text ships.
    assert "rsync-linking-exception" in gaps["vendored:rsync-folk"]["reason"]
    assert "rsync-linking-exception.txt" in gaps["vendored:rsync-folk"]["evidence"]
