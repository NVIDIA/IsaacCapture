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
