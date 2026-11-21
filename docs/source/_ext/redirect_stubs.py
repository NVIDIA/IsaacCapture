# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize redirect stubs: pages left at an old URL that refresh to the new one.

``conf.py`` keeps them out of llms.txt, and ``scripts/check_agent_docs.py`` uses the same
test to know which section indexes sphinx-llm writes.
"""

from __future__ import annotations

import re
from pathlib import Path

_REDIRECT_STUB_RE = re.compile(r"^\.\. meta::\n\s+:http-equiv=refresh:", re.MULTILINE)


def is_redirect_stub(path: Path) -> bool:
    """Whether the ``.rst`` page at path is a redirect stub."""
    return bool(_REDIRECT_STUB_RE.search(path.read_text(encoding="utf-8")))
