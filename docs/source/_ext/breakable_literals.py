# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Let long dotted or underscored names in HTML literals wrap after ``.`` and ``_``."""

from __future__ import annotations

import re

from docutils import nodes

BREAKABLE = re.compile(r"[A-Za-z0-9_.]{24,}")
AFTER_SEPARATOR = re.compile(r"(?<=[._])")


# The search index reads the doctree, so the breaks are added only when writing HTML.
def _visit_literal(self, node: nodes.literal) -> None:
    type(self).visit_literal(self, node)
    text = node.astext()
    if len(node.children) != 1 or not isinstance(node[0], nodes.Text):
        return
    if not BREAKABLE.fullmatch(text):
        return
    parts = [part for part in AFTER_SEPARATOR.split(text) if part]
    for index, part in enumerate(parts):
        if index:
            self.body.append("<wbr>")
        self.visit_Text(nodes.Text(part))
    type(self).depart_literal(self, node)
    raise nodes.SkipNode


def setup(app):
    app.add_node(nodes.literal, override=True, html=(_visit_literal, None))
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
