# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown for nodes that sphinx-markdown-builder has no visitor for.

That builder drops an unknown node together with its children, so without these the
``page.md`` copies sphinx-llm publishes lose captions, generic admonitions and
sphinx-design card titles. Nodes that only matter in HTML are silenced in ``conf.py``.
"""

from __future__ import annotations

from docutils import nodes
from sphinx_design.shared import PassthroughTextElement


def _visit_as_paragraph(self, node):
    self.visit_paragraph(node)


def _depart_as_paragraph(self, node):
    self.depart_paragraph(node)


def _visit_admonition(self, node):
    """A generic admonition becomes a heading line followed by its body."""
    title = node[0] if node.children and isinstance(node[0], nodes.title) else None
    if title is not None:
        self.add(f"#### {title.astext()}", prefix_eol=2)
    for child in node.children:
        if child is not title:
            child.walkabout(self)
    raise nodes.SkipNode


def _is_card_title(node):
    return "sd-card-title" in node.parent.get("classes", [])


def _visit_card_text(self, node):
    """Card titles render bold; the same element also wraps a card's stretched link."""
    self.visit_paragraph(node)
    if _is_card_title(node):
        self.add("**")


def _depart_card_text(self, node):
    if _is_card_title(node):
        self.add("**")
    self.depart_paragraph(node)


def setup(app):
    app.add_node(
        nodes.caption,
        override=True,
        markdown=(_visit_as_paragraph, _depart_as_paragraph),
    )
    app.add_node(nodes.admonition, override=True, markdown=(_visit_admonition, None))
    app.add_node(
        PassthroughTextElement,
        override=True,
        markdown=(_visit_card_text, _depart_card_text),
    )
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
