# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown for nodes that sphinx-markdown-builder has no visitor for, or gets wrong.

That builder drops an unknown node together with its children, so without these the
``page.md`` copies sphinx-llm publishes lose captions, generic admonitions and
sphinx-design card titles. Nodes that only matter in HTML are silenced in ``conf.py``.
Its links, images and downloads are also rewritten to resolve from the published page.
"""

from __future__ import annotations

import posixpath

from docutils import nodes
from sphinx import addnodes
from sphinx_design.shared import PassthroughTextElement
from sphinx_markdown_builder.contexts import WrappedContext


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


def _from_page(self, path):
    """``path`` relative to the build root, as a link from the page being written."""
    page_dir = posixpath.dirname(self.builder.current_doc_name)
    return posixpath.relpath(path, page_dir or ".")


def _parents(node):
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def _visit_reference(self, node):
    """Links in a literal block stay plain text, so a copied command still runs."""
    if any(isinstance(parent, nodes.literal_block) for parent in _parents(node)):
        self.add(node.astext())
        raise nodes.SkipNode
    # A plain RST link to a section (`Title`_) carries a refid but no "internal" flag,
    # which sphinx-markdown-builder reads as an external link with an empty URL.
    if node.get("refid") and "refuri" not in node:
        self._push_context(WrappedContext("[", f"](#{node['refid']})"))
        return
    type(self).visit_reference(self, node)


def _depart_reference(self, node):
    self._pop_context(node)


def _visit_image(self, node):
    """Point at the copy the HTML build serves, since the Markdown sits beside it."""
    uri = node["uri"]
    copied = self.builder.env.images.get(uri)
    if copied is not None:
        uri = _from_page(self, posixpath.join("_images", copied[1]))
    self.add(f"![{node.get('alt', 'image')}]({uri})")


def _visit_download_reference(self, node):
    if "refuri" in node or "filename" not in node:
        type(self).visit_download_reference(self, node)
        return
    target = _from_page(self, posixpath.join("_downloads", node["filename"]))
    self._push_context(WrappedContext("[", f"]({target})"))


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
    app.add_node(
        nodes.reference,
        override=True,
        markdown=(_visit_reference, _depart_reference),
    )
    app.add_node(nodes.image, override=True, markdown=(_visit_image, None))
    app.add_node(
        addnodes.download_reference,
        override=True,
        markdown=(_visit_download_reference, _depart_reference),
    )
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
