# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext"))

# -- Project information -----------------------------------------------------

project = "Isaac Teleop"
build_time = datetime.now(timezone.utc)
copyright = f"2025-{build_time.year}, NVIDIA CORPORATION & AFFILIATES"
copyright += f", last updated on {build_time.strftime('%B %d, %Y')}"
author = "NVIDIA"


def _smv_ref_name():
    """Git ref name sphinx-multiversion is building, or "" outside it.

    sphinx-multiversion builds every ref with ``-c`` pointing at the checkout it
    was invoked from, so this conf.py is always the deploy branch's and anything
    read from the tree describes that branch, not the ref. It passes the ref name
    as ``-D smv_current_version=``, which Sphinx applies to the config only after
    conf.py has run, so read it off the command line.
    """
    prefix = "smv_current_version="
    return next((a[len(prefix) :] for a in sys.argv if a.startswith(prefix)), "")


_version_file = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")
# ``VERSION`` is ``MAJOR.MINOR.x`` on main and on every release branch and tag, so its
# first two components name the release series a given build of the docs describes.
if os.path.exists(_version_file):
    with open(_version_file) as f:
        version = release = f.read().strip()
else:
    version = release = "0.0.0"


def _pip_pin(ref_name, fallback):
    """Install specifier for the series a build describes: ``1.5.x`` -> ``~=1.5.0``.

    Release refs name their own series (``release/1.4.x``, ``v1.4.7``); ``main``
    and plain ``sphinx-build`` fall back to the checked-out ``VERSION``.
    """
    tail = ref_name.rsplit("/", 1)[-1].removeprefix("v")
    major, _, rest = (tail if tail[:1].isdigit() else fallback).partition(".")
    minor = rest.partition(".")[0]
    if not (major.isdigit() and minor.isdigit()) or major == "0":
        return "~=1.0"
    return f"~={major}.{minor}.0"


_pip_version_pin = _pip_pin(_smv_ref_name(), version)

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinx_design",
    "ecosystem_grid",
]

exclude_patterns = ["build", "_templates", "_data", "_ext", "Thumbs.db", ".DS_Store"]

# sphinx-copybutton only targets highlighted blocks (``div.highlight pre``) by default,
# which skips ``parsed-literal`` (rendered as a bare ``pre.literal-block``).  Commands
# that interpolate a substitution have to use ``parsed-literal``, so widen the selector
# to keep the copy button on them.
copybutton_selector = "div.highlight pre, pre.literal-block"

templates_path = ["_templates"]

# sphinx-multiversion: which refs to build (avoids "No matching refs found" in CI)
smv_remote_whitelist = r"^.*$"
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|release/.*)$")
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")

# -- Options for HTML output ---------------------------------------------------

html_title = "Isaac Teleop Documentation"
html_theme = "nvidia_sphinx_theme"
html_favicon = "_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["css/custom.css", "css/ecosystem.css"]

# Per-version icon link overrides.  Keyed by the git ref name that
# sphinx-multiversion builds.  Unmatched refs (including plain ``sphinx-build``
# without multiversion) use _DEFAULT_ICONS.
_smv_name = _smv_ref_name()

_DEFAULT_ICONS = {
    "teleop_version": "main",
    "teleop_url": "https://github.com/NVIDIA/IsaacTeleop",
    "cloudxr_version": "6.2",
    "cloudxr_url": "https://docs.nvidia.com/cloudxr-sdk",
    "lab_version": "3.0",
    "lab_url": "https://isaac-sim.github.io/IsaacLab",
}
_VERSION_ICON_MAP = {
    "release/1.0.x": {
        "teleop_version": "1.0",
        "teleop_url": "https://github.com/NVIDIA/IsaacTeleop/tree/release/1.0.x",
        "cloudxr_version": "6.1",
        "cloudxr_url": "https://docs.nvidia.com/cloudxr-sdk/release/6",
        "lab_version": "3.0",
        "lab_url": "https://isaac-sim.github.io/IsaacLab/develop",
    },
}
_icons = _VERSION_ICON_MAP.get(_smv_name, _DEFAULT_ICONS)

# Branch-specific CloudXR web client ("CloudXR.js") deployment.  docs.yaml
# publishes the prebuilt web client to ``/client/<slug>/`` where ``<slug>`` is
# the built ref name with ``/`` replaced by ``-`` (e.g. ``main``,
# ``release-1.3.x``, ``v1.2.3``).  Resolving the same slug here lets each
# versioned docs build link to the matching client instead of always ``main``.
_client_slug = (_smv_name or "main").replace("/", "-")
_web_client_url = f"https://nvidia.github.io/IsaacCapture/client/{_client_slug}/"

# Shared substitutions + link targets injected into every page, so the
# branch-specific web client URL and version pin live in one place.
# ``|web_client_url|`` expands the bare URL and ``|pip_version_pin|`` the
# version specifier (both usable in prose and ``parsed-literal`` blocks); the
# named targets back ```...`_`` references in the prose.
rst_epilog = f"""
.. |web_client_url| replace:: {_web_client_url}
.. |pip_version_pin| replace:: {_pip_version_pin}
.. _`nvidia.github.io/IsaacCapture/client`: {_web_client_url}
.. _`Isaac Teleop Web Client`: {_web_client_url}
"""

html_theme_options = {
    "collapse_navigation": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "search_bar_text": "Search...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": _icons["teleop_url"],
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "CloudXR",
            "url": _icons["cloudxr_url"],
            "icon": f"https://img.shields.io/badge/CloudXR-{_icons['cloudxr_version']}-green.svg",
            "type": "url",
        },
        {
            "name": "Isaac Lab",
            "url": _icons["lab_url"],
            "icon": f"https://img.shields.io/badge/IsaacLab-{_icons['lab_version']}-silver.svg",
            "type": "url",
        },
    ],
    # The nvidia theme defaults navbar_center to its own version switcher, which
    # needs a switcher JSON; sphinx-multiversion feeds ``versioning.html`` instead.
    "navbar_center": [],
    "navbar_end": ["versioning.html", "search-button-field", "theme-switcher"],
    # Below 960px the theme hides navbar_end, so search survives as this magnifier.
    "navbar_persistent": ["search-button"],
}

# Primary sidebar (left): icon links row, then TOC (like Isaac Lab)
html_sidebars = {
    "**": ["icon-links", "sidebar-nav-bs"],
}

# Edit page button: link to GitHub so users can suggest edits (PyData theme uses html_context)
html_context = {
    "github_user": "NVIDIA",
    "github_repo": "IsaacTeleop",
    "github_version": _smv_name or "main",
    "doc_path": "docs/source",
}

# Base URL for linking to repository source (used by code-file and code-dir roles).
_GH_BASE = "https://github.com/NVIDIA/IsaacTeleop"
_GH_BRANCH = html_context["github_version"]


def _parse_code_role(text):
    """Parse role text as 'path' or 'label <path>'. Returns (label, path)."""
    text = " ".join(text.split())  # collapse newlines from source line wrapping
    if " <" in text and text.endswith(">"):
        label, path = text.rsplit(" <", 1)
        return label.strip(), path[:-1].strip()
    return text, text


def _code_file_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role for linking to a file in the GitHub repo: :code-file:`path` or :code-file:`label <path>`."""
    from docutils import nodes
    from docutils.utils import unescape

    label, path = _parse_code_role(unescape(text))
    url = f"{_GH_BASE}/blob/{_GH_BRANCH}/{path}"
    node = nodes.reference(rawtext, label, refuri=url)
    return [node], []


def _code_dir_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role for linking to a directory in the GitHub repo: :code-dir:`path` or :code-dir:`label <path>`."""
    from docutils import nodes
    from docutils.utils import unescape

    label, path = _parse_code_role(unescape(text))
    path = path.rstrip("/")
    url = f"{_GH_BASE}/tree/{_GH_BRANCH}/{path}"
    node = nodes.reference(rawtext, label if label != path else path, refuri=url)
    return [node], []


def _external_links_new_tab(app, doctree, docname):
    """Mark external links to open in a new tab."""
    from docutils import nodes

    for node in doctree.traverse(nodes.reference):
        refuri = node.get("refuri", "")
        if refuri.startswith(("http://", "https://")):
            node["target"] = "_blank"


def setup(app):
    app.add_role("code-file", _code_file_role)
    app.add_role("code-dir", _code_dir_role)
    app.add_config_value("html_external_links_new_tab", True, "html")
    # Add rel="noopener noreferrer" when target="_blank" so external links are safe
    from sphinx.writers.html5 import HTML5Translator

    _base_visit_reference = HTML5Translator.visit_reference

    def visit_reference(self, node):
        if (
            getattr(self.config, "html_external_links_new_tab", True)
            and node.get("target") == "_blank"
            and "rel" not in node
        ):
            node["rel"] = "noopener noreferrer"
        return _base_visit_reference(self, node)

    HTML5Translator.visit_reference = visit_reference
    app.connect("doctree-resolved", _external_links_new_tab)
