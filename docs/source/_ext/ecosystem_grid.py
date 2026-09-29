# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ecosystem page content, generated from ``_data/devices.yaml``.

The ``device-matrix`` directive renders one table section, ``device-count`` counts its
rows, and ``eco-block`` wraps layout. Adding an entry takes one YAML record; page
markup never changes.

Invalid records raise at build time rather than warn, because ``make current-docs``
runs ``sphinx-build -W`` and a silently dropped row is worse than a failed build.
"""

from __future__ import annotations

import os
import re
from typing import ClassVar

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.addnodes import pending_xref
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxRole

DEVICE_DATA = "_data/devices.yaml"
DEVICE_LOGO_DIR = "_static/logos"

# A device no release carries yet, so the only way to get it is to build main.
SINCE_MAIN = "main"

# The first stable release; everything before 1.0 was a pre-release. A device carried by
# an earlier tag is listed from here rather than from a 0.x version.
SINCE_FLOOR = "v1.0.191"

# A link into the docs site or into the IsaacTeleop repository stays inside this
# project, so it carries no external marker -- arrow or screen-reader label. Every
# other domain gets one, including other GitHub organizations such as isaac-sim.
PROJECT_REPO = "https://github.com/NVIDIA/IsaacTeleop"

# Group header rows within a table. The input wording is the old page's own table
# titles, so a reader arriving from an old link recognizes them.
DEVICE_GROUPS = {
    "xr": "XR Headsets and Tracking Peripherals",
    "peripheral": "Standalone Input Devices",
    "data_factory": "Data Factory",
    "cloud": "Cloud Infrastructure",
}
# One table per section. A section names the groups it renders and its two column
# headings: input devices list the modes that decide retargeting, everything else lists
# what the entry provides. A section spanning one group emits no group header row --
# the section heading above the table already said it.
DEVICE_SECTIONS = {
    "input": {
        "groups": ("xr", "peripheral"),
        "columns": ("Device", "Input modes"),
        "planned": True,
    },
    "data_factory": {
        "groups": ("data_factory",),
        "columns": ("Name", "Provides"),
        "planned": False,
    },
    "cloud": {
        "groups": ("cloud",),
        "columns": ("Name", "Provides"),
        "planned": False,
    },
}
# A group no section renders would drop its rows off the page silently.
_SECTION_GROUPS = {
    group for section in DEVICE_SECTIONS.values() for group in section["groups"]
}
# The Details panel's three columns, in this order and never renamed per device: set up
# first because it is what most readers came for, acquire last because few need it. A
# column with nothing to say still renders, so the three stay aligned down the table.
DETAIL_COLUMNS = (
    ("setup", "Set up"),
    ("requirements", "Requirements"),
    ("acquire", "Acquire"),
)
# Every row's disclosure shares one name, which is what makes the browser close the open
# panel when the reader opens another -- no JavaScript involved.
DISCLOSURE_GROUP = "isaac-teleop-device"


class eco_block(nodes.General, nodes.Element):
    """A plain ``<div>`` carrying only the classes we ask for.

    Deliberately not docutils' own ``container`` node: that emits
    ``class="docutils container"``, and because Bootstrap claims the same class name
    the theme neutralizes it with ``.docutils.container {padding-inline: unset}``.
    That rule outranks ours and silently flattens every horizontal padding.

    ``html_tag`` and ``html_attributes`` let one node also stand in for the handful of
    elements docutils has no equivalent for (``<details>``, ``<summary>``) while keeping
    their contents as ordinary docutils children, so non-HTML builders still render the
    text.
    """


def _visit_eco_block(self, node):
    self.body.append(
        self.starttag(
            node, node.get("html_tag", "div"), "", **node.get("html_attributes", {})
        )
    )


def _depart_eco_block(self, node):
    self.body.append(f"</{node.get('html_tag', 'div')}>\n")


class eco_inline(nodes.Inline, nodes.TextElement):
    """``eco_block``'s inline twin, for elements that sit inside a paragraph.

    ``TextElement`` and not ``Element``: Sphinx's HTML writer treats a reference whose
    parent is not one as an image reference, and asserts that its only child is an
    image. A link nested in this span would crash the build.
    """


def _element(tag: str, classes: list[str], *, inline: bool = False, **attributes):
    node = (eco_inline if inline else eco_block)(classes=classes)
    node["html_tag"] = tag
    node["html_attributes"] = attributes
    return node


def _passthrough(self, node):
    """Non-HTML builders render the children and ignore the wrapper."""


_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ISSUE_RE = re.compile(r"/(?:issues|pull)/(\d+)/?$")
# A release tag exactly as the repository spells it, because the link resolves it. Every
# tag from SINCE_FLOOR on carries the "v", and only the releases below it drop the
# prefix; matching those too lets the floor explain itself rather than reading as a
# misspelling.
_SINCE_RE = re.compile(r"^v?\d+\.\d+\.\d+$")
_TEL_RE = re.compile(r"[^\d+]")

# A stroked glyph rather than U+2197: the Unicode arrow renders far heavier than the
# surrounding 13px text in system fonts. Size and stroke come from the design mock.
_ARROW_SVG = (
    '<svg class="eco-link-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" aria-hidden="true" focusable="false">'
    '<path d="M7 17 17 7M9 7h8v8"></path></svg>'
)

_cache: dict[str, tuple[float, object]] = {}


def _fail(source: str, record: str | None, message: str) -> None:
    where = f"{source}: {record!r}" if record else source
    raise ExtensionError(f"ecosystem data: {where}: {message}")


_DEVICE_REQUIRED = ("id", "name", "url", "group", "modes")
_DEVICE_KNOWN = _DEVICE_REQUIRED + ("since", "details", "company")
_COMPANY_REQUIRED = ("name", "logo")
_COMPANY_KNOWN = _COMPANY_REQUIRED + ("logo_dark",)
_PLANNED_REQUIRED = ("id", "name", "url", "note", "tracking")


_ENTRY_FORMS = ("label", "email", "phone")
_ENTRY_KEYS = set(_ENTRY_FORMS) | {"doc", "ref", "url", "note"}


def _validate_entry(entry, source: str, name: str, column: str) -> None:
    """A panel entry is a bare string, a link, or a way to reach someone."""
    if isinstance(entry, str):
        if entry.endswith("."):
            _fail(
                source,
                name,
                f"{column} entry {entry!r}: drop the trailing period, panel entries are "
                "noun phrases and not sentences",
            )
        return
    if not isinstance(entry, dict):
        _fail(source, name, f"{column} entry must be a string or a mapping")

    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        _fail(source, name, f"{column} entry: unknown field(s) {', '.join(unknown)}")
    forms = [key for key in _ENTRY_FORMS if entry.get(key)]
    if len(forms) != 1:
        _fail(
            source,
            name,
            f"{column} entry: needs exactly one of {', '.join(_ENTRY_FORMS)}",
        )

    if forms == ["email"]:
        if "@" not in str(entry["email"]):
            _fail(source, name, f"{column} entry: {entry['email']!r} is not an address")
        return
    if forms == ["phone"]:
        return

    label = entry["label"]
    targets = [key for key in ("doc", "ref", "url") if entry.get(key)]
    if len(targets) != 1:
        _fail(
            source,
            name,
            f"{column} entry {label!r}: needs exactly one of 'doc', 'ref', or 'url'",
        )
    if targets == ["url"] and not str(entry["url"]).startswith(("http://", "https://")):
        _fail(source, name, f"{column} entry {label!r}: url must be http(s)")


def _validate_details(details, source: str, name: str) -> None:
    if not isinstance(details, dict):
        _fail(source, name, "'details' must be a mapping of panel columns")
    unknown = sorted(set(details) - {key for key, _ in DETAIL_COLUMNS})
    if unknown:
        _fail(
            source,
            name,
            f"details: unknown column(s) {', '.join(unknown)}; the panel renders "
            f"{', '.join(key for key, _ in DETAIL_COLUMNS)}",
        )
    if not any(details.get(key) for key, _ in DETAIL_COLUMNS):
        _fail(
            source,
            name,
            "details is empty; omit it and the row renders without a Details button",
        )
    for key, heading in DETAIL_COLUMNS:
        entries = details.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list) or not entries:
            _fail(
                source,
                name,
                f"details.{key} must be a non-empty list; leave it out and the "
                f"{heading!r} column renders an em dash",
            )
        for entry in entries:
            _validate_entry(entry, source, name, f"details.{key}")


def _release_order(tag: str) -> tuple:
    """Sortable form of a release tag, whose "v" is optional in the oldest ones."""
    return tuple(int(part) for part in tag.lstrip("v").split("."))


def _validate_since(since, source: str, name: str, details) -> None:
    """A release tag, or ``main``: both name something the tag can link to.

    Rejecting anything else keeps the field derivable rather than editorial -- a date, a
    release line, or "latest" would all make the tag a claim nobody can check, and none
    of them resolve to a page the link can point at.
    """
    if since != SINCE_MAIN and not _SINCE_RE.fullmatch(str(since)):
        _fail(
            source,
            name,
            f"since must be {SINCE_MAIN!r} or a release tag such as 'v1.3.131', "
            f"not {since!r}",
        )
    if since != SINCE_MAIN and _release_order(since) < _release_order(SINCE_FLOOR):
        _fail(
            source,
            name,
            f"since is {since!r}, older than {SINCE_FLOOR!r}; a device carried by an "
            f"earlier release is listed from {SINCE_FLOOR!r} instead",
        )
    if not details:
        _fail(
            source,
            name,
            "since renders inside the Details panel, and this row has no 'details' to "
            "open, so the tag would never appear",
        )


def _validate_company(company, source: str, name: str, confdir: str) -> None:
    """The panel header renders a mark and nothing else, so the schema carries no copy.

    The mark is the whole point, so ``logo`` is required: a company with no artwork
    leaves the block out rather than falling back to its name in text.
    """
    if not isinstance(company, dict):
        _fail(source, name, "'company' must be a mapping")
    unknown = sorted(set(company) - set(_COMPANY_KNOWN))
    if unknown:
        _fail(source, name, f"company: unknown field(s) {', '.join(unknown)}")
    for field in _COMPANY_REQUIRED:
        if not str(company.get(field) or "").strip():
            _fail(source, name, f"company.{field} is required")
    for key in ("logo", "logo_dark"):
        filename = company.get(key)
        if filename and not os.path.isfile(
            os.path.join(confdir, DEVICE_LOGO_DIR, filename)
        ):
            _fail(
                source,
                name,
                f"company.{key} points at {DEVICE_LOGO_DIR}/{filename}, "
                "which does not exist",
            )


def _validate_devices(
    data: dict, source: str, confdir: str
) -> tuple[list[dict], list[dict]]:
    """A row is a name, a link, and what it provides. Everything else is optional."""
    devices = data.get("devices")
    if not isinstance(devices, list) or not devices:
        _fail(source, None, "expected a non-empty 'devices' list")

    seen_ids: set[str] = set()
    for index, record in enumerate(devices):
        if not isinstance(record, dict):
            _fail(source, None, f"entry {index} is not a mapping")
        name = record.get("id", f"<entry {index}>")

        for field in _DEVICE_REQUIRED:
            if record.get(field) in (None, "", []):
                _fail(source, name, f"missing required field {field!r}")
        unknown = sorted(set(record) - set(_DEVICE_KNOWN))
        if unknown:
            _fail(source, name, f"unknown field(s) {', '.join(unknown)}")

        if not _ID_RE.match(record["id"]):
            _fail(source, name, "id must be lowercase kebab-case")
        if record["id"] in seen_ids:
            _fail(source, name, "duplicate id")
        seen_ids.add(record["id"])

        if record["group"] not in DEVICE_GROUPS:
            _fail(source, name, f"group must be one of {tuple(DEVICE_GROUPS)}")
        if record["group"] not in _SECTION_GROUPS:
            _fail(
                source,
                name,
                f"no table section renders group {record['group']!r}; add it to a "
                "DEVICE_SECTIONS entry or the row appears nowhere",
            )
        # The device name is the manufacturer link, which is the only reason no row
        # carries a separate "Overview" link.
        if not str(record["url"]).startswith(("http://", "https://")):
            _fail(source, name, "url must be the manufacturer's http(s) product page")

        if record.get("details") is not None:
            _validate_details(record["details"], source, name)

        if record.get("since") is not None:
            _validate_since(record["since"], source, name, record.get("details"))

        if record.get("company") is not None:
            _validate_company(record["company"], source, name, confdir)

    planned = data.get("planned") or []
    if not isinstance(planned, list):
        _fail(source, None, "'planned' must be a list")
    for index, record in enumerate(planned):
        name = (
            record.get("id", f"<planned {index}>") if isinstance(record, dict) else None
        )
        if not isinstance(record, dict):
            _fail(source, None, f"planned entry {index} is not a mapping")
        for field in _PLANNED_REQUIRED:
            if record.get(field) in (None, "", []):
                _fail(source, name, f"missing required field {field!r}")
        unknown = sorted(set(record) - set(_PLANNED_REQUIRED))
        if unknown:
            _fail(source, name, f"unknown field(s) {', '.join(unknown)}")
        if not _ISSUE_RE.search(str(record["tracking"])):
            _fail(
                source,
                name,
                "tracking urls must end in an issue or pull number, so the line can "
                "render 'Tracking #276'",
            )

    return devices, planned


def _read(confdir: str, relpath: str, build):
    """Parse and validate a data file, cached until it changes on disk."""
    source = os.path.join(confdir, relpath)
    if not os.path.isfile(source):
        raise ExtensionError(f"ecosystem data: {relpath} not found")

    mtime = os.path.getmtime(source)
    cached = _cache.get(source)
    if cached and cached[0] == mtime:
        return cached[1]

    with open(source, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parsed = build(data)
    _cache[source] = (mtime, parsed)
    return parsed


def _load_devices(env, relpath: str = DEVICE_DATA) -> tuple[list[dict], list[dict]]:
    """Read the device data, and tell Sphinx the page depends on it.

    The directive reads the file, so nothing else ties the page to it: without the
    dependency, an incremental build reuses the cached doctree and renders the previous
    table after the data changes.
    """
    confdir = env.app.confdir
    env.note_dependency(os.path.join(confdir, relpath))
    return _read(
        confdir, relpath, lambda data: _validate_devices(data, relpath, confdir)
    )


def _line(classes: list[str], text: str = "") -> nodes.paragraph:
    """A paragraph, which is what docutils expects as the parent of inline content."""
    para = nodes.paragraph(classes=classes)
    if text:
        para += nodes.Text(text)
    return para


def _is_external(url: str) -> bool:
    return not str(url).startswith(PROJECT_REPO)


def _mark_external(ref: nodes.Element, *, arrow: bool = True) -> None:
    """Say the link leaves the project. The arrow is decorative, so screen readers get
    the hidden label instead; a link that shows no arrow still carries it."""
    if arrow:
        ref += nodes.raw("", _ARROW_SVG, format="html")
    ref += nodes.inline("", " (external)", classes=["eco-link-external-label"])


def _xref(
    target: str, label: nodes.Node, docname: str, is_doc: bool, classes: list[str]
):
    """A cross-reference Sphinx resolves later, so a dead target fails the build."""
    return pending_xref(
        "",
        label,
        refdoc=docname,
        refdomain="std",
        reftype="doc" if is_doc else "ref",
        reftarget=target,
        refexplicit=True,
        refwarn=True,
        classes=classes,
    )


def _device_name(record: dict) -> nodes.paragraph:
    """The row label is the manufacturer link, so no row needs an Overview link.

    No arrow: with one on every row the column reads as decoration rather than as a
    signal, and the mock leaves them off. Screen readers still get the label.
    """
    line = _line(["device-name"])
    ref = nodes.reference("", record["name"], refuri=record["url"])
    _mark_external(ref, arrow=False)
    line += ref
    return line


def _detail_line(entry, docname: str) -> nodes.paragraph:
    """One panel entry: a noun phrase, a link, or an address to reach someone at."""
    line = _line(["device-panel-line"])
    if isinstance(entry, str):
        line += nodes.Text(entry)
        return line

    # No external marker on either: mailto: and tel: do not navigate anywhere, and the
    # arrow would read as "this leaves the docs".
    if entry.get("email"):
        address = str(entry["email"])
        line += nodes.reference("", address, refuri=f"mailto:{address}")
    elif entry.get("phone"):
        phone = str(entry["phone"])
        line += nodes.reference("", phone, refuri=f"tel:{_TEL_RE.sub('', phone)}")
    elif entry.get("url"):
        ref = nodes.reference("", entry["label"], refuri=entry["url"])
        if _is_external(entry["url"]):
            _mark_external(ref)
        line += ref
    else:
        line += _xref(
            entry.get("doc") or entry["ref"],
            nodes.Text(entry["label"]),
            docname,
            bool(entry.get("doc")),
            [],
        )
    if entry.get("note"):
        line += nodes.Text(f" — {entry['note']}")
    return line


def _disclosure(panel_id: str) -> nodes.Element:
    """The row's third column: a native ``<details>`` holding only its ``<summary>``.

    The panel itself is a sibling grid item so it can span all three columns, and the
    CSS reveals it from ``:has(details[open])``. Keeping the panel outside the
    ``<details>`` is also what lets the device name stay a real link: a link inside a
    ``<summary>`` both navigates and toggles.
    """
    cell = _element("div", ["device-action"])
    disclosure = _element("details", ["device-disclosure"], name=DISCLOSURE_GROUP)
    # On the summary rather than the details: the summary is the control, and it is what
    # a screen reader announces along with its expanded state.
    summary = _element("summary", ["device-toggle"], **{"aria-controls": panel_id})
    summary += nodes.inline("", "Details", classes=["device-toggle-label"])
    summary += nodes.inline("", "Close", classes=["device-toggle-label", "is-open"])
    disclosure += summary
    cell += disclosure
    return cell


def _since_tag(since: str) -> nodes.Element:
    """The earliest release carrying the device, linked to that release.

    A release takes the word "Since", which the number alone does not say: on its own it
    reads as a version of the device rather than of what runs it. ``main`` takes no word,
    because it is not a floor -- it is the moving tip, and nothing is available "since"
    it. The bare chip, in the theme's primary color against the other rows' "Since",
    carries that on its own, and it points at the branch because there is no release for
    it to point at.

    The label restores the "v" the first two releases were tagged without, so the chips
    read as one series; the link keeps the spelling the repository actually uses.
    """
    group = _element("span", ["device-since"], inline=True)
    classes = ["device-since-branch"]
    if since == SINCE_MAIN:
        label = SINCE_MAIN
        target = f"{PROJECT_REPO}/tree/{SINCE_MAIN}"
        classes.append("is-main")
    else:
        group += nodes.inline("", "Since", classes=["device-since-label"])
        label = since if since.startswith("v") else f"v{since}"
        target = f"{PROJECT_REPO}/releases/tag/{since}"
    # Into the project's own repository, so no external marker: see PROJECT_REPO.
    group += nodes.reference("", label, refuri=target, classes=classes)
    return group


def _company_head(company: dict) -> nodes.Element:
    """The mark of the company behind the device, above the panel's columns.

    A single-color mark disappears against the other theme, so a company may ship a
    second file; the theme's only-light/only-dark classes do the swapping. The panel
    names the company nowhere else, so the mark is never decorative -- it carries the
    company name as its alternative text.
    """
    head = _line(["device-panel-head"])
    dark = company.get("logo_dark")
    head += nodes.image(
        uri=f"/{DEVICE_LOGO_DIR}/{company['logo']}",
        alt=company["name"],
        classes=["only-light"] if dark else [],
    )
    if dark:
        head += nodes.image(
            uri=f"/{DEVICE_LOGO_DIR}/{dark}",
            alt=company["name"],
            classes=["only-dark", "pst-js-only"],
        )
    return head


def _since_foot(since: str) -> nodes.Element:
    """Under the panel's columns rather than over them.

    Fewer than half the rows carry a company mark, so sharing the mark's row left the
    rest with a chip against an empty half -- and a row holding only a chip is a row
    spent on nothing. Below the columns it also reads as what it is: a note on the
    listing, not a heading over it.
    """
    foot = _line(["device-panel-foot"])
    foot += _since_tag(since)
    return foot


def _device_panel(record: dict, panel_id: str, docname: str) -> nodes.Element:
    panel = _element("div", ["device-panel"])
    panel["ids"] = [panel_id]
    company = record.get("company")
    if company:
        panel += _company_head(company)
    details = record["details"]
    for key, heading in DETAIL_COLUMNS:
        column = _element("div", ["device-panel-col"])
        title = _element("h4", ["device-panel-heading"])
        title += nodes.Text(heading)
        column += title
        entries = details.get(key) or ["—"]
        for entry in entries:
            column += _detail_line(entry, docname)
        panel += column
    since = record.get("since")
    if since:
        panel += _since_foot(since)
    return panel


def _planned_line(record: dict) -> nodes.paragraph:
    """One line rather than a row: a planned device has no input modes or setup yet."""
    line = _line(["device-planned"])
    line += nodes.inline("", "Planned", classes=["device-planned-label"])

    entry = nodes.inline("", "", classes=["device-planned-entry"])
    entry += nodes.reference("", record["name"], refuri=record["url"])
    entry += nodes.Text(f" — {record['note']}")
    line += entry

    number = _ISSUE_RE.search(record["tracking"]).group(1)
    track = nodes.reference(
        "",
        f"Tracking #{number}",
        refuri=record["tracking"],
        classes=["device-planned-track"],
    )
    line += track
    return line


def _matrix(
    devices: list[dict], planned: list[dict], docname: str, section: str
) -> nodes.Element:
    groups = DEVICE_SECTIONS[section]["groups"]
    name_column, value_column = DEVICE_SECTIONS[section]["columns"]

    wrapper = eco_block(classes=["device-matrix"])
    grid = eco_block(classes=["device-grid"])
    grid += _line(["device-col"], name_column)
    grid += _line(["device-col"], value_column)
    # No label over the buttons -- they describe themselves. The cell stays to carry the
    # header rule across the third column.
    grid += _line(["device-col", "device-col-action"])

    for group in groups:
        rows = [record for record in devices if record["group"] == group]
        if not rows:
            continue
        if len(groups) > 1:
            grid += _line(["device-group"], DEVICE_GROUPS[group])
        for record in rows:
            panel_id = f"device-{record['id']}-details"
            grid += _device_name(record)
            modes = _element("p", ["device-modes"], **{"data-label": value_column})
            modes += nodes.Text(record["modes"])
            grid += modes
            # Emitted with or without a panel: the cell carries the row's bottom border.
            if record.get("details"):
                grid += _disclosure(panel_id)
                grid += _device_panel(record, panel_id, docname)
            else:
                grid += _element("div", ["device-action"])

    wrapper += grid
    if DEVICE_SECTIONS[section]["planned"]:
        for record in planned:
            wrapper += _planned_line(record)
    return wrapper


class DeviceMatrix(Directive):
    """Render one section of ``devices.yaml`` as a table: name, value, Details panel."""

    has_content = False
    option_spec: ClassVar[dict] = {
        "section": lambda arg: directives.choice(arg, tuple(DEVICE_SECTIONS)),
        "data": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        section = self.options.get("section", "input")
        devices, planned = _load_devices(env, self.options.get("data", DEVICE_DATA))
        return [_matrix(devices, planned, env.docname, section)]


class EcoBlock(Directive):
    """``.. eco-block:: class-a class-b`` — a styled ``<div>`` around nested content.

    Use this instead of ``.. container::`` for anything the ecosystem CSS gives
    padding to; see the ``eco_block`` node for why.
    """

    required_arguments = 1
    final_argument_whitespace = True
    has_content = True

    def run(self) -> list[nodes.Node]:
        node = eco_block(classes=self.arguments[0].split())
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class DeviceCount(SphinxRole):
    """``:device-count:`input``` — rows in one table section, group, or ``all``.

    Planned entries are excluded; they render as a line under the table, not a row.
    """

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        target = self.text.strip()
        targets = ("all",) + tuple(DEVICE_SECTIONS) + tuple(DEVICE_GROUPS)
        if target not in targets:
            raise ExtensionError(
                f"device-count: unknown target {target!r}; expected one of {targets}"
            )
        devices, _ = _load_devices(self.env)
        if target in DEVICE_SECTIONS:
            groups = DEVICE_SECTIONS[target]["groups"]
        elif target == "all":
            groups = tuple(DEVICE_GROUPS)
        else:
            groups = (target,)
        count = sum(1 for record in devices if record["group"] in groups)
        return [nodes.inline("", str(count), classes=["eco-count"])], []


def setup(app):
    for node_class in (eco_block, eco_inline):
        app.add_node(
            node_class,
            html=(_visit_eco_block, _depart_eco_block),
            latex=(_passthrough, _passthrough),
            text=(_passthrough, _passthrough),
            man=(_passthrough, _passthrough),
            texinfo=(_passthrough, _passthrough),
        )
    app.add_directive("device-matrix", DeviceMatrix)
    app.add_directive("eco-block", EcoBlock)
    app.add_role("device-count", DeviceCount())
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
