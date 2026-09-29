<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docs — agent notes

Rules that the docs tree cannot state for itself. Anything that describes how a page is
currently put together belongs next to it instead — in the `.rst`, in the YAML header, in
the CSS — where it moves when the code moves.

## Build the way CI does

`make current-docs` wipes `build/current` and runs `sphinx-build -W --keep-going`, which
is what CI runs, so a new warning fails the build. Check a docs change with that target
rather than a bare `sphinx-build`.

`make multi-docs` is the other half, and it constrains how extensions are written:
sphinx-multiversion builds every whitelisted ref with `-c` pointing at the checkout that
invoked it, so `app.confdir` is that checkout while `app.srcdir` is the ref being built.
Resolve paths from one of those two, never from the working directory or the repo root.

The config is therefore a site-wide input, not a per-version one: the deploy job takes
`docs/` from `main` before building, or a release or tag push renders main's pages with an
older `conf.py` and anything its extensions provide disappears. A push runs the workflow
file from its own ref, so that step has to stay alive on every release branch that still
receives pushes.

Images follow `.gitattributes` like everything else, Git LFS included. A checkout without
LFS holds pointer text and Sphinx copies it into the site verbatim, so a page that looks
right locally can still ship broken artwork. Commit artwork together with the page or data
that references it.

## Never wrap styled layout in `.. container::`

Docutils renders `container` as `class="docutils container"`, and Bootstrap claims the same
class name, so the theme ships `.docutils.container {padding-left: unset; padding-right:
unset}`. That outranks a plain class selector and silently flattens horizontal padding —
panels end up with text flush against their border. Use the `eco-block` directive, or the
`eco_block` node from Python, for any element the CSS pads.

## The docs ship no JavaScript

The docs set no `html_js_files`, and interactivity comes from platform features instead:
the device table's Details rows are one `<details>` per row sharing a `name`, and CSS alone
ties an open disclosure to its panel. Two rules generalize from that. An affordance that
would need a script does not go on the page. And where the CSS feature behind one is not
universal, the fallback reveals content rather than hiding it.

When docutils has no node for an element you need, give `eco_block` an `html_tag` and
`html_attributes` rather than emitting a `nodes.raw` blob, so the contents stay real nodes
for non-HTML builders.

## The external marker follows the target, not the record

The docs site and the IsaacTeleop repository are both this project, so a link into either
renders with no arrow and no screen-reader "(external)" — tracking links included. Only
third-party domains get the marker, and other GitHub organizations (`isaac-sim`) count as
third-party. `ecosystem_grid.PROJECT_REPO` is the one place that boundary is written down.

## The Ecosystem page is tables, not cards

Every section of `overview/ecosystem.rst` is a `device-matrix` table fed by
`source/_data/devices.yaml`, whose header documents the schema; the extension is
`source/_ext/ecosystem_grid.py`. The card grid that used to sit above the table was
deleted, not disabled, so "partner" is not a word to reintroduce in markup, data, or class
names — and a second data file describing companies is the shape that was removed.

That header, plus the four section dividers, is the only commentary the data file carries.
Why a row has no `since`, who cleared a contact, why a plugin is named generically — none
of it annotates the row: state it in a field the page renders, put a rule that outlives the
row in the header, and leave the rest to the commit message.

## What a company sends is copy and artwork, not a record

A logo lives in `source/_static/logos/`, is named after the company rather than the device,
and renders at the top of that row's Details panel. Every in-row position was tried first —
before the name, after it, ahead of the Details button — and each one crowded a table whose
job is to be scanned, so treat panel placement as settled.

Marketing copy has nowhere to land, and that is the boundary to hold when a company asks:
the table has no description column, so a capability claim enters `modes` only if our own
device page documents it, links belong in the Details panel, and the sentence itself is
dropped. The build keeps `company` that narrow by rejecting every field but `name`, `logo`,
and `logo_dark`.
