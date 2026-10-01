.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Wheel contents SBOM and license evidence
========================================

Every published ``isaaccapture`` wheel carries a machine-readable inventory of what it
redistributes, plus the license text behind each obligation. You can inspect both
without installing the package, running it, or contacting a hosted service.

What a wheel carries
--------------------

.. code-block:: text

   isaaccapture-<version>-<abi>-<platform>.whl
   └── isaaccapture-<version>.dist-info/
       ├── METADATA                       # License-Expression + one License-File per text below
       ├── licenses/
       │   ├── LICENSE.md                 # this project's own license
       │   ├── LICENSES/*.txt             # texts the SPDX headers on this project's sources name
       │   ├── THIRD-PARTY-NOTICES.md     # component -> license -> text, with roles
       │   └── third-party/<component>/   # its own file verbatim, or a reference text, labelled
       └── sboms/
           └── <wheel filename>.spdx.json # SPDX 2.3 contents inventory

Every text there comes from this repository, or from a component the wheel redistributes,
or — for a component that ships no license file and states only an identifier in its source
headers — is the reference text that identifier names, beside the copyright line those
headers state. See `How licenses are determined`_.

``.dist-info/sboms/`` is what the
`binary distribution format <https://packaging.python.org/en/latest/specifications/binary-distribution-format/>`_
reserves for SBOM documents, and it is shared: ``auditwheel`` writes a CycloneDX document
of its own there describing the libraries it vendored. This collector looks for its
document by name and leaves anything else alone. Reports and manifests are published beside
the wheel instead.

Verifying a wheel
-----------------

Integrity and provenance are the packaging ecosystem's job, not this document's:

``RECORD``
   Every wheel already carries a ``sha256`` and size for each of its files, and
   installers check them. That is the integrity mechanism for a wheel's contents.

`PEP 740 <https://peps.python.org/pep-0740/>`_ attestations
   Signed attestations binding a publisher identity to an artifact's hash, served
   through PyPI's Integrity API. That is the provenance mechanism.

The SBOM *describes* what those files are — where each came from, under what license.
Its per-file digests are there so the description can be shown to belong to the wheel in
front of you, and so it can be checked against ``RECORD``; they are not a second root of
trust.

The document lists every member of the wheel except two: ``RECORD`` and the document
itself. Neither can carry its own digest, since ``RECORD`` hashes the document and the
document would hash ``RECORD``. ``check`` reads both against the archive directly.

.. code-block:: bash

   git clone --depth 1 https://github.com/NVIDIA/IsaacCapture
   uv run --project IsaacCapture/scripts/sbom isaaccapture-sbom check \
     --wheel isaaccapture-<version>-<abi>-<platform>.whl

The collector is a project of its own, :code-file:`scripts/sbom/pyproject.toml`, so its
dependencies come with it and are pinned there rather than at each call site. They are the
packages that own the formats involved — ``installer`` for ``RECORD``, ``packaging`` for
core metadata, ``license-expression`` for SPDX expressions, ``packageurl-python`` for
purls, ``pyelftools`` for ELF.

One command, widening with whatever you have. With just a wheel it confirms that

- the SBOM covers every file in the wheel except ``RECORD`` and the SBOM itself,
- each digest the SBOM records matches the bytes present, and agrees with ``RECORD``,
- ``RECORD`` itself agrees with the archive, so the wheel is still installable,
- the packaged notices are there, and every ``LicenseRef-`` has its text embedded.

Add ``--build-evidence`` and it also checks that the build accounts for every member;
add ``--manifest`` and it checks the wheel against the digest bound at publication and
the sidecars beside it. ``check-set`` runs the whole thing over every wheel in a release.

To read the inventory directly, unzip ``.dist-info/sboms/*.spdx.json`` and feed it to any
SPDX 2.3 consumer. CI validates each published document with ``spdx-tools`` before release.

Evidence published beside a wheel
---------------------------------

Tagged releases attach the following to the GitHub release, so the evidence outlives the
CI artifacts it travelled in:

``<wheel>.spdx.json``
   Byte-identical copy of the document embedded in that wheel.

``<wheel>.licenses.md`` / ``<wheel>.licenses.json``
   Component-to-license report: every redistributed component, the SPDX expression
   concluded for it, the file that expression was read from, how the component reaches
   the wheel, what is supplied by the installation environment instead, and what is
   excluded and why.

``<wheel>.build-evidence.json``
   Resolved commits, declared constraints, archive digests and build options recorded on
   the machine that built the wheel.

``isaaccapture-sbom-manifest.json``
   Maps each published wheel filename and SHA-256 to its evidence files and their digests.
   The wheel digest is bound here, after the last change to the wheel, so a wheel that was
   touched again no longer matches.

Downloading a whole release? Check the wheels and the evidence against the manifest in
one step:

.. code-block:: bash

   uv run --project IsaacCapture/scripts/sbom isaaccapture-sbom check-set \
     --manifest isaaccapture-sbom-manifest.json \
     --wheel-dir . --evidence-dir .

What is in scope
----------------

The document describes **the contents of one built wheel**: compiled-in static and
header-only dependencies, bundled shared libraries, vendored sources, SDK binaries and
packaged assets, for all six published variants (``x86_64`` and ``aarch64`` × CPython
3.11, 3.12, 3.13).

Deliberately **not** described by it:

- Build-time-only inputs — test frameworks, code generators, CMake modules, toolchains.
  They appear in the report under *Present in this build but not in this wheel*; a full
  build inventory is separate work.
- Source distributions and the ``pip install .`` path, which produce different contents.
- Containers and the web client, which are distributed on their own.
- Dependencies resolved at install time. The wheel's ``Requires-Dist`` metadata stays
  authoritative for those; the document records the requirement, never a license for it.

How the inventory is worked out
-------------------------------

There is no checked-in dependency list. Every entry is derived from the build that
produced the wheel, so a dependency that builds is a dependency that appears:

``build/_deps/*-src``
   FetchContent checkouts. ``git`` in each one supplies the commit, the remote, and from
   the remote the supplier, homepage and purl.

CMake's file API codemodel
   The build graph, as CMake itself describes it: which sources were compiled into which
   binary, which targets it depends on, and — through include paths — which header-only
   dependencies it was compiled against. Followed transitively, so a library linked into a
   library linked into the wheel is still reported. The top-level ``CMakeLists.txt`` asks
   for it with ``cmake_file_api()``, which is why the project requires CMake 3.27; reading
   the generator's own output instead would only work under Makefiles, since Ninja writes
   no ``link.txt``.

Archives in the repository
   Every member of every tarball is hashed. A wheel member that matches one is recorded as
   having been unpacked from that archive.

The working tree and the build tree
   What remains is matched against the project's own sources and its generated output.

**Every member of the wheel must be explained by one of those.** A file that is not stops
the build rather than being quietly omitted — that is the check that keeps the inventory
honest as the build changes.

Post-processing is recorded as such. ``patchelf``, ``auditwheel``'s RPATH rewrite and the
build's MJCF asset stripping all leave bytes that no longer hash to their origin, so those
members are reported as *derived from* a named input rather than as an exact copy, with
the input's digest wherever the input was hashed — which it is for an archive member, and
is not for a file matched only by its path.

How licenses are determined
---------------------------

Licenses are never inferred from this project's license, and never read from a list
someone maintained by hand. For each component the collector finds the license and notice
files the component itself ships — ``LICENSE``, ``COPYING``, ``BSD-LICENSE``, a ``LICENSES/``
pool, or, where upstream ships no file at all, the license section of its README — and
matches each text against a corpus of SPDX reference texts.

Those reference texts are a dependency like any other. :code-file:`deps/third_party/CMakeLists.txt`
fetches
``spdx/license-list-data`` at a pinned commit into ``_deps/license-list-data-src``, the same
mechanism and the same place as every other dependency — which means the collector
discovers it as a component of the build and reports it, alongside the rest, as fetched but
shipping nothing. Pinning is by commit, so no tag or branch upstream can move underneath a
build, and each report records the commit its identifications were made against.

Consulting the list is not shipping it. Every current identifier is available to match
against, so a dependency arriving under any of them is identified. Naming it in the
document is a second question: a document states the licence list it was built against, and
``license-expression`` — which a consumer validates with, and which
:code-file:`scripts/sbom/pyproject.toml` pins — ships its own, older copy. An id the fetched
list knows and that grammar does not cannot be published as an id, so it travels as a
``LicenseRef-`` carrying the text, and the component is listed under *License evidence gaps*
with that as the stated reason.

What the wheel carries for a third-party component is that component's own license file,
verbatim, copyright holders and all. Substituting a reference text would convey the terms
and drop the notice — for MIT and the BSD licenses those are the same file, differing by
the one line that names who licensed it to you — and it would replace the component's
actual grant with our reading of it, which is what the matching is for and not what it
decides. So a reference text is substituted only where there is nothing of the component's
own to package: it states an identifier in its source headers and ships no licence file.
The document says so for every such text, in as many words, because a reader cannot tell a
template from an author's own notice by looking at it.

A component can also state terms for one part of itself — a vendored asset set inside a
larger project, with its own ``LICENSE`` and its own holder. Those are read and packaged
too, but only where that directory reached the wheel, and they never change the component's
own licence: they name a holder for the bytes that shipped, which is what the notice
obligation is about.

The distribution's own declared licence is an obligation like any other. A wheel declaring
``License-Expression`` and shipping no text for it stops the build, the same as a
redistributed component with no terms; where the wheel ships none, the reference text for
each identifier is packaged from the pinned list and declared under PEP 639.

The notices file opens with the distribution's own licence and copyright holder, read
from the packaging metadata it was built from where no file it ships states one. A wheel
that ships only metadata carries no notice otherwise: the reference text its declared
licence packages is a template naming nobody, so a recipient holding the wheel alone could
not say who licensed it to them.

``licenseDeclared`` on the wheel is what the distribution says of itself. ``licenseConcluded``
is what this document found in it, so a wheel carrying a component whose terms the build
could not name states that expression alongside the declaration — a reader asking "what is
this artifact" is told about the proprietary payload rather than about the declaration only.

Never a file from this checkout. ``LICENSES/`` holds this repository's REUSE pool, whose
copies exist to license *this* project and carry its notices; packaging one as a third
party's terms would put our copyright above their code. A substituted text comes from the
pinned list, never from the pool — and under ``.dist-info/licenses/LICENSES/`` a wheel
carries whichever applies: the project's own pool files where it has them, and otherwise
the reference text for the licence it declares.

A match needs a canonical license text to be almost entirely present *and* to account for a
real share of the file: a tenth of a long agreement reproducing a license in an appendix is
not that agreement's own terms. The second threshold is the looser of the two, so a short
file quoting a license in full still matches — the bound it sets is on padding around a
text, not on a document that is mostly one. A file that genuinely carries two *distinct*
licenses is reported as carrying both; where one is a near-duplicate of the other, only the
match survives, and a disjunctive grant is not expressible here at all.

A text that matches nothing is not guessed at. It becomes a ``LicenseRef-`` whose full text
travels in the document, which says "these are the terms that shipped" without claiming to
know which license they are. Those components are listed in the report under *License
evidence gaps*.

A component that states a licence nowhere at all blocks publication, because packaging the
text is the obligation the wheel has to meet. Stating one in its source headers and nowhere
else is not that case: there the reference text for the identifier those headers name ships
instead, labelled in the document as a substitution rather than as the component's own words
— it conveys the terms and names no holder, so the headers' copyright line travels with it.

A shared library that none of those indexes explains did not come from this build, so the
last thing asked is the build machine itself: resolve the library's SONAME, and take its
license from the supplying package's copyright file, or from a license file beside the
library for toolchains that install outside the package manager. Resolving the SONAME is
what licenses the claim — a library the machine cannot find is reported as unexplained,
not attributed to something nobody located. No repair tool's directory name is consulted,
so a library the build produced is attributed to the build wherever it ends up.

Running the collector locally
-----------------------------

Against a configured build tree:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   cmake --install build

   # Configuring fetched the SPDX reference texts into
   # build/_deps/license-list-data-src/json, which is where `build` looks
   # unless --license-data says otherwise.
   uv run --project scripts/sbom isaaccapture-sbom build \
     --build-dir build --wheel install/wheels/<name>.whl --out-dir sbom
   uv run --project scripts/sbom isaaccapture-sbom check --wheel install/wheels/<name>.whl \
     --manifest sbom/manifest.json --build-evidence sbom/<name>.build-evidence.json

``build`` rewrites the wheel in place, so run it on a repaired wheel and do not touch the
wheel afterwards. Set ``SOURCE_DATE_EPOCH`` to make the result reproducible.

``--project`` points ``uv`` at the collector, not at this repository: it resolves the
collector's own pins and runs it from wherever you are. The checkout it inventories is the
one you are standing in, or ``--repo-root``.

Maintaining it
--------------

Adding a dependency takes no SBOM change: declare it the way you already would — a
``FetchContent_Declare``, a tarball under ``deps/`` — and it appears in the next wheel's
inventory with its identity, its license and the route it took to get there.

Two things do need attention:

- **A component that states its license nowhere stops the build.** The fix is to obtain the
  terms, not to record an assumption. Stating one only in its source headers is not that
  case: the reference text for that identifier ships, labelled as a substitution.
- **A new member that the build cannot explain stops the build.** That means something
  reached the wheel by a route the collector cannot see; teach it that route rather than
  excluding the file.

Moving to a newer SPDX license list is a dependency bump like any other: change
``SPDX_LICENSE_DATA_COMMIT`` in :code-file:`deps/third_party/CMakeLists.txt`, with its release
comment. There is no separate list of identifiers to extend — the whole SPDX list comes
with the checkout.
