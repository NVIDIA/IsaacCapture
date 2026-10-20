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
       │   ├── LICENSES/<id>.txt          # the pool texts license-files in pyproject.toml names
       │   ├── THIRD-PARTY-NOTICES.md     # component -> license -> text, with roles
       │   └── third-party/<component>/   # its own file verbatim, or a reference text, labelled
       └── sboms/
           └── <wheel filename>.spdx.json # SPDX 2.3 contents inventory

Every text there comes from this repository, or from a component the wheel redistributes,
or — for material checked into this repository, whose terms its REUSE metadata declares —
is the reference text each declared identifier names, beside the copyright notice it is
declared under. See `How licenses are determined`_.

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
   through PyPI's Integrity API. That is the ecosystem's mechanism for binding
   provenance to a distribution; these releases are not published with attestations.

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
purls, ``pyelftools`` for ELF, ``reuse`` for REUSE metadata. ``reuse`` reads files through
libmagic, so ``build`` needs it (``libmagic1`` on Ubuntu); ``check`` does not.

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

Downloading a whole release? Put its wheels and evidence files in a directory of their
own, then check them against the manifest in one step:

.. code-block:: bash

   uv run --project IsaacCapture/scripts/sbom isaaccapture-sbom check-set \
     --manifest release/isaaccapture-sbom-manifest.json \
     --wheel-dir release --evidence-dir release

Any command reading a manifest refuses one that lists no wheel. ``check-set`` also fails a
wheel in ``--wheel-dir`` that the manifest does not list, which is why the directory holds
the release and nothing else.

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
   FetchContent trees. ``git`` in each checkout supplies the commit, the remote, and from
   the remote the supplier, homepage and purl. It is asked only of a tree that is a
   repository of its own: git otherwise answers for whichever checkout encloses the build
   directory, or for whatever repository ``GIT_DIR`` names, which is therefore cleared. Any
   other tree is identified by its declaration — ``GIT_REPOSITORY`` and ``GIT_TAG``, or
   ``URL`` and ``URL_HASH`` — from the stamps ExternalProject writes beside the populate
   sub-build, or under ``CMakeFiles/fc-stamp`` once policy ``CMP0168`` is ``NEW``, and
   from the clone script the git stamp names. A stamp outlives a change of method, so where
   both are present the tree decides: a git clone is a checkout of its own. The version is
   the declared tag, or the ref a GitHub URL names, verbatim, so a tag gives one version
   whether declared by tag or as a GitHub archive; any other archive's version is read
   from its name, and its supplier is not asserted, since the host serving a download is
   not who supplies it. A download location is stated only for an ``http``, ``https``,
   ``ftp`` or ``ftps`` archive URL, or a git remote reached over ``http``, ``https``,
   ``ssh`` or ``git``, and never with its credentials, query or fragment.

vcpkg's install records
   Where the build uses vcpkg, the ``vcpkg/info`` lists under ``VCPKG_INSTALLED_DIR`` (by
   default the build tree's ``vcpkg_installed``): which port installed each file.

CMake's file API codemodel
   The build graph, as CMake itself describes it: which sources were compiled into which
   binary, which targets it depends on, what its link line names, and —
   through include paths — which header-only dependencies it was compiled against.
   Followed transitively, so a library linked into a library linked into the wheel is still
   reported. The top-level ``CMakeLists.txt`` asks
   for it with ``cmake_file_api()``, which CMake 3.27 added — so a wheel with an SBOM
   needs at least that, though the project builds without it. Reading the generator's own
   output instead would only work under Makefiles, since Ninja writes no ``link.txt``.

The compiler's dependency output
   Which headers each object read: the per-object depfiles and ``compiler_depend.make``
   under the Makefile generators, ``ninja -t deps`` under Ninja — so a Ninja build is
   collected where ``ninja`` is on ``PATH``, and the collector says so once when it is
   not. Any target whose records do not cover every source it compiles falls back to
   what it lists and includes.

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

Licenses are never inferred from this project's license, and never read from a list someone
maintained by hand. REUSE metadata is not such a list: it is how this repository declares
the terms of each file it holds — an in-file ``SPDX-License-Identifier``, a ``.license``
sidecar or a ``REUSE.toml`` annotation, one declaration — and the pre-commit hook lints it
for source files. The collector reads it for every file here that REUSE covers, whatever its
type, and for every source here the build compiled or included: a file identified as a third
party's (see below) is held to the per-file gate even where the hook does not lint it, and a
file naming no holder at all counts as this project's own. A fetched component's licence is
not read from its source headers, ``.license`` sidecars or a ``REUSE.toml``: for each
fetched tree the collector finds the license and notice files the component itself ships —
``LICENSE``, ``COPYING``, ``BSD-LICENSE``, a ``LICENSES/`` pool, or, where none of those at
its root is a grant, the license section of its README — and matches each text against a
corpus of SPDX reference texts. An archive in the repository is read only for the license
files at its root.

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
list knows and that grammar does not cannot be published as an id, so the text it was read
from travels as a ``LicenseRef-`` — a REUSE declaration naming one travels whole, since an
``OR`` or a ``WITH`` cannot be kept in part — and the component is listed under *License
evidence gaps* with that as the stated reason.

What the wheel carries for a third-party component is that component's own license file,
verbatim, copyright holders and all. Substituting a reference text would convey the terms
and drop the notice — for MIT and the BSD licenses those are the same file, differing by
the one line that names who licensed it to you — and it would replace the component's
actual grant with our reading of it, which is what the matching is for and not what it
decides. So a fetched component ships its own text, or, having none, blocks the build
(below).

Third-party material checked into this repository rather than fetched is the exception: it
is identified by the copyright holder it is declared under — a ``.license`` sidecar or a
``REUSE.toml`` annotation where REUSE applies one, and the first copyright line of the file
itself unless REUSE applies those in the file's place — and its terms are the identifiers
REUSE gives each of its files, under REUSE's own precedence. The reference text for each
licence it names ships, and the expression keeps its structure, so a choice stays a choice;
an exception's text ships too where the expression travels as a ``LicenseRef-``, above. The
document says so for every such text, in as many words, because a reader cannot tell a
template from an author's own notice by looking at it.

A component can also state terms for one part of itself — a vendored asset set or a bundled
library inside a larger project, with its own ``LICENSE`` and its own holder. Those are read
and packaged too, but only where something under that directory reached the wheel: a file
copied or derived from it — a copy the build made into the component's own build tree
included — a source compiled from it, or a header the compiler's dependency output says it
read. Where a target compiles something its dependency output does not account for, the
sources it lists count instead, and so does any include directory above or below that
directory. An archive of the component linked in keeps all of them, since what it holds is
unknown. A REUSE ``.license`` sidecar covers only the file it annotates. Which of those kept
each text is in the build evidence.

A bundled library compiled in ships its texts and holders, each listed as terms stated for
part of the component; the component's concluded licence stays its own grant. In the evidence
listed for a text, one that matches several identifiers sharing one reference text —
``GPL-3.0-only`` and ``GPL-3.0-or-later`` — is listed as matching either, not both.

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
could not name states that component's ``LicenseRef-`` alongside the declaration — a reader
asking "what is this artifact" is told about the proprietary payload rather than about the
declaration only. Where nothing at all is concluded for a component, neither is anything
for the wheel: it states ``NOASSERTION``.

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

Every grant a component ships is accounted for in its conclusion. A text that matches
nothing is not guessed at: it travels in the document as the component's ``LicenseRef-``,
which says "these are the terms that shipped" without claiming to know which license they
are, AND-ed with what its other texts are identified as — an MIT ``LICENSE`` beside an
unmatched EULA concludes ``LicenseRef-<component> AND MIT``. A text matching identifiers
that share one reference text names the one the others extend by a suffix — ``MPL-2.0``
beside ``MPL-2.0-no-copyleft-exception``, ``OFL-1.1`` beside its ``-RFN`` variants — since
that is SPDX's name for the unmodified text. Where there is none, as for ``GPL-3.0-only``
and ``GPL-3.0-or-later``, the choice is the licensor's, stated in a notice and not in the
text, so the text is carried too. So is a text its identifiers account for less than four
fifths of, its copyright notices aside, beside the identifiers it names: the rest may be
terms no identifier names, as in fmt's MIT with its exception paragraph. The ``LicenseRef-`` carries exactly the texts
carried. Those components are listed in the report under *License evidence gaps*.

A component whose only texts are a REUSE ``LICENSES/`` pool ships them, and that meets
the obligation below, but a pool says what each file is under, not what the component is.
Nothing is concluded for it, so its licence is ``NOASSERTION``, and it is listed under
*License evidence gaps* too — as is a component whose grants name nothing the grammar
accepts.

A component with no licence text the collector reads blocks publication, because packaging
the text is the obligation the wheel has to meet. For a fetched component that is no licence
file at its root and, for a fetched tree, none in a ``LICENSES/`` pool and no licence
section in its README; a licence file in a subdirectory covers only that part, and terms
only its REUSE info states do not count. For material checked in here it is any one file
whose REUSE info names no licence, names one with no current SPDX reference text (a
``LicenseRef-`` among them, even beside one that has a text), states an expression that
does not parse, or names an id that is neither SPDX-listed nor a ``LicenseRef-`` or
``DocumentRef-``; an exception needs only to be listed as an exception. The build names
each such file and why, and another file of the same holder does not lend it its terms. The
remedy is to record upstream's terms as an ``SPDX-License-Identifier`` in the header, a
``.license`` sidecar or a ``REUSE.toml`` annotation. A tag that is not one — quoted in a
string, say, or prose that mentions the tag — goes between ``REUSE-IgnoreStart`` and
``REUSE-IgnoreEnd``, or is set aside by a ``.license`` sidecar or a ``REUSE.toml``
annotation with ``precedence = "override"``. Where a checked-in third-party file's REUSE
info declares an identifier, the reference text for it ships, labelled in the document as a
substitution rather than as the component's own words — it conveys the terms and names no
holder, so the file's copyright notice travels with it.

A shared library that none of those indexes explains did not come from this build, so the
last thing asked is the build machine itself: resolve the library's SONAME, and take its
license from the supplying package's copyright file, or from a license file beside the
library for toolchains that install outside the package manager. Resolving the SONAME is
what licenses the claim — a library the machine cannot find is reported as unexplained,
not attributed to something nobody located. No repair tool's directory name is consulted,
so a library the build produced is attributed to the build wherever it ends up. Its
copyright notice is read only from the ``Copyright:`` fields of a machine-readable Debian
copyright file, leaving out stanzas that cover only the Debian packaging; any other
text — an EULA, a free-form copyright file — quotes the notices of what it bundles, so
none is asserted, and the packaged text still carries them all.

Static code leaves no ``DT_NEEDED`` entry behind, so the link line is where it is read —
every fragment CMake records for it, ``-Wl,`` and ``-Xlinker`` words included, with the
values of options known to take one set aside. A path on it is judged by its content, not
its name: an archive or an object file is static code, a shared library is left to
``DT_NEEDED``, and a GNU ``ld`` script — ``libc.so`` is one, naming ``libc_nonshared.a`` —
is followed to the files and ``-l`` names its ``INPUT`` and ``GROUP`` commands give, a
relative name looked up beside the script first, as ``ld`` does. A
``-l`` name is resolved the way ``ld`` resolves it: through the line's ``-L`` directories,
then the directories the compiler hands the linker — CMake's toolchains reply, or the
compiler's own ``-print-search-dirs`` for a language that reply does not cover — then the
linker's built-in ``SEARCH_DIR`` list from ``ld --verbose``, taking a shared library before
an archive unless ``-Bstatic`` is in force. A leading ``=`` in either names the sysroot —
the target's ``CMAKE_SYSROOT``, else the compiler's — as does an absolute name in a script
inside it. ``--push-state`` and ``--pop-state`` are
honoured, and the driver's ``-static`` governs the whole line. The lines read are the
member's own and those of shared libraries it depends on; an executable or module it
merely waits for, such as a code generator, links nothing into it, and neither does
anything reached only through one.

What is found is then attributed:

- an archive or object a target of this build produced is that target's contribution;
- one inside a fetched component's tree is that component's;
- one vcpkg installed belongs to the port whose install record
  (``vcpkg/info/<port>_<version>_<triplet>.list``) names it. The port is identified by the
  upstream source its own ``share/<port>/vcpkg.spdx.json`` says it built — a ``pkg:github``
  purl where its one source is on GitHub, ``pkg:generic`` with no supplier where it is not
  or where several sources name no single upstream, since purl has no vcpkg type — at the
  upstream version, without vcpkg's ``#<n>`` recipe revision. Its licence is read from
  ``share/<port>/copyright``;
- one tracked in this repository belongs to the holder its REUSE info names, and is this
  project's own where that is this project;
- one from the build machine is asked about the same way as a shared library: the package
  that owns it and that package's copyright file, or, for an archive under a CUDA toolkit,
  the license file the toolkit ships;
- an archive with no members links no code — glibc 2.34 left ``librt.a``, ``libpthread.a``
  and ``libdl.a`` as such stubs — and is recorded in the build evidence rather than as a
  component.

Each component found this way is recorded with ``STATIC_LINK``. Anything else stops the
build: a name that resolves to nothing, a fragment that does not parse, a path that is none
of those kinds — which is also what an unknown option's value looks like — and an archive
inside the build tree that no target produced and no install record names, which is where
ExternalProject leaves its output. Not inventoried: what the compiler driver adds without
naming it on the link line, ``libgcc`` and the C runtime's start files among it; and a
vcpkg port's headers read through an include directory other ports also install into.
Every port shares one include root, so an include directory credits a port only where each
file vcpkg installed under it is that port's; headers read from the shared root are
credited only through a linked archive of the same port.

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

These do need attention:

- **A component with no license text the collector reads stops the build.** The fix is to
  obtain the terms, not to record an assumption; a fetched component's REUSE info, source
  headers included, is not read for them. For a file checked in here, obtain the terms,
  then record them as REUSE licence info naming current SPDX-listed identifiers; a stray
  tag that does not parse or names no such id is set aside instead. Both remedies are
  under `How licenses are determined`_.
- **A new member that the build cannot explain stops the build.** That means something
  reached the wheel by a route the collector cannot see; teach it that route rather than
  excluding the file.
- **Static code on a link line that the collector cannot trace stops the build.** Link
  what it can trace — a target of this build, a fetched dependency, a system package or a
  known toolkit — or teach it the route that input took, such as an ExternalProject install
  into the build tree.

Moving to a newer SPDX license list is a dependency bump like any other: change
``SPDX_LICENSE_DATA_COMMIT`` in :code-file:`deps/third_party/CMakeLists.txt`, with its release
comment. There is no separate list of identifiers to extend — the whole SPDX list comes
with the checkout.
