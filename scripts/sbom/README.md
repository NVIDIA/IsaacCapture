<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wheel SBOM collector

`isaaccapture-sbom` derives a wheel's contents SBOM and license evidence from the build
that produced it, packages both into the wheel, and checks the result. What a wheel
carries and how to verify one are on the user page,
[Wheel contents SBOM and license evidence](https://nvidia.github.io/IsaacCapture/main/references/sbom.html)
(source: [`docs/source/references/sbom.rst`](../../docs/source/references/sbom.rst)).
This file is how it works.

Where to look: running or testing it, [Running it](#running-it); a member the build
cannot explain, [How the inventory is worked out](#how-the-inventory-is-worked-out); a
wrong or missing license, [How licenses are determined](#how-licenses-are-determined); a
library from the build machine or static code,
[Libraries from the build machine](#libraries-from-the-build-machine) and
[Static code on the link line](#static-code-on-the-link-line); bumping the SPDX list,
[Reference texts](#reference-texts); what stops the build and how to fix it,
[What stops the Release build](#what-stops-the-release-build).

## Running it

The collector is a project of its own; its dependencies are pinned in
[`pyproject.toml`](pyproject.toml). `reuse` reads files through libmagic, so `build`
needs it (`libmagic1` on Ubuntu); `check` does not.

Build and install a Release wheel as in
[Building from source](../../docs/source/development/build_from_source/index.rst), then:

```bash
# `build` reads the SPDX reference texts configuring fetched into
# build/_deps/license-list-data-src/json, unless --license-data says otherwise.
uv run --project scripts/sbom isaaccapture-sbom build \
  --build-dir build --wheel install/wheels/<name>.whl --out-dir sbom
uv run --project scripts/sbom isaaccapture-sbom check --wheel install/wheels/<name>.whl \
  --manifest sbom/manifest.json --build-evidence sbom/<name>.build-evidence.json
```

- `build` rewrites the wheel in place, so run it on a repaired wheel and do not touch
  the wheel afterwards. Set `SOURCE_DATE_EPOCH` to make the result reproducible.
- `auditwheel` writes its own CycloneDX document into `.dist-info/sboms/`; the collector
  finds its document there by name and leaves the rest alone.
- `--project` makes `uv` resolve the collector's pins and run it from anywhere. It
  inventories the checkout you are in, or `--repo-root`.

### What `check` verifies

The SBOM's per-file digests show that the description belongs to the wheel in front of
you and agrees with its `RECORD`; they are not a second root of trust. The document
lists every member except `RECORD` and itself, since neither can carry its own digest;
`check` reads both against the archive directly. With just a wheel it confirms that

- the SBOM covers every file in the wheel except `RECORD` and the SBOM itself,
- each digest the SBOM records matches the bytes present, and agrees with `RECORD`,
- `RECORD` itself agrees with the archive, so the wheel is still installable,
- the packaged notices are there, and every `LicenseRef-` has its text embedded.

`--build-evidence` also checks that the build accounts for every member; `--manifest`
checks the wheel against the digest bound at publication and the sidecars beside it.
`check-set` runs the whole thing over every wheel in a release, and fails a wheel in
`--wheel-dir`, or a file in `--evidence-dir`, that the manifest does not name.
`merge-manifests` refuses a wheel name whose builds produced different bytes, SBOMs or
license reports. Only build evidence may differ; the first manifest's in path order (in
CI, artifact-name order) is bound, and `--evidence-out` copies exactly the bound files.
Any command reading a manifest refuses one that lists no wheel. CI validates each
published document with `spdx-tools` before release.

### Tests

From the repository root:

```bash
uv run --project scripts/sbom --extra dev pytest scripts/sbom/tests
```

They read the file API codemodel, and the SPDX list where `build` finds it without
`--license-data`, from the build tree `ISAACCAPTURE_SBOM_BUILD_DIR` names, else `build/`.
An unnamed `build/` that lacks them skips the tests; a named one fails them. A configured
tree is enough, so CI runs this command in a job of its own, beside the product build, and
publishes nothing unless it passes.

## How the inventory is worked out

Every inventory entry is derived from the build, from these sources:

**`build/_deps/*-src`** — FetchContent trees.

- A tree that is a git repository of its own is asked for its commit and remote, and
  from the remote the supplier, homepage and purl. Git is asked of no other tree, since
  it would answer for whichever checkout encloses the build directory, or for whatever
  `GIT_DIR` names; `GIT_DIR` is therefore cleared.
- Any other tree is identified by its declaration — `GIT_REPOSITORY` and `GIT_TAG`, or
  `URL` and `URL_HASH` — from the stamps ExternalProject writes beside the populate
  sub-build, or under `CMakeFiles/fc-stamp` once policy `CMP0168` is `NEW`, and from the
  clone script the git stamp names. A stamp outlives a change of method, so where both
  are present the tree decides: a git clone is a checkout of its own.
- The version is the declared tag, or the ref a GitHub URL names, verbatim, so a tag
  gives one version whether declared by tag or as a GitHub archive. Any other archive's
  version is read from its name, and its supplier is not asserted: the host serving a
  download is not who supplies it.
- A download location is stated only for an `http`, `https`, `ftp` or `ftps` archive
  URL, or a git remote reached over `http`, `https`, `ssh` or `git`, and never with its
  credentials, query or fragment.

**vcpkg's install records** — the `vcpkg/info` lists under `VCPKG_INSTALLED_DIR` (by
default the build tree's `vcpkg_installed`): which port installed each file.

**CMake's file API codemodel** — the build graph as CMake describes it: which sources
were compiled into which binary, which targets it depends on, what its link line names
and, through include paths, which header-only dependencies it was compiled against.
Followed transitively, so a library linked into a library linked into the wheel is still
reported. The top-level `CMakeLists.txt` requests it with `cmake_file_api()`; the
project builds without it, but a wheel with an SBOM needs it, hence the CMake version in
[Prerequisites](../../docs/source/development/build_from_source/index.rst#prerequisites).
Do not read the generator's own output instead: Ninja writes no `link.txt`.

**The compiler's dependency output** — which headers each object read: the per-object
depfiles and `compiler_depend.make` under the Makefile generators, `ninja -t deps` under
Ninja. A Ninja build is collected where `ninja` is on `PATH`; the collector says so once
when it is not. A target whose records miss a source it compiles falls back to its
listed sources and include directories.

**Archives in the repository** — every member of every tarball is hashed; a wheel member
that matches one is recorded as unpacked from that archive.

**The working tree and the build tree** — what remains is matched against the project's
own sources and its generated output.

**The build machine** — for a shared library none of the above explains; see
[Libraries from the build machine](#libraries-from-the-build-machine).

A member none of these explains stops the build
([below](#what-stops-the-release-build)). Adding a dependency needs no change here:
declare it as usual — a `FetchContent_Declare`, a tarball under `deps/` — and it appears
in the next wheel's inventory.

Post-processing is recorded as such. `patchelf`, `auditwheel`'s RPATH rewrite and the
build's MJCF asset stripping leave bytes that no longer hash to their origin, so those
members are reported as *derived from* a named input rather than as an exact copy, with
the input's digest wherever the input was hashed: an archive member is, a file matched
only by its path is not.

A file's `SPDXID` is its path in the distribution, reduced to the characters an SPDX
identifier allows, plus a short digest of the unreduced path, with the versioned
`.dist-info` and `.data` directories counted as fixed.

### Libraries from the build machine

A shared library none of the build's indexes explains did not come from this build, so
the build machine is asked: resolve its SONAME and take its license from the supplying
package's copyright file, or from a license file beside the library for toolchains that
install outside the package manager. A library the machine cannot find is reported as
unexplained, not attributed.

- The `pkg:deb` purl is namespaced by the archive publishing the installed version: the
  `Origin` apt records for it (`ubuntu` for Ubuntu's, `nvidia` for NVIDIA's CUDA
  repository). Where no archive lists the installed version — superseded, or a local
  rebuild — the archives listing its other versions decide, and the build evidence says
  so.
- Where no vendor can be named — no archive lists the package (a local `.deb`); the
  deciding archives disagree, ignoring case, or name no `Origin`; or the `Origin` is a
  URL or host name, which a purl namespace cannot be — the package is
  `pkg:generic/<package>@<version>`. The build machine's own distribution is not
  consulted.
- No repair tool's directory name is consulted, so a library the build produced is
  attributed to the build wherever it ends up.
- Copyright notices are read only from the `Copyright:` fields of a machine-readable
  Debian copyright file, leaving out stanzas that cover only the Debian packaging. Any
  other text (an EULA, a free-form copyright file) quotes the notices of what it
  bundles, so none is asserted; the packaged text still carries them all.

### Static code on the link line

Static code leaves no `DT_NEEDED` entry, so the link line is read: every fragment CMake
records for it, `-Wl,` and `-Xlinker` words included, with the values of options known
to take one set aside.

- A path is judged by its content, not its name: an archive or object file is static
  code, a shared library is left to `DT_NEEDED`, and a GNU `ld` script (`libc.so`,
  naming `libc_nonshared.a`) is followed to the files and `-l` names its `INPUT` and
  `GROUP` commands give, a relative name looked up beside the script first, as `ld`
  does.
- A `-l` name is resolved as `ld` resolves it: the line's `-L` directories, then the
  directories the compiler hands the linker (CMake's toolchains reply, or the compiler's
  own `-print-search-dirs` for a language that reply does not cover), then
  `ld --verbose`'s built-in `SEARCH_DIR` list, taking a shared library before an archive
  unless `-Bstatic` is in force. A leading `=` in either names the sysroot (the target's
  `CMAKE_SYSROOT`, else the compiler's), as does an absolute name in a script inside it.
  `--push-state` and `--pop-state` are honoured, and the driver's `-static` governs the
  whole line.
- The lines read are the member's own and those of shared libraries it depends on. An
  executable or module it merely waits for, such as a code generator, links nothing into
  it, nor does anything reached only through one.

What is found is attributed:

- an archive or object a target of this build produced is that target's contribution;
- one inside a fetched component's tree is that component's;
- one vcpkg installed belongs to the port whose install record
  (`vcpkg/info/<port>_<version>_<triplet>.list`) names it. The port is identified by the
  upstream source its `share/<port>/vcpkg.spdx.json` says it built — a `pkg:github` purl
  where its one source is on GitHub, `pkg:generic` with no supplier where it is not or
  where several sources name no single upstream, since purl has no vcpkg type — at the
  upstream version, without vcpkg's `#<n>` recipe revision. Its license is read from
  `share/<port>/copyright`;
- one tracked in this repository belongs to the holder its REUSE info names, and is this
  project's own where that is this project;
- one from the build machine is asked about as a shared library is: the owning package
  and its copyright file or, for an archive under a CUDA toolkit, the license file the
  toolkit ships;
- an archive with no members links no code — glibc 2.34 left `librt.a`, `libpthread.a`
  and `libdl.a` as such stubs — and is recorded in the build evidence rather than as a
  component.

Each component found this way is recorded with `STATIC_LINK`; anything else stops the
build ([below](#what-stops-the-release-build)).

Not inventoried: what the compiler driver adds without naming it on the link line
(`libgcc` and the C runtime's start files among it), and a vcpkg port's headers read
through an include directory other ports also install into. Every port shares one
include root, so an include directory credits a port only where each file vcpkg
installed under it is that port's; headers read from the shared root are credited only
through a linked archive of the same port.

## How licenses are determined

Licenses are never inferred from this project's license, and never read from a list
maintained by hand.

### Where terms are read from

- **Files in this repository** — from their REUSE info
  ([License headers](../../docs/source/development/license.rst),
  [REUSE specification](https://reuse.software/spec/)). The pre-commit hook lints it for
  source files; the collector reads it for every file REUSE covers, whatever its type,
  and for every source here the build compiled or included. A file identified as a third
  party's (below) is held to the per-file gate even where the hook does not lint it; a
  file naming no holder at all counts as this project's own.
- **Fetched components** — from the license and notice files the component ships:
  `LICENSE`, `COPYING`, `BSD-LICENSE`, a `LICENSES/` pool or, where none of those at its
  root is a grant, the license section of its README. Each text is matched against the
  SPDX reference texts. Its source headers, `.license` sidecars and `REUSE.toml` are not
  read.
- **Archives in the repository** — only the license files at the archive's root.

### Reference texts

[`deps/third_party/CMakeLists.txt`](../../deps/third_party/CMakeLists.txt) fetches
`spdx/license-list-data` at a pinned commit into `_deps/license-list-data-src`, like any
other dependency, so the build evidence records it with `"in_this_wheel": false`. Each
report records the commit its identifications were made against.

Every current identifier is available to match against. Naming one in the document is a
separate question: the document states the license list it was built against, and
`license-expression` — which consumers validate with and `pyproject.toml` pins — ships
its own, older copy. An id the fetched list knows and that grammar does not cannot be
published as an id, so the text it was read from travels as a `LicenseRef-`. A REUSE
declaration naming one travels whole, since an `OR` or a `WITH` cannot be kept in part.
The component is listed under *License evidence gaps* with that reason.

To move to a newer list, change `SPDX_LICENSE_DATA_COMMIT` in
`deps/third_party/CMakeLists.txt`, with its release comment, and the entry in
[`deps/README.md`](../../deps/README.md#spdx-license-list-data). There is no separate
list of identifiers to extend.

### What ships

- **A fetched component** ships its own license file verbatim, copyright holders and
  all. Do not substitute a reference text: it would drop the notice — for MIT and the
  BSD licenses the difference is the line naming who licensed it to you — and replace
  the component's actual grant with our reading of it.
- **Third-party material checked in here** is identified by the copyright holder it is
  declared under: a `.license` sidecar or `REUSE.toml` annotation where REUSE applies
  one, and the file's own first copyright line unless REUSE applies those in its place.
  Its terms are the identifiers REUSE gives each of its files, under REUSE's own
  precedence. The reference text for each license it names ships, labelled in the
  document as a substitution, since a reader cannot tell a template from an author's
  notice; the file's copyright notice travels with it. The expression keeps its
  structure, so a choice stays a choice; an exception's text ships too where the
  expression travels as a `LicenseRef-`.
- **Terms stated for part of a component** — a vendored asset set or bundled library
  with its own `LICENSE` and holder — are read and packaged only where something under
  that directory reached the wheel: a file copied or derived from it (including a copy
  the build made into the component's own build tree), a source compiled from it, or a
  header the compiler's dependency output says it read. Where a target compiles
  something its dependency output does not account for, the sources it lists count
  instead, and so does any include directory above or below that directory. A linked
  archive of the component keeps all of them, since what it holds is unknown. A
  `.license` sidecar covers only the file it annotates. The build evidence says which of
  those kept each text.
- **A bundled library compiled in** ships its texts and holders, each listed as terms
  stated for part of the component; the component's concluded license stays its own
  grant.
- **The distribution's own declared license** is an obligation like any other. Where the
  wheel ships no text for it, the reference text for each identifier is packaged from
  the pinned list and declared under PEP 639.
- **The notices file** opens with the distribution's own license and copyright holder,
  read from the packaging metadata where no file it ships states one. Otherwise a wheel
  shipping only metadata would carry a template naming nobody.
- **Never a file from `LICENSES/`** as a third party's terms: that pool licenses *this*
  project and carries its notices. Substituted texts come from the pinned list. Under
  `.dist-info/licenses/LICENSES/` a wheel carries the project's own pool files where it
  has them, and otherwise the reference text for the license it declares.

`licenseDeclared` on the wheel is what the distribution says of itself;
`licenseConcluded` is what the document found in it. A wheel carrying a component whose
terms the build could not name states that component's `LicenseRef-` alongside the
declaration. Where nothing at all is concluded for a component, the wheel's conclusion
is `NOASSERTION`.

### Matching

A match needs a canonical license text to be almost entirely present *and* to account
for a real share of the file, so a license reproduced in an appendix of a long agreement
is not that agreement's terms. The second threshold is the looser: it bounds padding
around a text, so a short file quoting a license in full still matches. A file carrying
two *distinct* licenses is reported as carrying both; where one is a near-duplicate of
the other, only the match survives. A disjunctive grant is not expressible.

Every grant a component ships is accounted for in its conclusion:

- A text that matches nothing travels as the component's `LicenseRef-`, AND-ed with what
  its other texts are identified as: an MIT `LICENSE` beside an unmatched EULA concludes
  `LicenseRef-<component> AND MIT`.
- A text matching identifiers that share one reference text names the one the others
  extend by a suffix — `MPL-2.0` beside `MPL-2.0-no-copyleft-exception`, `OFL-1.1`
  beside its `-RFN` variants — SPDX's name for the unmodified text. Where there is none,
  as for `GPL-3.0-only` and `GPL-3.0-or-later`, the choice is the licensor's, stated in
  a notice rather than the text, so the text is carried too, and the evidence lists it
  as matching either, not both.
- A text its identifiers account for less than four fifths of, copyright notices aside,
  is carried too, beside the identifiers it names: the rest may be terms no identifier
  names, as in fmt's MIT with its exception paragraph.

The `LicenseRef-` carries exactly the texts carried, and those components are listed
under *License evidence gaps*. So is a component whose only texts are a REUSE
`LICENSES/` pool: it ships them, which meets the obligation, but a pool says what each
file is under, not what the component is, so its license is `NOASSERTION`. So is a
component whose grants name nothing the grammar accepts.

## What stops the Release build

The build stops rather than ship a gap, and its error names the files or components
involved. None of these is fixed by excluding a file or recording an assumed license.
Other errors — an unconfigured build tree, no file API codemodel, an unreadable
archive — are not gaps, and name their own cause.

**A wheel member the build cannot explain** — none of the
[inventory's sources](#how-the-inventory-is-worked-out) accounts for it, including a
shared library the build machine cannot resolve. It reached the wheel by a route the
collector cannot see: teach the collector that route rather than excluding the file.

**Static code on a link line the collector cannot trace** — a name that resolves to
nothing, a fragment that does not parse, a path that is none of the
[kinds above](#static-code-on-the-link-line) (which is also what an unknown option's
value looks like), or an archive inside the build tree that no target produced and no
install record names, which is where ExternalProject leaves its output. Link something
the collector can trace — a target of this build, a fetched dependency, a system package
or a known toolkit — or teach it the route that input took.

**A redistributed component with no license text** — packaging the text is the
obligation. For a fetched component, that is none of the texts
[Where terms are read from](#where-terms-are-read-from) lists; a license file in a
subdirectory covers only that part. For any other component it is the source named above
for its kind: an archive's root license files, a vcpkg port's `share/<port>/copyright`,
a build-machine library's package copyright file or the license file beside it. Obtain
the terms; the component's REUSE info and source headers do not count.

**A declared license with no text shipped** — the wheel's `License-Expression` does not
parse, or names an identifier the pinned list cannot supply a text for, such as a
`LicenseRef-`, and the wheel packages none. A text counts only as `<identifier>.txt`
outside `third-party/`: add it to `license-files` in `pyproject.toml` under that name, or
drop the declaration.

**A checked-in third-party file whose REUSE info fails the per-file gate** — the file's
REUSE info names no license; names one with no current SPDX reference text (a
`LicenseRef-` among them, even beside one that has a text); states an expression that
does not parse; or names an id that is neither SPDX-listed nor a `LicenseRef-` or
`DocumentRef-`. An exception needs only to be listed as an exception. The build names
each such file and why; another file of the same holder does not lend it its terms.

- For real upstream terms, record them as REUSE info naming current SPDX-listed
  identifiers ([License headers](../../docs/source/development/license.rst),
  [REUSE specification](https://reuse.software/spec/)).
- A tag that is not a real declaration — quoted in a string, or mentioned in prose —
  goes between `REUSE-IgnoreStart` and `REUSE-IgnoreEnd`, or is set aside by a
  `.license` sidecar or a `REUSE.toml` annotation with `precedence = "override"`.
