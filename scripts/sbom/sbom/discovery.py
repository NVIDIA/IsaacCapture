# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find out what went into a build by reading what the build left behind.

There is no dependency list to keep in step with the code. A component exists
because a source tree was fetched into the build, and it reaches a wheel because
the link graph or a content hash says so.

Four indexes do the work:

* source trees   ``_deps/*-src`` checkouts -> identity, from git
* build graph    CMake's file API codemodel -> what links and compiles into what
* archives       members of tarballs in the repo, by content hash
* files          repo and build-tree files, by content hash
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tomllib
import tarfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

from packageurl import PackageURL

from . import elf, licensing

# Directories that never hold an input worth indexing.
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar", ".zip")
_MAX_ARCHIVE_BYTES = 1 << 31
# How much of a library nested in an archive is kept, to read its build-id from.
# An SDK ships its libraries under `lib/`, not at the archive root, so without
# this the only members with a build-id are the ones nothing links against.
_ELF_HEAD_BYTES = 1 << 16

_GITHUB_REMOTE = re.compile(
    r"github\.com[:/](?P<org>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_GITCLONE_URL = re.compile(r'clone\b[^\n]*?"(?P<url>[a-z]+[^"\s]*://[^"]+|git@[^"]+)"')
_GITCLONE_TAG = re.compile(r'(?<![-\w])checkout\s+"(?P<tag>[^"]+)"')
_INCLUDE_FLAG = re.compile(r"-(?:I|isystem)\s*(\S+)")
_DEPS_SOURCE = re.compile(r"_deps/(?P<name>[A-Za-z0-9_.+-]+)-src(?:/|$)")


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def component_of(display: str) -> str:
    """The component a `FileIndex` display path belongs to.

    `add_tree` renders `<tree-name>/<relative>` and a fetched checkout's tree is
    `<key>-src`, so this is the inverse of how the index was populated. Inverting
    it at the call site made the convention an undeclared contract.
    """
    return display.split("/", 1)[0].removesuffix("-src")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(root: Path):
    """Every regular file under `root`, never following a symlink."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _SKIP_DIRS:
                    stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


# ==============================================================================
# Source trees
# ==============================================================================


@dataclass(frozen=True)
class Component:
    """Something a wheel can contain, identified from where it came from."""

    key: str
    name: str
    kind: str  # "source-tree" | "archive" | "system-library"
    supplier: str
    homepage: str
    purl: str
    version: str
    download_location: str
    source_info: str
    root: Path | None = None
    license_concluded: str = "NOASSERTION"
    license_declared: str = "NOASSERTION"
    # The component's own copyright notice, where the build can read one. Naming
    # a holder and then declaring none would contradict the same document.
    copyright_text: str = "NOASSERTION"
    evidence: tuple[licensing.LicenseEvidence, ...] = ()


def repo_head(directory: Path) -> str | None:
    """The commit of the checkout rooted at `directory`, if it is one.

    `git -C <dir> rev-parse HEAD` answers for whichever repository encloses the
    directory, so a dependency whose checkout carries no .git of its own would
    otherwise be stamped with this project's commit.
    """
    top = git(directory, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != directory.resolve():
        return None
    return git(directory, "rev-parse", "HEAD")


def git(repo: Path, *args: str) -> str | None:
    """Run a read-only git query, or return None if git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        result.stdout.strip()
        if result.returncode == 0 and result.stdout.strip()
        else None
    )


def notice_from(evidence) -> str:
    """The copyright a component's own licence text states, where it states one.

    Permissive licences generally require the notice to travel with the terms,
    and for MIT and the BSDs it is a line of the same file -- shipped in the
    wheel but, until now, named nowhere a reader could act on.
    """
    # Grants only, though a pool counts as licence evidence everywhere else: a
    # pool is a directory of reference texts, and the notice in one is the
    # licence's own author -- OpenXR ships WTFPL, so reading its pool credited
    # OpenXR to the person who wrote WTFPL.
    notices = [
        found
        for item in evidence
        if item.kind in ("grant", "nested")
        for found in licensing.read_notices(item.text)
        if not licensing.notice_is_the_licence_authors(
            found, [license_id for license_id, _, _ in item.matches]
        )
    ]
    return "\n".join(licensing.fold_notices(notices)) if notices else "NOASSERTION"


def _identity(
    remote: str | None, name: str, commit: str | None
) -> tuple[str, str, str, str]:
    """Supplier, homepage, purl and download location, read off the remote URL."""
    if remote:
        match = _GITHUB_REMOTE.search(remote)
        if match:
            org, repo = match.group("org"), match.group("repo")
            purl = PackageURL(
                type="github", namespace=org, name=repo, version=commit
            ).to_string()
            download = f"git+https://github.com/{org}/{repo}.git" + (
                f"@{commit}" if commit else ""
            )
            return (
                f"Organization: {org}",
                f"https://github.com/{org}/{repo}",
                purl,
                download,
            )
        purl = f"pkg:generic/{name}" + (f"@{commit}" if commit else "")
        return (
            "NOASSERTION",
            remote,
            purl,
            (f"git+{remote}@{commit}" if commit else remote),
        )
    return "NOASSERTION", "NOASSERTION", f"pkg:generic/{name}", "NOASSERTION"


def _declared_constraint(deps_dir: Path, name: str) -> dict[str, str | None]:
    clone = (
        deps_dir
        / f"{name}-subbuild"
        / f"{name}-populate-prefix"
        / "tmp"
        / f"{name}-populate-gitclone.cmake"
    )
    if not clone.is_file():
        return {"url": None, "tag": None}
    text = clone.read_text(encoding="utf-8", errors="replace")
    url = _GITCLONE_URL.search(text)
    tag = _GITCLONE_TAG.search(text)
    return {
        "url": url.group("url") if url else None,
        "tag": tag.group("tag") if tag else None,
    }


def discover_source_trees(deps_dir: Path) -> dict[str, Component]:
    """Every FetchContent checkout under the build's ``_deps``."""
    components: dict[str, Component] = {}
    for root in sorted(deps_dir.glob("*-src")):
        if not root.is_dir():
            continue
        key = root.name.removesuffix("-src")
        declared = _declared_constraint(deps_dir, key)
        commit = repo_head(root)
        remote = git(root, "config", "--get", "remote.origin.url") or declared["url"]
        describe = git(root, "describe", "--tags", "--always")

        supplier, homepage, purl, download = _identity(remote, key, commit)
        evidence = licensing.discover_in_tree(key, root, root.name)
        concluded, declared_expression = licensing.expression(evidence, key)

        details = [f"Fetched into {root.name}."]
        if declared["tag"]:
            details.append(f"Declared constraint: {declared['tag']}.")
        if describe:
            details.append(f"git describe: {describe}.")

        name_match = _GITHUB_REMOTE.search(remote) if remote else None
        components[key] = Component(
            key=key,
            name=name_match.group("repo") if name_match else key,
            kind="source-tree",
            supplier=supplier,
            homepage=homepage,
            purl=purl,
            version=commit or declared["tag"] or "NOASSERTION",
            download_location=download,
            source_info=" ".join(details),
            root=root,
            copyright_text=notice_from(evidence),
            license_concluded=concluded,
            license_declared=declared_expression,
            evidence=tuple(evidence),
        )
    return components


# ==============================================================================
# Build graph
# ==============================================================================


@dataclass(frozen=True)
class Artifact:
    """One file a target produces, and the target that produced it."""

    output: Path
    target: str
    target_id: str


@dataclass(frozen=True)
class _Node:
    """A target as the CMake file API describes it."""

    name: str
    kind: str
    sources: tuple[Path, ...]
    includes: tuple[Path, ...]
    dependencies: tuple[str, ...]


def _owner(path: Path, ownership: Ownership | None) -> str | None:
    """The component a compiled source belongs to, if any."""
    match = _DEPS_SOURCE.search(path.as_posix())
    if match:
        return match.group("name")
    return ownership.owner_of(path) if ownership is not None else None


class SupplierError(Exception):
    """The project does not name exactly one author to record as supplier."""


class ArchiveError(Exception):
    """An archive the build fetched cannot be read."""


class FileApiError(Exception):
    """The build tree carries no CMake file API reply to read."""


class BuildGraph:
    """The link and compile graph, read from CMake's own file API.

    The top-level ``CMakeLists.txt`` asks for the codemodel with
    ``cmake_file_api()``, so any configure of this project leaves one behind,
    under every generator. Scraping ``link.txt`` instead would only work for the
    Makefile generators -- Ninja writes no such file.
    """

    def __init__(self, build_dir: Path, config: str | None = None) -> None:
        self.build_dir = build_dir
        self.artifacts: dict[Path, Artifact] = {}
        self._nodes: dict[str, _Node] = {}
        self._by_name: dict[str, list[Artifact]] = {}
        self._closures: dict[str, tuple[_Node, ...]] = {}
        self._load(config)

    def _load(self, config: str | None) -> None:
        reply = self.build_dir / ".cmake" / "api" / "v1" / "reply"
        models = sorted(reply.glob("codemodel-v2-*.json")) if reply.is_dir() else []
        if not models:
            raise FileApiError(
                f"no CMake file API codemodel under {reply}. Configure the project "
                "with CMake 3.27 or newer, which is where cmake_file_api() asks for it."
            )

        codemodel = json.loads(models[-1].read_text(encoding="utf-8"))
        source_root = Path(codemodel["paths"]["source"])
        build_root = Path(codemodel["paths"]["build"])

        configurations = codemodel["configurations"]
        chosen = next(
            (item for item in configurations if item["name"] == (config or "")),
            configurations[0],
        )

        def resolve(raw: str, root: Path) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else (root / path)

        for entry in chosen["targets"]:
            target = json.loads((reply / entry["jsonFile"]).read_text(encoding="utf-8"))
            includes = {
                resolve(item["path"], source_root).resolve()
                for group in target.get("compileGroups", [])
                for item in group.get("includes", [])
            }
            self._nodes[entry["id"]] = _Node(
                name=target["name"],
                kind=target["type"],
                sources=tuple(
                    resolve(item["path"], source_root).resolve()
                    for item in target.get("sources", [])
                ),
                includes=tuple(sorted(includes)),
                dependencies=tuple(
                    item["id"] for item in target.get("dependencies", [])
                ),
            )
            for item in target.get("artifacts", []):
                output = resolve(item["path"], build_root).resolve()
                artifact = Artifact(
                    output=output, target=target["name"], target_id=entry["id"]
                )
                self.artifacts[output] = artifact
                self._by_name.setdefault(output.name, []).append(artifact)

    def by_name(self, name: str) -> list[Artifact]:
        return self._by_name.get(name, [])

    def repo_inputs(self, repo_root: Path, build_dir: Path) -> set[Path]:
        """Files under `repo_root` this build compiled or included.

        `_deps` is the component-source domain and is attributed from its own
        checkouts; an include directory under the build tree holds generated
        headers, which state no upstream holder.
        """
        found: set[Path] = set()
        for node in self._nodes.values():
            found.update(
                source
                for source in node.sources
                if str(source).startswith(str(repo_root))
                and "/_deps/" not in source.as_posix()
            )
            for include in node.includes:
                text = include.as_posix()
                if not text.startswith(str(repo_root)) or "/_deps/" in text:
                    continue
                if include.is_dir() and not text.startswith(str(build_dir)):
                    found.update(item for item in include.rglob("*") if item.is_file())
        return found

    def _closure(self, target_id: str) -> tuple[_Node, ...]:
        """The target and everything it depends on, transitively.

        Memoised per instance, not with `lru_cache`: a cache on the class would
        hold every graph ever built alive for the life of the process.
        """
        cached = self._closures.get(target_id)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [target_id]
        found: list[_Node] = []
        while stack:
            current = stack.pop()
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            node = self._nodes[current]
            found.append(node)
            stack.extend(node.dependencies)
        self._closures[target_id] = tuple(found)
        return self._closures[target_id]

    def contributions(
        self, artifact: Artifact, ownership: Ownership | None = None
    ) -> dict[str, set[str]]:
        """Which components contributed to an artifact, and in what way.

        Walks the target's dependency closure. Compiled sources name a component
        outright; include paths are the only trace a header-only dependency
        leaves anywhere.
        """
        found: dict[str, set[str]] = {}

        def note(key: str, how: str) -> None:
            found.setdefault(key, set()).add(how)

        for node in self._closure(artifact.target_id):
            for source in node.sources:
                owner = _owner(source, ownership)
                if owner:
                    note(owner, "compiled-source")
            for include in node.includes:
                match = _DEPS_SOURCE.search(include.as_posix())
                if match:
                    note(match.group("name"), "header-include")
                elif ownership is not None:
                    # Which file in the directory was used is unknowable, so
                    # every component with files there is credited, the same
                    # way a fetched include directory is.
                    for owner in ownership.owners_under(include):
                        note(owner, "header-include")
        return found

    def first_party(
        self, artifact: Artifact, repo_root: Path, ownership: Ownership | None = None
    ) -> bool:
        """True when this build compiled the artifact from the project's own sources."""
        for node in self._closure(artifact.target_id):
            if node.name != artifact.target:
                continue
            for source in node.sources:
                if _owner(source, ownership):
                    continue
                try:
                    source.relative_to(repo_root)
                except ValueError:
                    continue
                return True
        return False


# ==============================================================================
# Ownership
# ==============================================================================


class Ownership:
    """Which component a path belongs to.

    One lookup for every kind of third-party code, so recognising a new kind is
    registering a root rather than adding a branch wherever paths are matched.
    """

    def __init__(self) -> None:
        self._files: dict[Path, str] = {}
        self._roots: list[tuple[Path, str]] = []

    def register_root(self, root: Path, key: str) -> None:
        self._roots.append((root.resolve(), key))

    def register_file(self, path: Path, key: str) -> None:
        self._files[path.resolve()] = key

    def owner_of(self, path: Path) -> str | None:
        resolved = path.resolve()
        owner = self._files.get(resolved)
        if owner is not None:
            return owner
        # Longest root wins, so a vendored subdirectory beats its parent.
        best: tuple[int, str] | None = None
        for root, key in self._roots:
            text = str(root)
            if str(resolved) == text or str(resolved).startswith(text + "/"):
                if best is None or len(text) > best[0]:
                    best = (len(text), key)
        return best[1] if best else None

    def owners_under(self, directory: Path) -> set[str]:
        """Components with files inside a directory, for include-path evidence."""
        prefix = str(directory.resolve())
        found = {
            key
            for path, key in self._files.items()
            if str(path) == prefix or str(path).startswith(prefix + "/")
        }
        found.update(
            key
            for root, key in self._roots
            if str(root) == prefix or str(root).startswith(prefix + "/")
        )
        return found


# ==============================================================================
# Third-party sources vendored in this repository
# ==============================================================================
# Not everything third-party arrives through FetchContent. A file checked in
# here that someone else wrote says so in its REUSE header, which is the same
# thing the project states about its own files -- so the copyright holder is
# what separates them, not the directory it happens to sit in.

# REUSE's tag where a file has one, and a plain copyright line where it does
# not -- upstream files usually carry the latter.
_COPYRIGHT_TAG = re.compile(
    r"SPDX-FileCopyrightText:\s*(?P<holder>[^\n\r]+)"
    r"|^[^\w\n]*Copyright\b(?P<plain>[^\n\r]+)",
    re.MULTILINE,
)
_YEARS = re.compile(r"(copyright|\(c\)|©|\d{4}(\s*-\s*\d{4})?|,|\.)", re.IGNORECASE)


def project_authors(repo_root: Path) -> set[str]:
    """Who the project says it is, read from its own packaging metadata.

    As written, not folded: the same names answer "is this holder ours" and
    "who supplied this distribution", and only the first wants them lowercased.
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    return {
        str(author["name"])
        for author in data.get("project", {}).get("authors", [])
        if author.get("name")
    }


def project_supplier(repo_root: Path) -> str:
    """Who supplied this distribution, as its own packaging metadata states it.

    SPDX takes one supplier. Choosing among several, or publishing a release
    that names none, is a question for whoever is publishing it -- so it is
    asked here rather than answered quietly.
    """
    authors = sorted(project_authors(repo_root))
    if len(authors) != 1:
        raise SupplierError(
            f"{repo_root}/pyproject.toml names {len(authors)} project authors "
            f"({', '.join(authors) or 'none'}); SPDX records one supplier for "
            "the distribution"
        )
    return f"Organization: {authors[0]}"


@dataclass(frozen=True)
class _Stated:
    """What a source file says about itself in its header."""

    holder: str
    expression: str | None
    notice: str


def _stated_by(path: Path) -> _Stated | None:
    """Holder, licence and notice, from one read of the file's head."""
    try:
        head = path.open("rb").read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    match = _COPYRIGHT_TAG.search(head)
    if not match:
        return None
    holder = _YEARS.sub(" ", match.group("holder") or match.group("plain") or "")
    holder = " ".join(holder.split()).removesuffix(" All rights reserved").strip()
    if not holder:
        return None
    notice = " ".join(match.group(0).split()).lstrip("/ *#")
    # The tag introduces the notice; it is not part of it.
    notice = re.sub(r"^SPDX-FileCopyrightText:\s*", "", notice)
    return _Stated(holder, licensing.read_spdx_tag(head), notice)


def discover_vendored(
    repo_root: Path, candidates: set[Path], authors: set[str]
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Group checked-in files whose copyright holder is not the project's."""
    grouped: dict[str, list[Path]] = {}
    licences: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    notices: dict[str, set[str]] = {}

    for path in sorted(candidates):
        stated = _stated_by(path)
        if stated is None:
            continue
        if any(author.lower() in stated.holder.lower() for author in authors):
            continue
        key = f"vendored:{licensing.spdx_safe(stated.holder).lower()}"
        grouped.setdefault(key, []).append(path)
        names[key] = stated.holder
        notices.setdefault(key, set()).add(stated.notice)
        if stated.expression:
            licences.setdefault(key, set()).add(stated.expression)

    components: dict[str, Component] = {}
    for key, paths in grouped.items():
        evidence = licensing.canonical_evidence(key, licences.get(key, set()))
        concluded, declared = licensing.expression(evidence, key)
        if licences.get(key):
            declared = licensing.combine(sorted(licences[key]))
        # Say when the list is shortened; without the marker it reads as the
        # whole set, and the first four by name need not be the ones that ship.
        named = sorted(str(item.relative_to(repo_root)) for item in paths)
        relative = ", ".join(named[:4]) + (
            f", and {len(named) - 4} more" if len(named) > 4 else ""
        )
        components[key] = Component(
            key=key,
            name=names[key],
            kind="vendored-source",
            supplier=f"Organization: {names[key]}",
            homepage="NOASSERTION",
            purl=PackageURL(
                type="generic", name=licensing.spdx_safe(names[key]).lower()
            ).to_string(),
            version="NOASSERTION",
            download_location="NOASSERTION",
            source_info=(
                f"Checked into this repository rather than fetched; states its own "
                f"copyright. Files: {relative}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            copyright_text="\n".join(
                licensing.fold_notices(notices.get(key, {names[key]}))
            ),
            evidence=tuple(evidence),
        )
    return components, grouped


# ==============================================================================
# The build host
# ==============================================================================
# The last discovery source: a shared library that no index explains was not
# produced or fetched by this build, so ask the machine that supplied it.

_SEARCH_ROOTS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/local/lib",
)


def _host_library_paths() -> list[Path]:
    """Every shared library the loader can see, plus the CUDA trees it cannot."""
    paths: set[Path] = set()
    if shutil.which("ldconfig"):
        try:
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            for line in result.stdout.splitlines():
                _, _, target = line.partition("=>")
                target = target.strip()
                if target:
                    paths.add(Path(target))
        except (OSError, subprocess.SubprocessError):
            pass
    roots = _library_roots()
    for root in roots:
        paths.update(item for item in root.glob("*.so*") if item.is_file())
    return sorted(paths)


@functools.lru_cache(maxsize=1)
def _host_build_id_index() -> dict[str, str]:
    """Host libraries by GNU build-id.

    A repair tool rewrites SONAME on the way into the wheel, so the name is no
    longer a way back to the machine's copy. The build-id is, and it is the same
    note the linker wrote. Built once; reading it from every library the loader
    knows costs about a second.
    """
    index: dict[str, str] = {}
    for path in _host_library_paths():
        build_id = elf.read_build_id(path)
        if not build_id:
            continue
        # A regular file beats a symlink to it: dpkg-query knows the real path.
        current = index.get(build_id)
        if current is None or (
            path.is_symlink() is False and Path(current).is_symlink()
        ):
            index[build_id] = str(path)
    return index


def resolve_system_library_by_build_id(build_id: str) -> dict:
    """Trace a vendored library back by build-id, whatever it was renamed to."""
    origin: dict = {
        "soname": None,
        "build_id": build_id,
        "resolved_path": None,
        "package": None,
        "copyright": None,
    }
    found = _host_build_id_index().get(build_id)
    if found is None:
        return origin
    library = Path(found).resolve()
    origin["soname"] = library.name
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def resolve_system_library(soname: str) -> dict:
    """Trace an auditwheel-vendored library back to the package that supplied it."""
    origin: dict = {
        "soname": soname,
        "resolved_path": None,
        "package": None,
        "copyright": None,
    }
    roots = _library_roots()

    library = next((root / soname for root in roots if (root / soname).exists()), None)
    if library is None:
        return origin
    library = library.resolve()
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def _identify_host_package(library: Path, origin: dict) -> dict:
    """Name the distro package that owns a resolved host library."""
    if shutil.which("dpkg-query") is None:
        return origin
    try:
        result = subprocess.run(
            ["dpkg-query", "-S", str(library)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return origin
    if result.returncode != 0 or ":" not in result.stdout:
        return origin

    package = result.stdout.split(":", 1)[0].strip()
    origin["package"] = package
    origin["supplier"] = _package_supplier(package)
    origin["version"] = _package_field(package, "Version")
    origin["distro"] = _os_release_id()
    copyright_file = Path("/usr/share/doc") / package / "copyright"
    if copyright_file.is_file():
        origin["copyright"] = str(copyright_file)
    return origin


def _package_field(package: str, field: str) -> str | None:
    """One field of a distro package's metadata, as dpkg records it."""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", f"-f=${{{field}}}", package],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _os_release_id() -> str | None:
    """The distribution this package manager belongs to, for the purl type."""
    try:
        return platform.freedesktop_os_release().get("ID") or None
    except OSError:
        return None


def _package_supplier(package: str) -> str | None:
    """Who ships a distro package, in the form SPDX wants.

    The package name is not a supplier: `libbsd0` names the thing, not whoever
    provided it. dpkg records a maintainer, which is the answer.
    """
    maintainer = _package_field(package, "Maintainer")
    if not maintainer:
        return None
    # "Name <email>" is Debian's form; SPDX wants "Organization: Name (email)".
    match = re.match(r"^(?P<name>.+?)\s*<(?P<email>[^>]+)>$", maintainer)
    if match:
        return f"Organization: {match['name']} ({match['email']})"
    return f"Organization: {maintainer}"


def _library_roots() -> list[Path]:
    """Where a pre-built library the build machine supplied can be found."""
    roots = [Path(item) for item in _SEARCH_ROOTS if Path(item).is_dir()]
    return roots + [Path(item) for item in sorted(glob.glob("/usr/local/cuda*/lib64"))]


def _license_near(library: Path) -> Path | None:
    """A toolchain that installs outside the package manager still ships its terms.

    CUDA is the case that matters: the runtime lands in /usr/local/cuda*/lib64 with
    EULA.txt a level or two up, and dpkg knows nothing about it.
    """
    current = library.parent
    for _ in range(3):
        for entry in sorted(current.iterdir()) if current.is_dir() else []:
            if entry.is_file() and licensing.classify(entry.name):
                return entry
        if current.parent == current:
            break
        current = current.parent
    return None


def system_component(soname: str, origin: dict) -> Component:
    """A pre-built library the build machine supplied, named by what it came from."""
    evidence: list[licensing.LicenseEvidence] = []
    key = f"system:{soname}"
    license_path = origin.get("copyright")
    if not license_path and origin.get("resolved_path"):
        found = _license_near(Path(origin["resolved_path"]))
        license_path = str(found) if found else None
        origin["license_file"] = license_path
    if license_path and Path(license_path).is_file():
        text = Path(license_path).read_text(encoding="utf-8", errors="replace")
        evidence.append(
            licensing.evidence_for(key, "build-host", license_path, text, "grant")
        )
    # `key`, not `soname`: the key is what identifies this component everywhere
    # else, and two components reduced to the same label would mint one LicenseRef
    # for two different texts.
    concluded, declared = licensing.expression(evidence, key)

    package = origin.get("package")
    details = [
        f"Vendored into the wheel by auditwheel from {origin.get('resolved_path') or soname}."
    ]
    if package:
        details.append(f"Supplied by system package {package}.")
    return Component(
        key=key,
        name=soname,
        kind="system-library",
        supplier=origin.get("supplier") or "NOASSERTION",
        # The distro copyright file is where the holder is stated, and it was
        # read, packaged and put on the file record -- then left off the package
        # and out of the notices every other component appears in.
        copyright_text=notice_from(evidence),
        homepage="NOASSERTION",
        # A distro package the machine can name resolves; pkg:generic does not.
        purl=(
            PackageURL(
                type="deb",
                namespace=origin.get("distro"),
                name=origin["package"],
                version=origin.get("version"),
            ).to_string()
            if origin.get("package")
            else PackageURL(type="generic", name=soname).to_string()
        ),
        version=origin.get("version") or "NOASSERTION",
        download_location="NOASSERTION",
        source_info=" ".join(details),
        license_concluded=concluded,
        license_declared=declared,
        evidence=tuple(evidence),
    )


# ==============================================================================
# Content indexes
# ==============================================================================


@dataclass(frozen=True)
class Member:
    """A file inside an archive, or a file on disk."""

    container: str
    path: str
    sha256: str
    size: int
    build_id: str | None = None


class ArchiveIndex:
    """Members of every archive in the repo, so extracted bytes can be traced back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, Member] = {}
        self.all_by_hash: dict[str, list[Member]] = {}
        self.by_name: dict[str, list[Member]] = {}
        # Survives the RPATH/SONAME rewrite a repair tool applies on the way in,
        # so a patched wheel member still points at the archive it came from.
        self.all_by_build_id: dict[str, list[Member]] = {}
        self.archives: dict[str, dict] = {}

    def add_archive(self, path: Path, display: str) -> None:
        size = path.stat().st_size
        if size > _MAX_ARCHIVE_BYTES:
            return
        members = self._read_members(path, display)
        self.archives[display] = {
            "path": display,
            "sha256": sha256_file(path),
            "size": size,
            "member_count": len(members),
        }
        for member, data in members:
            # `data` is the whole member only at the archive root; elsewhere it
            # is the head kept for exactly this, so ask only for the build-id.
            build_id = elf.build_id_of(data)
            if build_id:
                member = replace(member, build_id=build_id)
                self.all_by_build_id.setdefault(build_id, []).append(member)
            self.by_hash.setdefault(member.sha256, member)
            self.all_by_hash.setdefault(member.sha256, []).append(member)
            self.by_name.setdefault(Path(member.path).name, []).append(member)
            self._maybe_license(display, member, data)

    def _maybe_license(self, display: str, member: Member, data: bytes) -> None:
        name = Path(member.path).name
        if Path(member.path).parent.as_posix() not in {".", ""}:
            return  # only the archive root states the bundle's own terms
        if name.upper() == "VERSION" and len(data) < 256:
            # First line only. These files carry more than one, and a newline in
            # a version breaks every tag-value and table rendering downstream.
            text = data.decode("utf-8", "replace").strip().splitlines()
            self.archives[display]["version_text"] = text[0].strip() if text else ""
            return
        if not licensing.classify(name):
            return
        self.archives[display].setdefault("license_members", []).append((member, data))

    @staticmethod
    def _read_members(path: Path, display: str) -> list[tuple[Member, bytes]]:
        """Hash every member; keep the bytes only of the few worth keeping.

        An SDK tarball expands to far more than it compresses to, so holding
        every member would size peak memory by the largest archive rather than
        by what is retained.
        """
        collected: list[tuple[Member, bytes]] = []

        def take(name: str, handle) -> None:
            # A root member's bytes are kept whole, for its licence text; every
            # other member streams through the digest, and keeps a head only if
            # it is a library, for its build-id. Reading every member whole
            # sizes peak memory by the largest file in the archive.
            at_root = Path(name).parent.as_posix() in {".", ""}
            digest = hashlib.sha256()
            size = 0
            kept = bytearray()
            head = b""
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                if at_root:
                    kept += chunk
                elif not size and elf.is_elf(chunk):
                    head = chunk[:_ELF_HEAD_BYTES]
                size += len(chunk)
            collected.append(
                (
                    Member(display, name, digest.hexdigest(), size),
                    bytes(kept) if at_root else head,
                )
            )

        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if not info.is_dir():
                            with archive.open(info) as handle:
                                take(info.filename, handle)
            else:
                with tarfile.open(path) as archive:
                    for info in archive:
                        if not info.isfile():
                            continue
                        handle = archive.extractfile(info)
                        if handle is not None:
                            take(info.name.removeprefix("./"), handle)
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
            # Only archives this build deliberately fetched are indexed, so one
            # that will not open is a broken download, not a stray file. Saying
            # so beats reporting every file it should have explained as a member
            # that reached the wheel by an unknown route.
            raise ArchiveError(f"{display} cannot be read: {error}") from error
        return collected


def tracked_files(repo_root: Path) -> list[str]:
    """What this repository actually holds, as git records it.

    A working tree also holds whatever was built in it. An install prefix has no
    marker saying so, so walking the directory cannot tell the two apart, and a
    wheel member matching a *build output* would be reported as a copy of
    repository source. Git knows the difference.
    """
    listing = git(repo_root, "ls-files", "-z", "--cached")
    if listing is None:
        raise FileApiError(
            f"{repo_root} is not a git checkout this build can query; the SBOM "
            "distinguishes repository source from build output by what git tracks"
        )
    return [item for item in listing.split("\0") if item]


def discover_archives(
    repo_root: Path, held: frozenset[str], skip: tuple[Path, ...] = ()
) -> ArchiveIndex:
    """Index archives held in the repository, not ones a build unpacked.

    `held` is what the repository tracks. It has no default: an empty set
    silently indexes nothing, and a caller that forgot it would see every
    archive-derived wheel member become unexplainable rather than wrong.
    """
    index = ArchiveIndex()
    skipped = tuple(item.resolve() for item in skip)
    for path in _walk(repo_root):
        # By path segment, not by string prefix: `dist` prefixes `distribution`
        # and a build directory named `build` prefixes `buildtools`, and an
        # archive skipped that way goes on to explain nothing.
        if any(path.is_relative_to(item) for item in skipped):
            continue
        name = path.name.lower()
        if not name.endswith(ARCHIVE_SUFFIXES):
            continue
        relative = path.relative_to(repo_root).as_posix()
        # Held here, or fetched by this build and recorded as such. An archive
        # anywhere else is output -- staged into an install prefix, say -- and
        # letting it explain a wheel member is the build explaining itself.
        if relative not in held and not path.with_name(path.name + ".source").is_file():
            continue
        index.add_archive(path, relative)
    return index


class FileIndex:
    """Files on disk by content hash, for tracing a copied wheel member back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, str] = {}
        # Every path sharing a digest, because one is not evidence of origin when
        # several files hold the same bytes -- a stock licence text, or a module
        # that is nothing but an SPDX header.
        self.all_by_hash: dict[str, list[str]] = {}
        self.by_suffix: dict[str, list[str]] = {}
        self._digests: dict[str, str] = {}
        self._by_display: dict[str, str] = {}

    def digest_of(self, path: Path) -> str | None:
        """The digest this index already computed for a path, if it indexed it."""
        return self._digests.get(str(path.resolve()))

    def add_tree(
        self, root: Path, display_root: str, skip: tuple[Path, ...] = ()
    ) -> None:
        # Compare skip entries as paths relative to `root`: `--build-dir` may be
        # relative, and `_deps/foo-srcx` string-prefixes `_deps/foo-src`.
        inside = []
        for item in skip:
            try:
                inside.append(item.resolve().relative_to(root.resolve()))
            except ValueError:
                continue
        self._add(
            root,
            display_root,
            (
                path
                for path in _walk(root)
                if not any(
                    path.relative_to(root).is_relative_to(item) for item in inside
                )
            ),
        )

    def add_paths(self, root: Path, display_root: str, paths) -> None:
        """Index an explicit list rather than whatever the tree happens to hold."""
        self._add(root, display_root, (root / item for item in paths))

    def _add(self, root: Path, display_root: str, paths) -> None:
        # One resolve for the tree, not one per file: `_walk` never crosses a
        # symlink, so the resolved root plus the relative path is the real path.
        resolved_root = root.resolve()
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            display = f"{display_root}/{relative}"
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            self._digests[str(resolved_root / relative)] = digest
            self.by_hash.setdefault(digest, display)
            self.all_by_hash.setdefault(digest, []).append(display)
            self._by_display[display] = digest
            # Two trailing segments is enough to disambiguate an __init__.py.
            self.by_suffix.setdefault(
                "/".join(relative.rsplit("/", 2)[-2:]), []
            ).append(display)

    def digest_of_display(self, display: str) -> str | None:
        """The digest this index recorded for one of its own display paths."""
        return self._by_display.get(display)

    def under(self, display_prefix: str):
        """Every (display, digest) this index holds beneath a display path."""
        prefix = display_prefix.rstrip("/") + "/"
        return (
            (display, digest)
            for display, digest in self._by_display.items()
            if display.startswith(prefix)
        )

    def path_suffix_match(self, wheel_path: str) -> str | None:
        key = "/".join(wheel_path.rsplit("/", 2)[-2:])
        matches = self.by_suffix.get(key, [])
        return matches[0] if len(matches) == 1 else None
