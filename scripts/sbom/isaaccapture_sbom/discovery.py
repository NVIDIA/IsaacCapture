# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover components and provenance from build outputs.

Indexes cover fetched trees, the CMake graph, archive members, and repository
and generated files. See scripts/sbom/README.md for attribution rules.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tomllib
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from urllib.parse import urlparse

from packageurl import PackageURL

from . import elf, licensing

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
# Retain ELF prefixes even for nested archive members to read their build IDs.
_ELF_HEAD_BYTES = 1 << 16

_GITHUB_REMOTE = re.compile(
    r"github\.com[:/](?P<org>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_GITHUB_REPO = r"(?P<org>[^/]+)/(?P<repo>[^/]+)/"
_GITHUB_DOWNLOADS = (
    re.compile(
        rf"^https?://github\.com/{_GITHUB_REPO}releases/download/(?P<ref>[^/]+)/[^/]+$"
    ),
    re.compile(
        rf"^https?://github\.com/{_GITHUB_REPO}archive/(?:refs/(?:tags|heads)/)?"
        rf"(?P<ref>.+?)(?:{'|'.join(map(re.escape, ARCHIVE_SUFFIXES))})$"
    ),
    re.compile(
        rf"^https?://codeload\.github\.com/{_GITHUB_REPO}(?:legacy\.)?(?:tar\.gz|zip)/"
        r"(?:refs/(?:tags|heads)/)?(?P<ref>.+)$"
    ),
)
# Local download paths identify only the builder, not a public origin.
_DOWNLOAD_SCHEMES = ("http", "https", "ftp", "ftps")
_GIT_TRANSPORTS = ("http", "https", "ssh", "git")
# git's scp-like `[user@]host:path`, which SPDX knows only as `ssh://`.
_SCP_REMOTE = re.compile(r"^(?P<user>[^@/:]+@)?(?P<host>[^@/:]{2,}):(?!//)/?")
_VERSION_IN_NAME = re.compile(r"(\d+\.\d+[\w.+-]*?)(?=-|\.tar|\.tgz|\.zip|$)")
# The ref ExternalProject's clone script checks out; `clone --no-checkout` is not one.
_GITCLONE_TAG = re.compile(r'(?<![-\w])checkout\s+"(?P<tag>[^"]+)"')
_DEPS_SOURCE = re.compile(r"_deps/(?P<name>[A-Za-z0-9_.+-]+)-src(?:/|$)")


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_LOG = logging.getLogger("isaaccapture.sbom.discovery")


def component_of(display: str) -> str:
    """Recover the component key from FileIndex's <key>-src/<relative> display path."""
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


# Source trees


@dataclass(frozen=True)
class Component:
    """Something a wheel can contain, identified from where it came from."""

    key: str
    name: str
    # "source-tree" | "archive" | "system-library" | "vendored-source" | "vcpkg-port"
    kind: str
    supplier: str
    homepage: str
    purl: str
    version: str
    download_location: str
    source_info: str
    root: Path | None = None
    license_concluded: str = "NOASSERTION"
    license_declared: str = "NOASSERTION"
    copyright_text: str = "NOASSERTION"
    evidence: tuple[licensing.LicenseEvidence, ...] = ()
    # (path, reason) for per-file REUSE failures; a holder's other files cannot
    # lend their licenses to an unlicensed file.
    unlicensed: tuple[tuple[str, str], ...] = ()


def own_git(directory: Path, *args: str) -> str | None:
    """Query git only when directory is the checkout root.

    An archive extracted beneath the project must not inherit the enclosing
    repository's identity.
    """
    if not _is_own_checkout(directory.resolve()):
        return None
    return git(directory, *args)


@functools.cache
def _is_own_checkout(resolved: Path) -> bool:
    """Cached per resolved path: one collector run does not create checkouts."""
    top = git(resolved, "rev-parse", "--show-toplevel")
    return top is not None and Path(top).resolve() == resolved


def git(repo: Path, *args: str) -> str | None:
    """Run a read-only git query, returning None on failure.

    Clear repository-local variables (including GIT_DIR and GIT_CONFIG) that
    override -C. Preserve GIT_CONFIG_PARAMETERS/COUNT for settings such as
    safe.directory, following git's submodule behavior.
    """
    local = set(_git_local_variables())
    env = {name: value for name, value in os.environ.items() if name not in local}
    return _ask("git", "-C", str(repo), *args, env=env).strip() or None


@functools.cache
def _git_local_variables() -> tuple[str, ...]:
    listed = _ask("git", "rev-parse", "--local-env-vars").split()
    kept = {"GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT"}
    return tuple(name for name in listed if name not in kept)


def version_in_name(name: str) -> str | None:
    """The version an archive's file name states, as in `SDL2-2.32.10.tar.gz`."""
    match = _VERSION_IN_NAME.search(name)
    return match.group(1) if match else None


def notice_from(evidence) -> str:
    """Read copyright notices from the component's license grants."""
    # Pool texts can name the license author (for example WTFPL's), not the component.
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


class _Identity(NamedTuple):
    name: str
    supplier: str
    homepage: str
    purl: str
    download_location: str
    version: str


def _github_identity(org: str, repo: str, ref: str | None) -> tuple[str, str, str]:
    purl = PackageURL(type="github", namespace=org, name=repo, version=ref)
    return f"Organization: {org}", f"https://github.com/{org}/{repo}", purl.to_string()


def _identity(
    remote: str | None, name: str, commit: str | None, tag: str | None = None
) -> _Identity:
    """Identify a git source by remote URL and commit, falling back to its declared tag."""
    ref = commit or tag
    version = ref or "NOASSERTION"
    match = _GITHUB_REMOTE.search(remote) if remote else None
    if match:
        org, repo = match.group("org"), match.group("repo")
        download = f"git+https://github.com/{org}/{repo}.git" + (
            f"@{ref}" if ref else ""
        )
        return _Identity(repo, *_github_identity(org, repo, ref), download, version)
    if remote:
        purl = PackageURL(type="generic", name=name, version=ref).to_string()
        location = _SCP_REMOTE.sub(r"ssh://\g<user>\g<host>/", remote)
        if urlparse(location).scheme.lower() not in _GIT_TRANSPORTS:
            download = "NOASSERTION"
        else:
            download = f"git+{location}@{ref}" if ref else location
        return _Identity(name, "NOASSERTION", remote, purl, download, version)
    return _Identity(
        name,
        "NOASSERTION",
        "NOASSERTION",
        f"pkg:generic/{name}",
        "NOASSERTION",
        version,
    )


def _archive_identity(url: str, digest: str | None, name: str) -> tuple[_Identity, str]:
    """Identify an archive from its declared URL and record its sourceInfo."""
    parsed = urlparse(url)
    file_name = PurePosixPath(parsed.path).name
    downloadable = parsed.scheme.lower() in _DOWNLOAD_SCHEMES
    download = url if downloadable else "NOASSERTION"
    detail = f"Declared archive: {url if downloadable else f'local file {file_name}'}"
    detail += f" ({digest})." if digest else "."

    match = next(filter(None, (form.match(url) for form in _GITHUB_DOWNLOADS)), None)
    if match:
        org, repo, ref = match.group("org", "repo", "ref")
        github = _github_identity(org, repo, ref)
        return _Identity(repo, *github, download, ref), detail

    qualifiers = {"download_url": url} if downloadable else {}
    if digest and "=" in digest:
        algorithm, value = digest.split("=", 1)
        algorithm = algorithm.lower().replace("_", "-")
        qualifiers["checksum"] = f"{algorithm}:{value.lower()}"
    version = version_in_name(file_name)
    purl = PackageURL(
        type="generic", name=name, version=version, qualifiers=qualifiers or None
    )
    identity = _Identity(
        name,
        "NOASSERTION",
        "NOASSERTION",
        purl.to_string(),
        download,
        version or "NOASSERTION",
    )
    return identity, detail


@dataclass(frozen=True)
class _Declaration:
    """What a FetchContent declaration asked for."""

    repository: str | None = None
    tag: str | None = None
    url: str | None = None
    hash: str | None = None


def _stamps(deps_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Index FetchContent stamps by tree and download method."""
    # Under the populate sub-build, or, with CMP0168 NEW, the build's own CMakeFiles.
    roots = (
        (deps_dir, "*-subbuild/*-populate-prefix/src/*-populate-stamp"),
        (deps_dir.parent / "CMakeFiles" / "fc-stamp", "*"),
    )
    newest: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    for base, pattern in roots:
        for method in ("url", "git"):
            for stamp in base.glob(f"{pattern}/*-{method}info.txt"):
                fields = {}
                text = stamp.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    key, separator, value = line.partition("=")
                    if separator and not line.startswith("#"):
                        fields[key.strip()] = value.strip()
                if not fields.get("source_dir"):
                    continue
                # Both layouts can hold one after a policy change; the newer wins.
                slot = (Path(fields["source_dir"]).name, method)
                written = stamp.stat().st_mtime_ns
                if slot not in newest or written > newest[slot][0]:
                    newest[slot] = (written, fields)
    by_tree: dict[str, dict[str, dict[str, str]]] = {}
    for (tree, method), (_, fields) in newest.items():
        by_tree.setdefault(tree, {})[method] = fields
    return by_tree


def _declaration(deps_dir: Path, fields: dict[str, str]) -> _Declaration:
    tag = None
    # The git stamp's command runs the clone script, which names the ref; the
    # stamp's paths are absolute, so a moved build tree is rebased onto this one.
    built_in = Path(fields["source_dir"]).parent.parent
    for word in fields.get("command", "").split(";"):
        if not word.endswith("-gitclone.cmake"):
            continue
        script = Path(word)
        if not script.is_file() and script.is_relative_to(built_in):
            script = deps_dir.parent / script.relative_to(built_in)
        if script.is_file():
            text = script.read_text(encoding="utf-8", errors="replace")
            match = _GITCLONE_TAG.search(text)
            tag = match.group("tag") if match else None
    return _Declaration(
        repository=fields.get("repository") or None,
        tag=tag,
        # A CMake list of mirrors, tried in order.
        url=fields.get("url(s)", "").split(";")[0] or None,
        hash=fields.get("hash") or None,
    )


def _public_url(url: str) -> str:
    """`url` without credentials in its authority, and without the query and
    fragment, where a signed download carries its token."""
    parsed = urlparse(url)
    if "@" not in parsed.netloc and not parsed.query and not parsed.fragment:
        return url
    netloc = parsed.netloc.rpartition("@")[2]
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def discover_source_trees(deps_dir: Path) -> dict[str, Component]:
    """Every FetchContent tree under the build's ``_deps``."""
    components: dict[str, Component] = {}
    stamps = _stamps(deps_dir)
    for root in sorted(deps_dir.glob("*-src")):
        if not root.is_dir():
            continue
        key = root.name.removesuffix("-src")
        # A stamp outlives a switch of method, and an unchanged one keeps its old
        # mtime, so the tree decides: a git clone is a checkout of its own.
        methods = stamps.get(root.name, {})
        method = "git" if _is_own_checkout(root.resolve()) else "url"
        fields = methods.get(method) or next(iter(methods.values()), None)
        declared = _declaration(deps_dir, fields) if fields else _Declaration()
        commit = own_git(root, "rev-parse", "HEAD")
        remote = (
            own_git(root, "config", "--get", "remote.origin.url") or declared.repository
        )
        remote = _public_url(remote) if remote else None
        describe = own_git(root, "describe", "--tags", "--always")

        details = [f"Fetched into {root.name}."]
        if remote or not declared.url:
            identity = _identity(remote, key, commit, declared.tag)
            if declared.tag:
                details.append(f"Declared constraint: {declared.tag}.")
        else:
            identity, detail = _archive_identity(
                _public_url(declared.url), declared.hash, key
            )
            details.append(detail)
        if describe:
            details.append(f"git describe: {describe}.")

        evidence = licensing.discover_in_tree(key, root, root.name)
        concluded, declared_expression = licensing.expression(evidence, key)
        components[key] = Component(
            key=key,
            name=identity.name,
            kind="source-tree",
            supplier=identity.supplier,
            homepage=identity.homepage,
            purl=identity.purl,
            version=identity.version,
            download_location=identity.download_location,
            source_info=" ".join(details),
            root=root,
            copyright_text=notice_from(evidence),
            license_concluded=concluded,
            license_declared=declared_expression,
            evidence=tuple(evidence),
        )
    return components


# Build graph


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
    # What its link line names. Only linkable targets have one; a static
    # library's own requirements land on its consumers' lines.
    linked: tuple[Path, ...] = ()  # outputs of targets in this graph
    inputs: tuple[LinkInput, ...] = ()  # everything else that can carry code
    # Sources with compile groups; merely listed sources are not necessarily compiled.
    compiled: tuple[Path, ...] = ()
    objects: Path | None = None


# Evidence strength for nested-license reach. Missing compiler records fall
# back to listed sources and include directories. Prebuilt archives retain
# the whole component's terms because their source contents are unknown.
REACH = (
    "copied-file",
    "derived-file",
    "compiled-source",
    "header-read",
    "listed-source",
    "include-dir",
)
PREBUILT = "prebuilt-library"


class Reach(NamedTuple):
    """Evidence retaining a nested license: method, source path, and wheel member."""

    how: str
    path: str
    member: str


def reach_rank(how: str) -> int:
    """Strongest first; a linked archive ranks below every path."""
    return REACH.index(how) if how in REACH else len(REACH)


def note_reach(found: dict, path, how: str) -> None:
    """Record `how` for `path` unless stronger evidence is already there."""
    current = found.get(path)
    if current is None or reach_rank(how) < reach_rank(current):
        found[path] = how


def covers(path: str, how: str, scope: str) -> bool:
    """Whether a reached path puts something under a nested text's scope in
    the wheel. An include directory may have been read anywhere beneath it,
    so one above the scope keeps it too."""
    reached, scope_path = PurePosixPath(path), PurePosixPath(scope)
    return reached.is_relative_to(scope_path) or (
        how == "include-dir" and scope_path.is_relative_to(reached)
    )


@dataclass(frozen=True)
class LinkInput:
    """An archive or object a link line names, or a name it could not resolve."""

    named: str  # as the link line spells it
    path: Path | None
    kind: str  # "archive" | "object" | "unresolved"
    reason: str = ""
    members: int | None = None  # an archive's object files

    @property
    def key(self) -> str:
        return str(self.path) if self.path else self.named


@dataclass
class _Line:
    """One target's link line while it is read."""

    link_dir: Path
    directories: list[Path]
    sysroot: str = ""
    linked: list[Path] = field(default_factory=list)
    inputs: list[LinkInput] = field(default_factory=list)


@dataclass(frozen=True)
class _Search:
    """Implicit linker search directories and the sysroot used for leading = paths."""

    compiler_dirs: list[Path]
    compiler: str | None
    sysroot: str

    @property
    def directories(self) -> list[Path]:
        linker = (
            _linker_search_dirs(self.compiler, self.sysroot) if self.compiler else []
        )
        return self.compiler_dirs + linker


# Driver and linker options whose value is the next word. Skipping the value
# keeps a version script or an rpath from being read as an input.
_DRIVER_TAKES_VALUE = frozenset(
    """
    -o -x -T -u -e -z -I -B -isystem -include -imacros -MF -MT -MQ
    -Xassembler -Xpreprocessor --param
    """.split()
)
_LINKER_TAKES_VALUE = frozenset(
    """
    -rpath --rpath -rpath-link --rpath-link --version-script -soname --soname -h
    -z -T --script -Map --Map --dynamic-list -e --entry -u --undefined --defsym
    -o --output --sysroot -y --trace-symbol --wrap -m -R --just-symbols
    -dynamic-linker --dynamic-linker -I -F -f --filter --auxiliary --exclude-libs
    -plugin --plugin -plugin-opt --plugin-opt --section-start -Ttext -Tdata -Tbss
    --retain-symbols-file -Y -G --gpsize -a -A --architecture -b --format -assert
    -init -fini -c --mri-script -P --depaudit --audit --require-defined
    --export-dynamic-symbol --export-dynamic-symbol-list --dependency-file
    --sort-section --section-ordering-file
    """.split()
)
_STATIC_MODE = frozenset({"-Bstatic", "-dn", "-non_shared", "-static"})
_DYNAMIC_MODE = frozenset({"-Bdynamic", "-dy", "-call_shared"})
# GNU ld script commands that name inputs; AS_NEEDED nests inside them.
_SCRIPT_INPUTS = re.compile(r"\b(?:INPUT|GROUP)\s*\(")
_SCRIPT_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SCRIPT_DEPTH = 8
_NOT_FOUND = (
    "not found; if it is the value of a linker option, that option is one the "
    "collector does not know takes a value"
)
_UNKNOWN_KIND = "not an archive, an object, a shared library or a linker script"
_UNSEARCHED = (
    "found in no -L directory, compiler link directory or linker search directory"
)
_NO_COMPILER = (
    "found in no -L directory, and the build records no compiler to ask for its "
    "search directories; reconfigure so the file API writes its toolchains reply"
)


def _linker_words(fragments: list[dict]) -> tuple[list[str], list[str]]:
    """Unpack linker words and retain unparseable fragments.

    Read every CMake role: paths passed as link options appear under flags.
    Unpack -Wl, and -Xlinker, and apply the driver's -static to the whole line.
    """
    stream: list[str] = []
    unparsed: list[str] = []
    static = False
    for fragment in fragments:
        try:
            words = iter(shlex.split(fragment["fragment"]))
        except ValueError:
            unparsed.append(fragment["fragment"])
            continue
        if fragment.get("role") == "libraryPath":
            stream.extend(f"-L{word.removeprefix('-L')}" for word in words)
            continue
        for word in words:
            if word.startswith("-Wl,"):
                stream.extend(word[4:].split(","))
            elif word == "-Xlinker":
                stream.append(next(words, ""))
            elif word in _DRIVER_TAKES_VALUE:
                next(words, None)
            elif word == "-static":
                static = True
            elif word.startswith(("-l", "-L")) or not word.startswith("-"):
                stream.append(word)
    return (["-Bstatic"] if static else []) + _joined(stream), unparsed


def _joined(stream: list[str]) -> list[str]:
    """Each search option as one word: `-l x`, `--library x` and `--library=x`
    all become `-lx`, and the same for `-L`."""
    words: list[str] = []
    pending = iter(word for word in stream if word)
    for word in pending:
        option, equals, value = word.partition("=")
        if word in ("-L", "--library-path", "-l", "--library"):
            option, value = word, next(pending, "")
        elif not (equals and option in ("--library-path", "--library")):
            words.append(word)
            continue
        words.append(("-L" if option in ("-L", "--library-path") else "-l") + value)
    return words


def _search_library(name: str, directories: list[Path], static: bool) -> Path | None:
    """The file ld takes for `-l<name>`: `.so` before `.a` in each directory."""
    if name.startswith(":"):
        candidates = [name[1:]]
    else:
        candidates = ([] if static else [f"lib{name}.so"]) + [f"lib{name}.a"]
    for directory in directories:
        for candidate in candidates:
            if (directory / candidate).is_file():
                return directory / candidate
    return None


def _sysrooted(text: str, sysroot: str, base: Path) -> Path:
    """A search directory as ld reads it: a leading `=` stands for the sysroot."""
    if text.startswith("="):
        return Path(sysroot + text[1:]).resolve()
    return (base / text).resolve()


def _inspect(path: Path) -> tuple[str | None, int | None, list[str] | None]:
    """(kind, archive members, script entries), judged by content.

    The kind is "archive", "object", "shared", "script", "missing", or None for
    a file that is none of them.
    """
    if not path.is_file():
        return "missing", None, None
    members = archive_members(path)
    if members is not None:
        return "archive", members, None
    kind = {"ET_REL": "object", "ET_DYN": "shared"}.get(elf.file_type(path) or "")
    if kind:
        return kind, None, None
    entries = _script_inputs(path)
    return ("script" if entries is not None else None), None, entries


def _script_inputs(path: Path) -> list[str] | None:
    """Return INPUT/GROUP paths and -l names, or None for a non-linker-script file."""
    try:
        with path.open("rb") as handle:
            head = handle.read(1 << 16)
    except OSError:
        return None
    if b"\0" in head:
        return None
    body = _SCRIPT_COMMENT.sub(" ", head.decode("utf-8", "replace"))
    commands = list(_SCRIPT_INPUTS.finditer(body))
    if not commands:
        return None
    found: list[str] = []
    for match in commands:
        depth, end = 1, match.end()
        while end < len(body) and depth:
            depth += {"(": 1, ")": -1}.get(body[end], 0)
            end += 1
        found += [
            item
            for item in re.split(r"[\s,()]+", body[match.end() : end - 1])
            if item and item != "AS_NEEDED"
        ]
    return found


def owned_input(item: LinkInput, ownership: Ownership | None) -> str | None:
    """The component a linked archive or object belongs to by the ownership
    index -- a fetched tree, a vcpkg port, vendored material -- if it holds code."""
    if ownership is None or item.path is None or item.kind == "unresolved":
        return None
    if item.members == 0:
        return None
    return ownership.owner_of(item.path)


def _owner(path: Path, ownership: Ownership | None) -> str | None:
    """The component a compiled source belongs to, if any."""
    match = _DEPS_SOURCE.search(path.as_posix())
    if match:
        return match.group("name")
    return ownership.owner_of(path) if ownership is not None else None


def _include_owners(include: Path, ownership: Ownership | None) -> set[str]:
    """Credit every owner under an include directory when individual headers are unknown."""
    match = _DEPS_SOURCE.search(include.as_posix())
    if match:
        return {match.group("name")}
    return ownership.owners_under(include) if ownership is not None else set()


class SupplierError(Exception):
    """The project does not name exactly one author to record as supplier."""


class ArchiveError(Exception):
    """An archive the build fetched cannot be read."""


class FileApiError(Exception):
    """The build tree carries no CMake file API reply to read."""


def _within(repo_root: Path, build_dir: Path):
    """Test checkout membership outside the build tree; both roots must be resolved."""

    def check(path: Path) -> bool:
        resolved = path if path.is_absolute() else path.resolve()
        return resolved.is_relative_to(repo_root) and not resolved.is_relative_to(
            build_dir
        )

    return check


def _reply_objects(reply: Path) -> tuple[dict[str, Path], str]:
    """Read objects from the lexicographically latest index.

    Other reply filenames contain unordered content hashes, and stale codemodels
    may survive reconfiguration. Do not select them by glob or filename order.
    """
    indexes = sorted(reply.glob("index-*.json")) if reply.is_dir() else []
    if not indexes:
        return {}, ""
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    generator = index.get("cmake", {}).get("generator", {}).get("name", "")
    wanted = {"codemodel": 2, "toolchains": 1}
    objects: dict[str, Path] = {}
    for item in index.get("objects", []):
        version = item.get("version") or {}
        if item.get("kind") in wanted and wanted[item["kind"]] == version.get("major"):
            found = reply / item["jsonFile"]
            if found.is_file():
                objects[item["kind"]] = found
    return objects, generator


_CACHE_COMPILER = re.compile(r"^CMAKE_(?P<language>[A-Z]+)_COMPILER$")
_SEARCH_DIR = re.compile(r'SEARCH_DIR\("(?P<path>[^"]+)"\)')


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """Every entry of a build tree's CMakeCache.txt, by name; empty if it has none."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        entries[line.split(":", 1)[0]] = line.split("=", 1)[1]
    return entries


def _search_directories(toolchains: Path | None, build_dir: Path) -> dict[str, _Search]:
    """Find implicit linker directories by language.

    CMake records only top-level enabled languages; query other compilers directly.
    Append the linker's built-in SEARCH_DIRs after the compiler's directories.
    """
    compilers = {
        match["language"]: value
        for name, value in cmake_cache(build_dir).items()
        if (match := _CACHE_COMPILER.match(name))
    }
    recorded: dict[str, list[str] | None] = {}
    if toolchains is not None:
        payload = json.loads(toolchains.read_text(encoding="utf-8"))
        for item in payload.get("toolchains", []):
            compiler = item.get("compiler", {})
            recorded[item["language"]] = compiler.get("implicit", {}).get(
                "linkDirectories"
            )
            if compiler.get("path"):
                compilers[item["language"]] = compiler["path"]

    found: dict[str, _Search] = {}
    for language in sorted(set(compilers) | set(recorded)):
        compiler = compilers.get(language)
        listed = recorded.get(language)
        if listed is not None:
            directories = [Path(item) for item in listed]
        elif compiler:
            directories = _compiler_search_dirs(compiler)
        else:
            directories = None
        if directories is None:
            continue
        sysroot = _ask(compiler, "-print-sysroot").strip() if compiler else ""
        found[language] = _Search(directories, compiler, sysroot)
    return found


def _ask(*command: str, env: dict[str, str] | None = None) -> str:
    """Return command stdout, or an empty string on failure."""
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


# GCC/Clang depfiles and CMake compiler_depend.make use make rule syntax.
_MAKE_CONTINUATION = re.compile(r"\\\r?\n")
_MAKE_RULE = re.compile(r"^(?:\\.|[^:\\#])+:(?:\s+(?P<prerequisites>.*))?$")
_MAKE_WORD = re.compile(r"(?:\\.|[^\s\\])+")
_MAKE_ESCAPE = re.compile(r"\\(.)")
_NINJA_DEPS = re.compile(r"^(?P<output>\S.*): #deps \d+")
# The `CMakeFiles/<target>.dir/` an output sits in, from the top of the build.
_OBJECTS_DIR = re.compile(r"^(?:.*?/)?CMakeFiles/[^/]+\.dir/")


def _make_rules(text: str) -> list[list[str]]:
    """The prerequisites of each rule of a depfile that names any, unescaped.

    An object's rule names its source first; `-MP`'s rules for headers name
    nothing.
    """
    found = []
    for line in _MAKE_CONTINUATION.sub(" ", text).splitlines():
        match = _MAKE_RULE.match(line)
        words = _MAKE_WORD.findall(match["prerequisites"] or "") if match else []
        if words:
            found.append(
                [_MAKE_ESCAPE.sub(r"\1", word).replace("$$", "$") for word in words]
            )
    return found


def _ninja_deps(build_root: Path) -> dict[str, list[str]]:
    """`ninja -t deps`: each output's recorded dependencies, by output path."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in _ask("ninja", "-C", str(build_root), "-t", "deps").splitlines():
        match = _NINJA_DEPS.match(line)
        if match:
            current = found.setdefault(match["output"], [])
        elif current is not None and line.startswith(" ") and line.strip():
            current.append(line.strip())
    return found


class DependencyLog:
    """What the compiler recorded reading, per target objects directory."""

    def __init__(self, generator: str, build_root: Path) -> None:
        self._ninja_generator = "Ninja" in generator
        self._build_root = build_root
        self._ninja: dict[str, list[list[str]]] | None = None
        self._resolved: dict[str, Path] = {}
        self._read: dict[Path, tuple[set[Path], bool]] = {}

    def read(self, objects: Path, compiled: tuple[Path, ...]) -> tuple[set[Path], bool]:
        """Return source/header prerequisites and whether every compiled source has its own record.

        A source included by another object does not count as its own compile record.
        """
        if objects not in self._read:
            read: set[Path] = set()
            recorded: set[Path] = set()
            wanted, own = set(compiled), objects.resolve()
            for base, prerequisites in self._records(objects):
                paths = [self._resolve_under(base, item) for item in prerequisites]
                # A record for a source the target no longer compiles is stale;
                # one CMake generated into the objects directory (a PCH) is
                # current, except a unity source the codemodel no longer lists.
                source = paths[0]
                generated = source.is_relative_to(own) and not source.is_relative_to(
                    own / "Unity"
                )
                if source not in wanted and not generated:
                    continue
                recorded.add(source)
                read.update(paths)
            self._read[objects] = (read, set(compiled) <= recorded)
        return self._read[objects]

    def _records(self, objects: Path) -> list[tuple[Path, list[str]]]:
        """One record per object, its source first, with the directory its
        relative paths are relative to."""
        if self._ninja_generator:
            return self._ninja_records(objects)
        return self._make_records(objects) if objects.is_dir() else []

    def _ninja_records(self, objects: Path) -> list[tuple[Path, list[str]]]:
        """Ninja's deps log names objects and paths from the top of the build."""
        if self._ninja is None:
            self._ninja = {}
            for output, items in _ninja_deps(self._build_root).items():
                match = _OBJECTS_DIR.match(output)
                if match and items:
                    self._ninja.setdefault(match.group(0), []).append(items)
            if not self._ninja:
                _LOG.warning(
                    "`ninja -t deps` in %s recorded nothing; nested licence "
                    "texts are kept by what targets list and include instead. "
                    "Collect where `ninja` is on PATH.",
                    self._build_root,
                )
        try:
            prefix = objects.relative_to(self._build_root).as_posix() + "/"
        except ValueError:
            return []
        return [(self._build_root, items) for items in self._ninja.get(prefix, [])]

    def _make_records(self, objects: Path) -> list[tuple[Path, list[str]]]:
        """A depfile is written where make runs the compiler, the target's own
        build directory; compiler_depend.make is relative to the top."""
        files = [
            *((objects.parent.parent, path) for path in sorted(objects.rglob("*.d"))),
            (self._build_root, objects / "compiler_depend.make"),
        ]
        return [
            (base, rule)
            for base, path in files
            if path.is_file()
            for rule in _make_rules(path.read_text(encoding="utf-8", errors="replace"))
        ]

    def _resolve_under(self, base: Path, item: str) -> Path:
        joined = os.path.join(base, item)
        if joined not in self._resolved:
            self._resolved[joined] = Path(joined).resolve()
        return self._resolved[joined]


@functools.lru_cache(maxsize=None)
def _compiler_search_dirs(compiler: str) -> list[Path] | None:
    """The `libraries:` list of `<compiler> -print-search-dirs`, GCC's and Clang's."""
    for line in _ask(compiler, "-print-search-dirs").splitlines():
        if line.startswith("libraries:"):
            listed = line.split(":", 1)[1].strip().removeprefix("=")
            return [
                Path(item).resolve()
                for item in listed.split(":")
                if Path(item).is_dir()
            ]
    return None


@functools.lru_cache(maxsize=None)
def _linker_search_dirs(compiler: str, sysroot: str) -> list[Path]:
    """The SEARCH_DIRs of the linker the compiler runs, from `ld --verbose`."""
    linker = _ask(compiler, "-print-prog-name=ld").strip() or "ld"
    found = []
    for match in _SEARCH_DIR.finditer(_ask(linker, "--verbose")):
        directory = _sysrooted(match["path"], sysroot, Path("/"))
        if directory.is_dir():
            found.append(directory)
    return found


class BuildGraph:
    """Read the compile/link graph from CMake's file API.

    Do not scrape link.txt: Ninja does not produce it.
    """

    def __init__(self, build_dir: Path, config: str | None = None) -> None:
        self.build_dir = build_dir
        self.artifacts: dict[Path, Artifact] = {}
        self._nodes: dict[str, _Node] = {}
        self._by_name: dict[str, list[Artifact]] = {}
        self._closures: dict[str, tuple[_Node, ...]] = {}
        self._static: dict[str, dict[LinkInput, set[str]]] = {}
        self._inspected: dict[
            Path, tuple[str | None, int | None, list[str] | None]
        ] = {}
        self._dependencies = DependencyLog("", build_dir)
        self._linked: dict[str, tuple[_Node, ...]] = {}
        self._node_reach: dict[
            tuple[str, Ownership | None], dict[str, dict[Path, str]]
        ] = {}
        self._load(config)

    def _load(self, config: str | None) -> None:
        reply = self.build_dir / ".cmake" / "api" / "v1" / "reply"
        objects, generator = _reply_objects(reply)
        if "codemodel" not in objects:
            raise FileApiError(
                f"no CMake file API codemodel under {reply}. Configure the project "
                "with CMake 3.27 or newer, which is where cmake_file_api() asks for it."
            )
        model_file = objects["codemodel"]
        search = _search_directories(objects.get("toolchains"), self.build_dir)
        # A relative link-line path is relative to where the link runs: the
        # target's own build directory under the Makefile generators, the top
        # of the build tree under Ninja.
        per_directory = "Makefiles" in generator or "WMake" in generator

        codemodel = json.loads(model_file.read_text(encoding="utf-8"))
        source_root = Path(codemodel["paths"]["source"])
        build_root = Path(codemodel["paths"]["build"])
        self._dependencies = DependencyLog(generator, build_root)

        configurations = codemodel["configurations"]
        chosen = next(
            (item for item in configurations if item["name"] == (config or "")), None
        )
        if chosen is None:
            # Do not attribute a wheel to a different build configuration.
            if config:
                raise FileApiError(
                    f"codemodel under {reply} describes no {config!r} configuration, "
                    f"only {[item['name'] for item in configurations]}"
                )
            chosen = configurations[0]

        def resolve(raw: str, root: Path) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else (root / path)

        lines: dict[str, tuple[list[str], list[str], Path, _Search | None]] = {}
        for entry in chosen["targets"]:
            # Target JSON paths are relative to the codemodel file.
            target = json.loads(
                (model_file.parent / entry["jsonFile"]).read_text(encoding="utf-8")
            )
            includes = {
                resolve(item["path"], source_root).resolve()
                for group in target.get("compileGroups", [])
                for item in group.get("includes", [])
            }
            target_dir = resolve(target.get("paths", {}).get("build", "."), build_root)
            link_dir = target_dir if per_directory else build_root
            link = target.get("link", {})
            words, unparsed = _linker_words(link.get("commandFragments", []))
            found = search.get(link.get("language", ""))
            # CMAKE_SYSROOT reaches the linker as --sysroot, which the compiler's
            # own -print-sysroot does not report.
            sysroot = link.get("sysroot", {}).get("path")
            if found is not None and sysroot:
                found = replace(found, sysroot=sysroot)
            lines[entry["id"]] = (words, unparsed, link_dir, found)
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
                compiled=tuple(
                    resolve(item["path"], source_root).resolve()
                    for item in target.get("sources", [])
                    if "compileGroupIndex" in item
                ),
                # Where the Makefile and Ninja generators put a target's objects.
                objects=target_dir / "CMakeFiles" / f"{target['name']}.dir",
            )
            for item in target.get("artifacts", []):
                output = resolve(item["path"], build_root).resolve()
                artifact = Artifact(
                    output=output, target=target["name"], target_id=entry["id"]
                )
                self.artifacts[output] = artifact
                self._by_name.setdefault(output.name, []).append(artifact)

        # Resolve link inputs after indexing every target output.
        for target_id, (words, unparsed, link_dir, found) in lines.items():
            line = self._read_line(words, link_dir, found)
            line.inputs += [
                LinkInput(item, None, "unresolved", "does not parse as shell words")
                for item in unparsed
            ]
            self._nodes[target_id] = replace(
                self._nodes[target_id],
                linked=tuple(dict.fromkeys(line.linked)),
                inputs=tuple(dict.fromkeys(line.inputs)),
            )

    def _read_line(
        self, words: list[str], link_dir: Path, search: _Search | None
    ) -> _Line:
        """Resolve linker inputs using all -L paths, then implicit search paths.

        Shared libraries are handled separately through DT_NEEDED.
        """
        sysroot = search.sysroot if search else ""
        line = _Line(
            link_dir,
            [
                _sysrooted(word[2:], sysroot, link_dir)
                for word in words
                if word.startswith("-L")
            ]
            + (search.directories if search else []),
            sysroot,
        )
        static = False
        saved: list[bool] = []
        pending = iter(words)
        for word in pending:
            if word in _STATIC_MODE:
                static = True
            elif word in _DYNAMIC_MODE:
                static = False
            elif word == "--push-state":
                saved.append(static)
            elif word == "--pop-state":
                static = saved.pop() if saved else False
            elif word.startswith("-l"):
                hit = _search_library(word[2:], line.directories, static)
                if hit is None:
                    reason = _UNSEARCHED if search else _NO_COMPILER
                    line.inputs.append(LinkInput(word, None, "unresolved", reason))
                else:
                    self._classify(word, hit, line, static)
            elif word.startswith("-"):
                if word in _LINKER_TAKES_VALUE:
                    next(pending, None)
            else:
                self._classify(word, link_dir / word, line, static)
        return line

    def _classify(
        self, spelling: str, path: Path, line: _Line, static: bool, depth: int = 0
    ) -> None:
        """Classify inputs, following linker scripts under the current -Bstatic state."""
        # ld looks beside the name it was given, not beside a symlink's target.
        beside = path.parent
        path = path.resolve()
        if path in self.artifacts:
            line.linked.append(path)
            return
        if path not in self._inspected:
            self._inspected[path] = _inspect(path)
        kind, members, entries = self._inspected[path]
        if kind in ("archive", "object"):
            line.inputs.append(LinkInput(spelling, path, kind, members=members))
        elif kind == "missing":
            line.inputs.append(LinkInput(spelling, path, "unresolved", _NOT_FOUND))
        elif kind is None:
            line.inputs.append(LinkInput(spelling, path, "unresolved", _UNKNOWN_KIND))
        elif kind == "script" and depth > _SCRIPT_DEPTH:
            line.inputs.append(
                LinkInput(
                    spelling, path, "unresolved", "linker scripts nest too deeply here"
                )
            )
        elif kind == "script":
            # An absolute name in a script inside the sysroot, or one opening
            # with `=`, is read under the sysroot.
            in_sysroot = bool(line.sysroot) and path.is_relative_to(
                Path(line.sysroot).resolve()
            )
            for entry in entries or []:
                if entry.startswith("-l"):
                    hit = _search_library(entry[2:], line.directories, static)
                else:
                    if entry.startswith("=") or (in_sysroot and entry.startswith("/")):
                        entry = line.sysroot + entry.removeprefix("=")
                    # Beside the script, then where the link runs, then the
                    # search path; an absolute name is itself in each.
                    hit = _search_library(
                        ":" + entry, [beside, line.link_dir, *line.directories], static
                    )
                if hit is None:
                    named = f"{spelling} ({path.name}: {entry})"
                    line.inputs.append(LinkInput(named, None, "unresolved", _NOT_FOUND))
                else:
                    self._classify(spelling, hit, line, static, depth + 1)

    def by_name(self, name: str) -> list[Artifact]:
        return self._by_name.get(name, [])

    def repo_inputs(self, repo_root: Path, build_dir: Path) -> set[Path]:
        """Find compiled/included files belonging to the checkout.

        Exclude the build tree: fetched sources and vcpkg headers have separate
        ownership, and generated headers are not checked-in third-party sources.
        """
        inside = _within(repo_root.resolve(), build_dir.resolve())
        found: set[Path] = set()
        for node in self._nodes.values():
            found.update(item for item in node.sources if inside(item))
            for include in node.includes:
                if inside(include) and include.is_dir():
                    found.update(
                        item
                        for item in include.rglob("*")
                        if item.is_file() and inside(item)
                    )
        return found

    def _reach(self, target_id: str, prune=None) -> tuple[_Node, ...]:
        """Traverse dependencies and linked target outputs, even without declared edges.

        Pruned nodes and paths reachable only through them are excluded; the root remains.
        """
        seen: set[str] = set()
        stack = [target_id]
        found: list[_Node] = []
        while stack:
            current = stack.pop()
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            node = self._nodes[current]
            if prune is not None and current != target_id and prune(node):
                continue
            found.append(node)
            stack.extend(node.dependencies)
            stack.extend(self.artifacts[item].target_id for item in node.linked)
        return tuple(found)

    def _closure(self, target_id: str) -> tuple[_Node, ...]:
        """`_reach`, memoised per instance, not with `lru_cache`: a cache on the
        class would hold every graph ever built alive for the life of the process."""
        if target_id not in self._closures:
            self._closures[target_id] = self._reach(target_id)
        return self._closures[target_id]

    def static_inputs(self, artifact: Artifact) -> dict[LinkInput, set[str]]:
        """Map external static inputs to the targets that link them.

        Exclude build-order executables/modules and their dependencies.
        """
        if artifact.target_id not in self._static:
            found: dict[LinkInput, set[str]] = {}
            for node in self._linked_closure(artifact.target_id):
                for item in node.inputs:
                    found.setdefault(item, set()).add(node.name)
            self._static[artifact.target_id] = found
        return self._static[artifact.target_id]

    def _linked_closure(self, target_id: str) -> tuple[_Node, ...]:
        """`_reach` without other executables and modules, nor what only they
        reach: what can end up inside the artifact."""
        if target_id not in self._linked:
            self._linked[target_id] = self._reach(
                target_id, lambda node: node.kind in ("EXECUTABLE", "MODULE_LIBRARY")
            )
        return self._linked[target_id]

    def contributions(
        self, artifact: Artifact, ownership: Ownership | None = None
    ) -> dict[str, set[str]]:
        """Map contributing components to compiled-source, header-include, or prebuilt roles."""
        found: dict[str, set[str]] = {}

        def note(key: str, how: str) -> None:
            found.setdefault(key, set()).add(how)

        for node in self._closure(artifact.target_id):
            for source in node.sources:
                owner = _owner(source, ownership)
                if owner:
                    note(owner, "compiled-source")
            for include in node.includes:
                for owner in _include_owners(include, ownership):
                    note(owner, "header-include")
        for item in self.static_inputs(artifact):
            owner = owned_input(item, ownership)
            if owner:
                note(owner, PREBUILT)
        return found

    def reached(
        self, artifact: Artifact, ownership: Ownership | None = None
    ) -> dict[str, dict[Path, str]]:
        """The resolved paths of each component that went into an artifact,
        each with its strongest `REACH`."""
        reached: dict[str, dict[Path, str]] = {}
        for node in self._linked_closure(artifact.target_id):
            for key, paths in self._reach_of(node, ownership).items():
                into = reached.setdefault(key, {})
                for path, how in paths.items():
                    note_reach(into, path, how)
        return reached

    def _reach_of(
        self, node: _Node, ownership: Ownership | None
    ) -> dict[str, dict[Path, str]]:
        """One target's owned paths by `REACH`; what it lists and includes only
        where it compiles something its dependency output does not cover."""
        if (node.name, ownership) in self._node_reach:
            return self._node_reach[(node.name, ownership)]
        found: dict[str, dict[Path, str]] = {}

        def mark(owner: str | None, path: Path, how: str) -> None:
            if owner:
                note_reach(found.setdefault(owner, {}), path, how)

        if node.objects is None:
            return found
        for source in node.compiled:
            mark(_owner(source, ownership), source, "compiled-source")
        read, covered = self._dependencies.read(node.objects, node.compiled)
        for path in read:
            mark(_owner(path, ownership), path, "header-read")
        if not covered:
            for source in node.sources:
                mark(_owner(source, ownership), source, "listed-source")
            for include in node.includes:
                for owner in _include_owners(include, ownership):
                    mark(owner, include, "include-dir")
        self._node_reach[(node.name, ownership)] = found
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
    """Resolve ownership across registered component roots and individual files."""

    def __init__(self) -> None:
        # path -> (owner, whether an include directory credits it outright)
        self._files: dict[Path, tuple[str, bool]] = {}
        self._roots: list[tuple[Path, str]] = []
        # directory -> (owners credited outright, owners credited if alone)
        self._index: dict[Path, tuple[set[str], set[str]]] | None = None
        # Cache repeated header lookups across compilation units.
        self._owners: dict[Path, str | None] = {}

    def register_root(self, root: Path, key: str) -> None:
        self._roots.append((root.resolve(), key))
        self._owners.clear()

    def roots_of(self, key: str) -> list[Path]:
        """The resolved trees registered for a component."""
        return [root for root, owner in self._roots if owner == key]

    def register_file(self, path: Path, key: str, *, by_directory: bool = True) -> None:
        """Register a file's owner.

        With by_directory=False, an include directory credits this owner only if
        it owns every registered file beneath that directory.
        """
        self._files[path.resolve()] = (key, by_directory)
        self._index = None
        self._owners.clear()

    def owner_of(self, path: Path) -> str | None:
        if path not in self._owners:
            self._owners[path] = self._find_owner(path)
        return self._owners[path]

    def _find_owner(self, path: Path) -> str | None:
        resolved = path.resolve()
        entry = self._files.get(resolved)
        if entry is not None:
            return entry[0]
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
        resolved = directory.resolve()
        outright, alone = self._by_directory().get(resolved, (set(), set()))
        found = set(outright)
        prefix = str(resolved)
        found.update(
            key
            for root, key in self._roots
            if str(root) == prefix or str(root).startswith(prefix + "/")
        )
        if len(alone) == 1:
            found |= alone
        return found

    def _by_directory(self) -> dict[Path, tuple[set[str], set[str]]]:
        if self._index is None:
            self._index = {}
            for path, (key, outright) in self._files.items():
                for directory in (path, *path.parents):
                    entry = self._index.setdefault(directory, (set(), set()))
                    entry[0 if outright else 1].add(key)
        return self._index


# Vendored sources: identify third parties by copyright holder, not directory.

# Accept REUSE tags and plain upstream copyright notices.
_COPYRIGHT_TAG = re.compile(
    r"SPDX-FileCopyrightText:\s*(?P<holder>[^\n\r]+)"
    r"|^[^\w\n]*Copyright\b(?P<plain>[^\n\r]+)",
    re.MULTILINE,
)
# Comment markup around a copyright line: `//`, `#`, ` * `, `<!-- ... -->`, `/* ... */`.
_COMMENT_MARKUP = re.compile(r"^(?:<!--|[/*#\s])+|\s*(?:-->|\*/)\s*$")
_YEARS = re.compile(r"(copyright|\(c\)|©|\d{4}(\s*-\s*\d{4})?|,|\.)", re.IGNORECASE)


def project_authors(repo_root: Path) -> set[str]:
    """Read author names verbatim; supplier output must retain their casing."""
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
    """Require exactly one declared author for SPDX's single supplier field."""
    authors = sorted(project_authors(repo_root))
    if len(authors) != 1:
        raise SupplierError(
            f"{repo_root}/pyproject.toml names {len(authors)} project authors "
            f"({', '.join(authors) or 'none'}); SPDX records one supplier for "
            "the distribution"
        )
    return f"Organization: {authors[0]}"


def project_copyright(repo_root: Path) -> str:
    """Read the project notice from its packaging file's REUSE header.

    Metadata-only wheels have no source notice, and reference license texts
    do not identify the distribution's copyright holder.
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return "NOASSERTION"
    try:
        head = pyproject.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return "NOASSERTION"
    return licensing.read_copyright(head) or "NOASSERTION"


@dataclass(frozen=True)
class _Stated:
    """Who holds a file, and the notices that say so."""

    holder: str
    notices: tuple[str, ...]


def _holder_in(text: str) -> str:
    """The holder a copyright statement names, without years or boilerplate."""
    holder = " ".join(_YEARS.sub(" ", _COMMENT_MARKUP.sub("", text)).split())
    return holder.removesuffix(" All rights reserved").strip()


def _notice_line(text: str) -> str:
    notice = _COMMENT_MARKUP.sub("", " ".join(text.split()))
    return re.sub(r"^SPDX-(?:File|Snippet)CopyrightText:\s*", "", notice)


def _stated_by(path: Path) -> _Stated | None:
    """Holder and notice, from the first copyright line in a text file's head."""
    from reuse.extract import HEURISTICS_CHUNK_SIZE, detect_encoding

    try:
        with path.open("rb") as handle:
            raw = handle.read(4096)
    except OSError:
        return None
    # REUSE's own text test: a PNG's `Copyright` text chunk is not a header.
    encoding = detect_encoding(raw[:HEURISTICS_CHUNK_SIZE])
    if encoding is None:
        return None
    match = _COPYRIGHT_TAG.search(raw.decode(encoding, "replace"))
    if not match:
        return None
    holder = _holder_in(match.group("holder") or match.group("plain") or "")
    return _Stated(holder, (_notice_line(match.group(0)),)) if holder else None


def _header_applies(project, used: Path, infos) -> bool:
    """Whether REUSE applies the file's own text beside its declared info.

    It does where it took info from the header, and under `aggregate` whatever
    it parsed -- unless a `.license` sidecar exists or any `override` applies,
    the conditions `Project.reuse_info_of` tests before reading the file.
    """
    from reuse.copyright import SourceType
    from reuse.global_licensing import PrecedenceType

    if any(info.source_type is SourceType.FILE_HEADER for info in infos):
        return True
    if project.global_licensing is None or Path(f"{used}.license").exists():
        return False
    found = project.global_licensing.reuse_info_of(project.relative_from_root(used))
    return PrecedenceType.AGGREGATE in found and PrecedenceType.OVERRIDE not in found


def _held_by(project, used: Path, infos, authors: set[str]) -> _Stated | None:
    """Who other than the project holds a file, as a `_Stated`, if anyone does.

    A sidecar or REUSE.toml notice naming someone else makes the file theirs,
    under the first such holder by name. The file's first copyright line counts
    as well unless REUSE applied declared notices that all name a holder in the
    file's place: a sidecar, `override`, or `closest` where REUSE found nothing
    in the file. Then the file is ours if those all name the project.
    """
    from reuse.copyright import SourceType

    def ours(holder: str) -> bool:
        return any(author.lower() in holder.lower() for author in authors)

    # (holder, notice, whether it names the project), once per declared notice.
    declared = [
        (
            _holder_in(notice.name),
            _notice_line(notice.to_string(original=True)),
            ours(notice.name),
        )
        for info in infos
        if info.source_type is not SourceType.FILE_HEADER
        for notice in info.copyright_notices
    ]
    set_aside = (
        bool(declared)
        and all(holder for holder, _, _ in declared)
        and not _header_applies(project, used, infos)
    )
    stated = None if set_aside else _stated_by(used)
    scraped = stated if stated is not None and not ours(stated.holder) else None
    foreign = [(holder, line) for holder, line, mine in declared if holder and not mine]
    if not foreign:
        return scraped
    holder = min(foreign, key=lambda item: (item[0].casefold(), item))[0]
    lines = [line for _, line in foreign] + list(scraped.notices if scraped else ())
    return _Stated(holder, tuple(lines))


def _covered(project, path: Path, *, built: bool) -> bool:
    """Whether REUSE covers a file.

    For a file the build read, only what REUSE excludes by name counts: a link
    is judged by its target, and a submodule, a meson subproject or a
    VCS-ignored file it compiled still ships.
    """
    from reuse.covered_files import is_path_ignored

    if not built and path.is_symlink():
        return False
    real = path.resolve()
    if not real.is_relative_to(project.root) or not real.is_file():
        return False
    # With no VCS strategy REUSE checks neither ignore rules nor submodules.
    return not any(
        is_path_ignored(
            item,
            include_submodules=project.include_submodules,
            include_meson_subprojects=built or project.include_meson_subprojects,
            vcs_strategy=None if built else project.vcs_strategy,
        )
        for item in (real, *real.parents)
        if item != project.root and item.is_relative_to(project.root)
    )


def _info_for(project, own: Path, target: Path):
    """REUSE info for a file's holder and for its licence, and the holder's path.

    A link's own declarations are its sidecar and the annotations on its path.
    Where they give a field, REUSE's view of the link decides it, as for a file
    in its place; otherwise REUSE's view of the target, which applies the
    target's precedence to the bytes they share.
    """
    from reuse.copyright import SourceType

    infos = project.reuse_info_of(own)
    if target == own:
        return infos, infos, own
    mine = [info for info in infos if info.source_type is not SourceType.FILE_HEADER]
    theirs = project.reuse_info_of(target)
    licensed = infos if any(info.spdx_expressions for info in mine) else theirs
    if any(info.copyright_notices for info in mine):
        return infos, licensed, own
    return theirs, licensed, target


def discover_vendored(
    repo_root: Path,
    candidates: set[Path],
    authors: set[str],
    *,
    tracked: Iterable[str],
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Group third-party files from build inputs and REUSE-covered tracked files.

    Tracked files include data and Python absent from the compile graph.
    Licenses follow REUSE precedence; holders follow _held_by.
    """
    # Here, not at module level: only discovery needs reuse, and reuse needs libmagic.
    from reuse.project import Project

    project = Project.from_directory(repo_root.resolve())
    built = {path for path in candidates if _covered(project, path, built=True)}
    candidates = built | {
        path
        for path in (repo_root / item for item in tracked)
        if _covered(project, path, built=False)
    }
    grouped: dict[str, list[Path]] = {}
    licences: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    notices: dict[str, set[str]] = {}
    unlicensed: dict[str, list[tuple[str, str]]] = {}

    for path in sorted(candidates):
        under_root = path.relative_to(repo_root)
        own = project.root / under_root
        infos, licensed, used = _info_for(project, own, path.resolve())
        stated = _held_by(project, used, infos, authors)
        if stated is None:
            continue
        reuse_declared = [
            expression for info in licensed for expression in info.spdx_expressions
        ]
        key = f"vendored:{licensing.spdx_safe(stated.holder).lower()}"
        grouped.setdefault(key, []).append(path)
        names[key] = stated.holder
        notices.setdefault(key, set()).update(stated.notices)
        reason = licensing.why_unlicensed(reuse_declared)
        if reason:
            unlicensed.setdefault(key, []).append((str(under_root), reason))
        else:
            licences.setdefault(key, set()).update(map(str, reuse_declared))

    components: dict[str, Component] = {}
    for key, paths in grouped.items():
        concluded, evidence = licensing.declared_terms(key, licences.get(key, set()))
        declared = concluded
        # List build inputs first and mark truncation explicitly.
        named = [
            name
            for _, name in sorted(
                (item not in built, str(item.relative_to(repo_root))) for item in paths
            )
        ]
        relative = ", ".join(named[:4]) + (
            f", and {len(named) - 4} more" if len(named) > 4 else ""
        )
        components[key] = Component(
            key=key,
            name=f"Vendored source ({names[key]})",
            kind="vendored-source",
            supplier=f"Organization: {names[key]}",
            homepage="NOASSERTION",
            # A copyright holder is not a package identity; do not mint a purl from it.
            purl="NOASSERTION",
            version="NOASSERTION",
            download_location="NOASSERTION",
            source_info=(
                f"Checked into this repository rather than fetched; its REUSE info "
                f"or header names its copyright holder. Files: {relative}."
            ),
            license_concluded=concluded,
            license_declared=declared,
            copyright_text="\n".join(
                licensing.fold_notices(notices.get(key, {names[key]}))
            ),
            evidence=tuple(evidence),
            unlicensed=tuple(unlicensed.get(key, ())),
        )
    return components, grouped


# vcpkg install records identify ports whose outputs are outside the CMake graph.


def discover_vcpkg(
    installed: Path, target_triplet: str | None = None
) -> tuple[dict[str, Component], dict[str, list[Path]]]:
    """Every port vcpkg installed under `installed`, and the files each owns.

    A port installed for the host as well is described from its install for
    `target_triplet` -- what the build links -- where there is one.
    """
    info = installed / "vcpkg" / "info"
    components: dict[str, Component] = {}
    owned: dict[str, list[Path]] = {}
    for listing in sorted(info.glob("*.list")) if info.is_dir() else []:
        # <port>_<version>_<triplet>.list; port names carry no underscore.
        port, _, rest = listing.stem.partition("_")
        version, _, triplet = rest.rpartition("_")
        key = f"vcpkg:{port}"
        owned.setdefault(key, []).extend(
            installed / line
            for line in listing.read_text(encoding="utf-8").splitlines()
            if line and not line.endswith("/")
        )
        if key not in components or triplet == target_triplet:
            components[key] = _vcpkg_component(key, port, version, triplet, installed)
    return components, owned


def _vcpkg_component(
    key: str, port: str, version: str, triplet: str, installed: Path
) -> Component:
    """A port, identified by the upstream source its own SBOM says it built.

    vcpkg writes `share/<port>/vcpkg.spdx.json` with the recipe and the
    downloaded sources, and copies upstream's licence to `share/<port>/copyright`.
    """
    share = installed / triplet / "share" / port
    try:
        sbom = json.loads((share / "vcpkg.spdx.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sbom = {}
    packages = {item.get("SPDXID"): item for item in sbom.get("packages", [])}
    recipe = packages.get("SPDXRef-port", {})
    sources = [
        item["downloadLocation"]
        for spdx_id, item in packages.items()
        if str(spdx_id).startswith("SPDXRef-resource-") and item.get("downloadLocation")
    ]
    port_version = recipe.get("versionInfo") or version
    # `#<n>` is the recipe's revision, not anything upstream released.
    version = port_version.partition("#")[0]
    # One source names the upstream; several (or none) name no single one.
    upstream = sources[0] if len(sources) == 1 else None
    location = (upstream or "").removeprefix("git+")
    remote, _, ref = (
        location.rpartition("@")
        if "@" in location.rsplit("/", 1)[-1]
        else (location, "", "")
    )
    if remote and _GITHUB_REMOTE.search(remote):
        identity = _identity(remote, port, ref or None)
        supplier, homepage, purl = identity.supplier, identity.homepage, identity.purl
        download = identity.download_location
    else:
        supplier, homepage, download = "NOASSERTION", "NOASSERTION", upstream
        purl = PackageURL(type="generic", name=port, version=version).to_string()
    homepage = recipe.get("homepage") or homepage

    evidence = []
    licence = share / "copyright"
    if licence.is_file():
        evidence.append(
            licensing.evidence_for(
                key,
                "vcpkg-port",
                licence.relative_to(installed.parent).as_posix(),
                licence.read_text(encoding="utf-8", errors="replace"),
                "grant",
            )
        )
    concluded, declared = licensing.expression(evidence, key)
    details = [
        f"Built and installed by vcpkg port {port} {port_version} for {triplet}."
    ]
    if upstream:
        details.append(f"Source: {upstream}.")
    if recipe.get("downloadLocation"):
        details.append(f"Port recipe: {recipe['downloadLocation']}.")
    return Component(
        key=key,
        name=port,
        kind="vcpkg-port",
        supplier=supplier,
        homepage=homepage,
        purl=purl,
        version=version,
        download_location=download or "NOASSERTION",
        source_info=" ".join(details),
        license_concluded=concluded,
        license_declared=declared,
        copyright_text=notice_from(evidence),
        evidence=tuple(evidence),
    )


# Build-host libraries: resolve inputs not explained by build or source indexes.

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
    for line in _ask("ldconfig", "-p").splitlines():
        _, _, target = line.partition("=>")
        if target.strip():
            paths.add(Path(target.strip()))
    for root in _library_roots():
        paths.update(item for item in root.glob("*.so*") if item.is_file())
    return sorted(paths)


@functools.lru_cache(maxsize=1)
def _host_build_id_index() -> dict[str, str]:
    """Index host libraries by build ID, which survives SONAME/RPATH repair.

    Cache the index to avoid rescanning host libraries for every wheel member.
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


def _host_origin(library: Path | None, **known) -> dict:
    """Resolve a host file and identify its supplying package and terms."""
    origin: dict = {**known, "resolved_path": None, "package": None, "copyright": None}
    if library is None:
        return origin
    library = library.resolve()
    origin["resolved_path"] = str(library)
    return _identify_host_package(library, origin)


def resolve_system_library_by_build_id(build_id: str) -> dict:
    """Trace a vendored library back by build-id, whatever it was renamed to."""
    found = _host_build_id_index().get(build_id)
    library = Path(found).resolve() if found else None
    return _host_origin(
        library, soname=library.name if library else None, build_id=build_id
    )


def resolve_system_library(soname: str) -> dict:
    """Trace an auditwheel-vendored library back to the package that supplied it."""
    library = next(
        (root / soname for root in _library_roots() if (root / soname).exists()), None
    )
    return _host_origin(library, soname=soname)


def resolve_static_archive(archive: Path) -> dict:
    """Identify the package or toolkit supplying a linked archive or object.

    The link line already supplies its path; existence alone proves no ownership.
    """
    if not archive.is_file():
        return _host_origin(None, toolkit_root=None)
    root = _toolkit_root(archive)
    return _host_origin(archive, toolkit_root=str(root) if root else None)


def archive_members(path: Path) -> int | None:
    """How many object files a `!<arch>` archive holds; None if it is not one.

    glibc 2.34 folded librt, libpthread and libdl into libc and left each as an
    archive with no members, so naming one on a link line links nothing.
    """
    try:
        with path.open("rb") as handle:
            magic = handle.read(8)
            if magic not in (b"!<arch>\n", b"!<thin>\n"):
                return None
            count = 0
            while header := handle.read(60):
                if len(header) < 60 or header[58:60] != b"`\n":
                    return None
                name = header[:16].rstrip()
                size = int(header[48:58])
                skip = size + size % 2
                # BSD writes a long name at the head of the member's data.
                if name.startswith(b"#1/"):
                    length = int(name[3:])
                    name = handle.read(length).rstrip(b"\0")
                    skip -= length
                # The symbol table and the long-name table are not members.
                index = name in (b"/", b"//", b"/SYM64/") or name.startswith(
                    b"__.SYMDEF"
                )
                count += not index
                # A thin archive stores only its index; members stay outside it.
                if index or magic == b"!<arch>\n":
                    handle.seek(skip, os.SEEK_CUR)
            return count
    except (OSError, ValueError):
        return None


def _identify_host_package(library: Path, origin: dict) -> dict:
    # dpkg names a multiarch package with its architecture, `libc6:i386`, and
    # dpkg-query -W and apt answer for one architecture only when asked so.
    qualified = _dpkg_owner(library)
    if not qualified:
        return origin
    package = qualified.partition(":")[0]
    origin["package"] = package
    origin["supplier"] = _package_supplier(qualified)
    origin["version"] = _package_field(qualified, "Version")
    origin["apt_origin"], origin["apt_origin_from"] = _apt_origin(qualified)
    copyright_file = Path("/usr/share/doc") / package / "copyright"
    if copyright_file.is_file():
        origin["copyright"] = str(copyright_file)
    return origin


_DPKG_PACKAGE = re.compile(r"^[a-z0-9][a-z0-9+.-]+(?::[a-z0-9-]+)?$")


def _dpkg_owner(library: Path) -> str | None:
    """The first package `dpkg-query -S` names as owning `library`.

    Its last line answers, `<package>[, <package>...]: <path>`; diversion lines
    come before it, translated. Where `library` is diverted, the file there may
    not be the listed package's, so none is named.
    """
    lines = _ask("dpkg-query", "-S", str(library)).splitlines()
    suffix = f": {library}"
    if not lines or any(line.endswith(suffix) for line in lines[:-1]):
        return None
    if not lines[-1].endswith(suffix):
        return None
    owner = lines[-1].removesuffix(suffix).split(",", 1)[0].strip()
    return owner if _DPKG_PACKAGE.match(owner) else None


def _package_field(package: str, field: str) -> str | None:
    return _ask("dpkg-query", "-W", f"-f=${{{field}}}", package).strip() or None


# `apt-cache policy` lines. Its headings are translated; these are not. With no
# package named: a source (`%4i <description>`) and, directly under it, its
# Release fields. In a version table: a version (`***` when installed, perhaps
# with a note such as `(phased 20%)`) and, indented further, a source listing it.
_APT_SOURCE = re.compile(r"^(?P<indent> *)-?\d+ (?P<source>\S.*?)\s*$")
_APT_RELEASE = re.compile(r"^ {5}release (?P<fields>.*)$")
_APT_RELEASE_FIELD = re.compile(r",(?=[a-z]=)")
_APT_VERSION = re.compile(r"^ (?P<installed>\*\*\*|   ) \S+ -?\d+(?: .*)?$")


@functools.lru_cache(maxsize=1)
def _apt_releases() -> dict[str, dict[str, str]]:
    """Each source apt knows, with the fields of its Release file (`o` is Origin)."""
    releases: dict[str, dict[str, str]] = {}
    source = None
    for line in _ask("apt-cache", "policy").splitlines():
        listed = _APT_SOURCE.match(line)
        if listed and len(listed["indent"]) <= 3:
            source = listed["source"]
            releases[source] = {}
        elif source and (release := _APT_RELEASE.match(line)):
            releases[source] = dict(
                item.partition("=")[::2]
                for item in _APT_RELEASE_FIELD.split(release["fields"])
            )
            source = None
        else:
            source = None
    return releases


@functools.lru_cache(maxsize=None)
def _apt_origin(package: str) -> tuple[str | None, str | None]:
    """`(Origin, "installed version")` where the archives listing the installed
    version agree on one, ignoring case; else `(Origin, "other versions")` where
    none lists it and those listing other versions agree; else `(None, None)`.
    """
    releases = _apt_releases()
    # Installed or not: lowercased Origin ("" for none) -> first spelling.
    origins: dict[bool, dict[str, str]] = {True: {}, False: {}}
    installed = False
    for line in _ask("apt-cache", "policy", package).splitlines():
        if version := _APT_VERSION.match(line):
            installed = version["installed"] == "***"
        elif (listed := _APT_SOURCE.match(line)) and len(listed["indent"]) >= 5:
            fields = releases.get(listed["source"], {})
            # dpkg's status file, release `a=now`, says what is installed, not
            # where it came from.
            if fields.get("a") != "now":
                origin = fields.get("o", "")
                origins[installed].setdefault(origin.lower(), origin)
    decided_by_installed = bool(origins[True])
    found = origins[decided_by_installed]
    if len(found) == 1 and "" not in found:
        decided = "installed version" if decided_by_installed else "other versions"
        return next(iter(found.values())), decided
    return None, None


def _package_supplier(package: str) -> str | None:
    """Format the package Maintainer as an SPDX supplier; the package name is not one."""
    maintainer = _package_field(package, "Maintainer")
    if not maintainer:
        return None
    # "Name <email>" is Debian's form; SPDX wants "Organization: Name (email)".
    match = re.match(r"^(?P<name>.+?)\s*<(?P<email>[^>]+)>$", maintainer)
    if match:
        return f"Organization: {match['name']} ({match['email']})"
    return f"Organization: {maintainer}"


def _library_roots() -> list[Path]:
    roots = [Path(item) for item in _SEARCH_ROOTS if Path(item).is_dir()]
    return roots + [root / "lib64" for root in _toolkit_roots()]


@functools.lru_cache(maxsize=None)
def _toolkit_roots() -> list[Path]:
    """Toolkits that may install outside the package manager: CUDA's."""
    return sorted(
        {
            Path(item).resolve()
            for item in glob.glob("/usr/local/cuda*")
            if Path(item).is_dir()
        }
    )


def _toolkit_root(path: Path) -> Path | None:
    resolved = path.resolve()
    return next(
        (root for root in _toolkit_roots() if resolved.is_relative_to(root)), None
    )


def _license_near(library: Path) -> Path | None:
    """Find terms beside a library, searching up to its toolkit root.

    CUDA can install outside dpkg, with targets/<arch>/lib below the root EULA.txt.
    """
    root = _toolkit_root(library)
    current = library.resolve().parent
    levels = 3 if root is None else len(current.relative_to(root).parts) + 1
    for _ in range(levels):
        for entry in sorted(current.iterdir()) if current.is_dir() else []:
            if entry.is_file() and licensing.classify(
                entry.name, licensing.files_in(current)
            ):
                return entry
        if current.parent == current:
            break
        current = current.parent
    return None


def _system_notice(evidence: list[licensing.LicenseEvidence]) -> str:
    """Read holders from machine-readable Debian copyright fields, or NOASSERTION.

    Do not scrape free-form agreements: bundled notices can name other components.
    """
    for item in evidence:
        fields = licensing.dep5_copyright(item.text)
        if fields:
            return "\n".join(licensing.fold_notices(fields))
    return "NOASSERTION"


def _system_purl(soname: str, origin: dict) -> str:
    """`pkg:deb/<vendor>/...` for a package an archive's Origin names, else generic.

    The deb type requires its namespace, the vendor, lowercased. ECMA-427 5.6.3
    keeps `/` out of a namespace segment and a URL host out of a namespace, so
    an Origin that is a URL or a host name names no vendor either.
    """
    package = origin.get("package")
    vendor = (origin.get("apt_origin") or "").lower() if package else ""
    if "/" in vendor or ("." in vendor and not re.search(r"\s", vendor)):
        vendor = ""
    return PackageURL(
        type="deb" if vendor else "generic",
        namespace=vendor or None,
        name=package or soname,
        version=origin.get("version"),
    ).to_string()


def system_component(
    soname: str, origin: dict, route: str | None = None, key: str | None = None
) -> Component:
    """Describe a host library; route defaults to auditwheel and key to system:<soname>."""
    evidence: list[licensing.LicenseEvidence] = []
    key = key or f"system:{soname}"
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
    # Use the component key: different libraries can share a SONAME but not terms.
    concluded, declared = licensing.expression(evidence, key)

    package = origin.get("package")
    details = [
        route
        or f"Vendored into the wheel by auditwheel from {origin.get('resolved_path') or soname}."
    ]
    if package:
        details.append(f"Supplied by system package {package}.")
    return Component(
        key=key,
        name=soname,
        kind="system-library",
        supplier=origin.get("supplier") or "NOASSERTION",
        copyright_text=_system_notice(evidence),
        homepage="NOASSERTION",
        purl=_system_purl(soname, origin),
        version=origin.get("version") or "NOASSERTION",
        download_location="NOASSERTION",
        source_info=" ".join(details),
        license_concluded=concluded,
        license_declared=declared,
        evidence=tuple(evidence),
    )


# Content indexes


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
        # Build IDs survive auditwheel/patchelf RPATH and SONAME changes.
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
        at_root = frozenset(
            Path(member.path).name
            for member, _ in members
            if Path(member.path).parent.as_posix() in {".", ""}
        )
        for member, data in members:
            # Nested members retain only a prefix; read their build ID, not full ELF metadata.
            build_id = elf.build_id_of(data)
            if build_id:
                member = replace(member, build_id=build_id)
                self.all_by_build_id.setdefault(build_id, []).append(member)
            self.by_hash.setdefault(member.sha256, member)
            self.all_by_hash.setdefault(member.sha256, []).append(member)
            self.by_name.setdefault(Path(member.path).name, []).append(member)
            self._maybe_license(display, member, data, at_root)

    def _maybe_license(
        self, display: str, member: Member, data: bytes, at_root: frozenset[str]
    ) -> None:
        name = Path(member.path).name
        if Path(member.path).parent.as_posix() not in {".", ""}:
            return  # only the archive root states the bundle's own terms
        if name.upper() == "VERSION" and len(data) < 256:
            # Version files may contain extra lines that are not part of the version.
            text = data.decode("utf-8", "replace").strip().splitlines()
            self.archives[display]["version_text"] = text[0].strip() if text else ""
            return
        if not licensing.classify(name, at_root.__contains__):
            return
        self.archives[display].setdefault("license_members", []).append((member, data))

    @staticmethod
    def _read_members(path: Path, display: str) -> list[tuple[Member, bytes]]:
        """Stream member hashes; retain root files and ELF prefixes only."""
        collected: list[tuple[Member, bytes]] = []

        def take(name: str, handle) -> None:
            # Retain root license texts and nested ELF prefixes without buffering large members.
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
            # Report unreadable fetched archives directly, before attribution failures obscure them.
            raise ArchiveError(f"{display} cannot be read: {error}") from error
        return collected


def tracked_files(repo_root: Path) -> list[str]:
    """List tracked files; a directory walk would also treat install output as source."""
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
    """Index tracked or explicitly fetched archives, excluding build outputs.

    Require held explicitly; an omitted tracked-file list silently loses provenance.
    """
    index = ArchiveIndex()
    skipped = tuple(item.resolve() for item in skip)
    for path in _walk(repo_root):
        # Compare path segments: skipping dist must not skip distribution.
        if any(path.is_relative_to(item) for item in skipped):
            continue
        name = path.name.lower()
        if not name.endswith(ARCHIVE_SUFFIXES):
            continue
        relative = path.relative_to(repo_root).as_posix()
        # Untracked archives need a download marker; staged copies cannot prove origin.
        if relative not in held and not path.with_name(path.name + ".source").is_file():
            continue
        index.add_archive(path, relative)
    return index


class FileIndex:
    """Files on disk by content hash, for tracing a copied wheel member back."""

    def __init__(self) -> None:
        self.by_hash: dict[str, str] = {}
        # Keep all candidates: identical license texts or source files do not prove ownership.
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
        # Resolve relative skip paths and compare segments, not string prefixes.
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
        # Resolve once per tree: _walk does not cross symlinks.
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
            # Include the parent directory to distinguish common names such as __init__.py.
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
