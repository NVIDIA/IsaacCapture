# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify component license texts against the SPDX corpus.

Preserve unmatched terms verbatim and report them as unidentified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from license_expression import (
    ExpressionError,
    LicenseWithExceptionSymbol,
    combine_expressions,
    get_spdx_licensing,
)
from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import Reach


# Match standalone grant names: BSD-LICENSE and LICENSE.md, not license indexes.
# Notices are packaged separately and do not determine the license expression.
_GRANT_NAMES = re.compile(
    r"(^|[-_.])(LICEN[CS]E|COPYING|UNLICENSE|EULA)([-_.]|$)", re.IGNORECASE
)
_NOTICE_NAMES = re.compile(r"^(NOTICE|COPYRIGHT|AUTHORS|PATENTS)", re.IGNORECASE)
_README_NAMES = re.compile(r"^README", re.IGNORECASE)
# Skip machine-readable siblings; the text beside them is the evidence.
_SKIP_SUFFIXES = {".py", ".cmake", ".in", ".json", ".yaml", ".yml", ".spdx"}

_WORD = re.compile(r"[^a-z0-9]+")
_NGRAM = 6
# A canonical text has to be almost entirely present before it is named.
_CONTAINMENT_THRESHOLD = 0.90
# Coverage rejects an OSS license quoted in a larger agreement's appendix
# (for example CUDA's EULA) as the agreement's own terms.
_COVERAGE_THRESHOLD = 0.20
# Carry partially explained grants too: unmatched prose may add terms, as
# in fmt's MIT exception. Exclude copyright notices from this coverage measure.
_EXPLAINED_THRESHOLD = 0.80
# Two licenses are one family when either text is nearly inside the other.
_FAMILY_THRESHOLD = 0.90
# REUSE-IgnoreStart
_SPDX_TAG = re.compile(r"SPDX-License-Identifier:\s*(?P<expression>[^\n\r*/#]+)")
_FILE_COPYRIGHT = re.compile(r"SPDX-FileCopyrightText:\s*(?P<notice>[^\n\r]+)")
# REUSE-IgnoreEnd
# Require a licensing heading; a title such as "License List Data" is not a grant.
_README_LICENSE_SECTION = re.compile(
    r"^#{1,6}\s*(licen[cs]e|licen[cs]es|licensing|licen[cs]e information|copyright|legal)"
    r"\s*$(?P<body>.*?)(?=^#{1,6}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class LicenseEvidence:
    """A license text that was read out of the component, not assumed."""

    component: str
    # "component-file" | "archive-member" | "readme-section" | "build-host"
    # | "vcpkg-port"
    origin: str
    path: str
    sha256: str
    size: int
    text: str
    # "grant" is the component's own license, "notice" an attribution file,
    # "pool" a REUSE LICENSES/ entry that applies to individual files only, and
    # "nested" terms stated for one subdirectory of a larger component.
    kind: str
    # Display text for a reader, not an expression; the ids are in `matches`.
    identified: str | None = None
    matches: tuple[tuple[str, float, float], ...] = ()
    spdx_tag: str | None = None
    # Nested terms cover a directory or one sidecar-annotated file. reached_by
    # records the strongest evidence that this scope contributed to the wheel.
    scope: str | None = None
    reached_by: Reach | None = None

    def as_json(self) -> dict:
        found = {
            "component": self.component,
            "origin": self.origin,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "kind": self.kind,
            "identified": self.identified,
            "matches": [
                {
                    "license": license_id,
                    "containment": round(containment, 4),
                    "coverage": round(coverage, 4),
                }
                for license_id, containment, coverage in self.matches
            ],
            "spdx_tag": self.spdx_tag,
        }
        if self.reached_by:
            found["reached_by"] = self.reached_by._asdict()
        return found


# A REUSE pool satisfies text packaging even when no component-wide license
# can be concluded. Share this predicate across build, report, and verification.
OBLIGATION_KINDS = frozenset({"grant", "pool"})


def satisfies_obligation(kind: str) -> bool:
    return kind in OBLIGATION_KINDS


def obligation_texts(evidence) -> list[LicenseEvidence]:
    return [item for item in evidence if satisfies_obligation(item.kind)]


def grant_texts(evidence) -> list[LicenseEvidence]:
    return [item for item in evidence if item.kind == "grant"]


def pool_texts(evidence) -> list[LicenseEvidence]:
    return [item for item in evidence if item.kind == "pool"]


def pool_names(paths) -> str:
    return ", ".join(sorted(Path(path).stem for path in paths))


def carried_texts(evidence) -> list[LicenseEvidence]:
    """The grants a component's LicenseRef stands for."""
    return [item.text for item in _reading(evidence).carried]


def carried_and_named(evidence) -> list[LicenseEvidence]:
    """Carried grants whose ids the conclusion names as well."""
    return _reading(evidence).also_named


def verbatim_terms(evidence) -> str:
    """Join carried grants, or all grants if none need carrying.

    Build and verification must use identical joins when binding these terms by hash.
    """
    texts = carried_texts(evidence) or grant_texts(evidence)
    return "\n\n".join(item.text for item in texts)


_DIGITS = re.compile(r"\d+")


def _normalize(text: str) -> list[str]:
    """Normalize years and clause numbers for matching.

    Retain copyright lines: BSD and MIT clauses contain that word, and deleting
    whole lines would remove terms differently depending on upstream wrapping.
    """
    return [
        word for word in _WORD.sub(" ", _DIGITS.sub(" ", text.lower())).split() if word
    ]


def _ngrams(words: list[str]) -> set[str]:
    if len(words) < _NGRAM:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[index : index + _NGRAM])
        for index in range(len(words) - _NGRAM + 1)
    }


@dataclass(frozen=True)
class Reference:
    """An SPDX reference with n-grams for matching and verbatim text for packaging."""

    name: str
    grams: frozenset[str]
    text: str = ""

    @property
    def size(self) -> int:
        return len(self.grams)


class CorpusError(Exception):
    """The SPDX reference texts this build fetched are missing or unreadable."""


_CORPUS: tuple[str, dict[str, Reference]] | None = None
_CORPUS_META: dict[str, object] = {}
_LISTED: "_Listed | None" = None
# Cache parsing and n-gram construction by corpus directory within this process.
_PARSED: dict[str, tuple[str, dict[str, Reference], "_Listed"]] = {}


@dataclass(frozen=True)
class _Listed:
    """Every id the pinned list names, deprecated ones included."""

    licences: frozenset[str]
    exceptions: frozenset[str]
    # Exception reference texts, from json/exceptions/<id>.json.
    exception_texts: dict[str, str]


def corpus_provenance() -> dict[str, object]:
    """Which reference texts an identification was made against."""
    return dict(_CORPUS_META)


def load_corpus(json_dir: Path) -> str:
    """Load the pinned SPDX corpus from its json/ directory.

    Retain all identifiers and exception texts for declaration checks; exclude
    deprecated license IDs from matching candidates.
    """
    global _CORPUS, _LISTED
    details = json_dir / "details"
    index_file = json_dir / "licenses.json"
    if not details.is_dir() or not index_file.is_file():
        raise CorpusError(
            f"SPDX license list data not found at {json_dir}. It is fetched by the "
            "build (deps/third_party); configure and build, or pass --license-data."
        )

    cached = _PARSED.get(str(details.resolve()))
    if cached is not None:
        version, entries, listed = cached
    else:
        entries = {}
        for path in sorted(details.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CorpusError(f"{path} could not be read: {error}") from error
            if payload.get("isDeprecatedLicenseId"):
                continue
            grams = _ngrams(_normalize(payload.get("licenseText", "")))
            if grams:
                entries[payload["licenseId"]] = Reference(
                    name=payload.get("name", payload["licenseId"]),
                    grams=frozenset(grams),
                    text=payload.get("licenseText", ""),
                )

        if not entries:
            raise CorpusError(f"no usable license texts under {details}")

        index = json.loads(index_file.read_text(encoding="utf-8"))
        version = index.get("licenseListVersion", "unknown")
        exceptions = json_dir / "exceptions.json"
        try:
            exception_ids = frozenset(
                item["licenseExceptionId"]
                for item in json.loads(exceptions.read_text(encoding="utf-8"))[
                    "exceptions"
                ]
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise CorpusError(
                f"{exceptions} could not be read: {error}. Re-configure the build "
                "so deps/third_party fetches it."
            ) from error
        texts: dict[str, str] = {}
        for path in sorted((json_dir / "exceptions").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CorpusError(f"{path} could not be read: {error}") from error
            if payload.get("licenseExceptionText"):
                texts[payload["licenseExceptionId"]] = payload["licenseExceptionText"]
        listed = _Listed(
            frozenset(item["licenseId"] for item in index["licenses"]),
            exception_ids,
            texts,
        )
        _PARSED[str(details.resolve())] = (version, entries, listed)

    _CORPUS, _LISTED = (version, entries), listed
    _CORPUS_META.clear()
    _CORPUS_META.update(
        {
            "license_list_version": version,
            "source": str(json_dir),
            "commit": _checkout_commit(json_dir.parent),
            "licenses": len(entries),
        }
    )
    return version


def _checkout_commit(source_dir: Path) -> str:
    """The commit deps/third_party pinned, read back from the checkout itself."""

    from .discovery import own_git

    return own_git(source_dir, "rev-parse", "HEAD") or "NOASSERTION"


def corpus() -> tuple[str, dict[str, Reference]]:
    """Return (license-list version, references indexed by identifier)."""
    if _CORPUS is None:
        raise CorpusError(
            "SPDX license list data has not been loaded; it is fetched by the build "
            "into _deps/license-list-data-src (see deps/third_party)"
        )
    return _CORPUS


def identify(text: str) -> list[tuple[str, float, float]]:
    """Return (id, canonical-text containment, input-text coverage) matches.

    Both thresholds must pass. Keep distinct matches after family resolution;
    do not prune by length, which can miss matches as acceptance thresholds change.
    """
    candidate = _ngrams(_normalize(text))
    if not candidate:
        return []

    entries = corpus()[1]
    scored: dict[str, tuple[float, float, float]] = {}
    for license_id, reference in entries.items():
        overlap = len(reference.grams & candidate)
        if not overlap:
            continue
        containment = overlap / reference.size
        coverage = overlap / len(candidate)
        if containment >= _CONTAINMENT_THRESHOLD and coverage >= _COVERAGE_THRESHOLD:
            scored[license_id] = (
                containment,
                coverage,
                overlap / len(reference.grams | candidate),
            )

    dominated = _dominated(scored, entries)
    return sorted(
        (license_id, containment, coverage)
        for license_id, (containment, coverage, _) in scored.items()
        if license_id not in dominated
    )


def _dominated(
    scored: dict[str, tuple[float, float, float]],
    entries: dict[str, Reference],
) -> set[str]:
    """Keep the highest Jaccard match within near-identical license families.

    Length alone would favor ECL-2.0 over Apache-2.0, or CC-BY-NC over CC-BY,
    even when the input contains no additional terms.
    """
    dropped: set[str] = set()
    for left in scored:
        for right in scored:
            if left == right or right in dropped:
                continue
            if _family(entries, left, right) and scored[right][2] > scored[left][2]:
                dropped.add(left)
                break
    return dropped


def _family(entries: dict[str, Reference], left: str, right: str) -> bool:
    if left not in entries or right not in entries:
        return False
    shared = len(entries[left].grams & entries[right].grams)
    return shared / min(entries[left].size, entries[right].size) >= _FAMILY_THRESHOLD


def alternatives(matches) -> list[tuple[str, ...]]:
    """Group tied matches that their texts cannot distinguish.

    For example, GPL-3.0-only and GPL-3.0-or-later share a reference text.
    """
    entries = corpus()[1]
    groups: list[list[str]] = []
    for license_id, _, _ in matches:
        group = next(
            (
                group
                for group in groups
                if any(_family(entries, license_id, other) for other in group)
            ),
            None,
        )
        if group is None:
            groups.append([license_id])
        else:
            group.append(license_id)
    return [tuple(group) for group in groups]


def _name_for(group: tuple[str, ...]) -> str | None:
    """Select an unsuffixed base ID when every alternative extends it.

    MPL/OFL variants need a selecting notice; GPL -only/-or-later has no base
    ID, so choosing either from the text alone would assert an unsupported grant.
    """
    plain = [
        base
        for base in group
        if all(other == base or other.startswith(f"{base}-") for other in group)
    ]
    return plain[0] if len(plain) == 1 else None


def _either(names) -> str:
    return "matches the reference text of " + " or ".join(names)


def matched_prose(matches, name=lambda match: match[0], join=" AND ", lead=""):
    """Render matches with tied alternatives as prose, not a license conjunction.

    Separate alternative clauses with semicolons so AND cannot join them.
    """
    if not matches:
        return None
    by_id = {match[0]: match for match in matches}
    groups = [(group, _name_for(group)) for group in alternatives(matches)]
    single = [name(by_id[plain]) for _, plain in groups if plain]
    prose = [
        _either(name(by_id[item]) for item in group)
        for group, plain in groups
        if not plain
    ]
    parts = ([lead + join.join(single)] if single else []) + prose
    return ("; " if prose else join).join(parts)


def _read(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip() or b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", "replace")


SPDX_REFERENCE = "spdx-reference"


def evidence_for(
    component: str, origin: str, display: str, text: str, kind: str
) -> LicenseEvidence:
    # Do not score reference texts against their own corpus. Nested matches
    # identify license-author notices; they do not conclude the component license.
    matchable = kind in ("grant", "nested") and origin != SPDX_REFERENCE
    tag = _SPDX_TAG.search(text[:4096]) if matchable else None
    matches = tuple(identify(text)) if matchable else ()
    identified = matched_prose(matches)
    if origin == SPDX_REFERENCE:
        identified = Path(display).stem
    return LicenseEvidence(
        component=component,
        origin=origin,
        path=display,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        size=len(text.encode("utf-8")),
        text=text,
        kind=kind,
        identified=identified,
        matches=matches,
        spdx_tag=tag.group("expression").strip() if tag else None,
    )


def files_in(directory: Path) -> Callable[[str], bool]:
    return lambda name: (directory / name).is_file()


def is_sidecar(name: str, beside: Callable[[str], bool]) -> bool:
    """A .license file is a sidecar only when its annotated file exists beside it."""
    annotated = name.removesuffix(".license")
    return annotated not in {name, ""} and beside(annotated)


def classify(name: str, beside: Callable[[str], bool]) -> str | None:
    if Path(name).suffix.lower() in _SKIP_SUFFIXES or is_sidecar(name, beside):
        return None
    if _GRANT_NAMES.search(name):
        return "grant"
    if _NOTICE_NAMES.search(name):
        return "notice"
    return None


def discover_in_tree(
    component: str, root: Path, display_prefix: str
) -> list[LicenseEvidence]:
    """Collect root grants/notices, reference pools, and scoped nested terms."""
    found: list[LicenseEvidence] = []
    candidates: list[tuple[Path, str]] = []

    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        if entry.is_file() and (kind := classify(entry.name, files_in(root))):
            candidates.append((entry, kind))
        elif entry.is_dir() and entry.name.upper() in {"LICENSES", "LICENSE"}:
            candidates.extend(
                (item, "pool") for item in sorted(entry.rglob("*")) if item.is_file()
            )

    # Nested terms apply only to reached files (build._with_reached_nested);
    # they neither satisfy the root grant requirement nor set the component license.
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if (
            path.is_file()
            and path.parent != root
            and path.parent.name.upper() not in {"LICENSES", "LICENSE"}
            and (
                classify(path.name, files_in(path.parent))
                or _NOTICE_NAMES.match(path.name)
                or is_sidecar(path.name, files_in(path.parent))
            )
        ):
            candidates.append((path, "nested"))

    for path, kind in candidates:
        text = _read(path)
        if text is None:
            continue
        display = f"{display_prefix}/{path.relative_to(root).as_posix()}"
        item = evidence_for(component, "component-file", display, text, kind)
        if kind == "nested":
            covered = (
                path.with_name(path.name.removesuffix(".license"))
                if is_sidecar(path.name, files_in(path.parent))
                else path.parent
            )
            scope = f"{display_prefix}/{covered.relative_to(root).as_posix()}"
            item = replace(item, scope=scope)
        found.append(item)

    if not grant_texts(found):
        found.extend(_readme_fallback(component, root, display_prefix))
    return found


def _readme_fallback(
    component: str, root: Path, display_prefix: str
) -> list[LicenseEvidence]:
    """Some upstreams state the grant in a README section and ship no license file."""
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        if (
            not entry.is_file()
            or not _README_NAMES.match(entry.name)
            or is_sidecar(entry.name, files_in(root))
        ):
            continue
        text = _read(entry)
        if text is None:
            continue
        section = _README_LICENSE_SECTION.search(text)
        if section is None or not section.group("body").strip():
            continue
        # Hash and package only the licensing section. The path fragment makes its
        # digest reproducible and keeps unrelated README prose out of the terms.
        display = f"{display_prefix}/{entry.name}#license-section"
        return [
            evidence_for(
                component, "readme-section", display, section.group(0), "grant"
            )
        ]
    return []


NOASSERTION = "NOASSERTION"
_ID_SAFE = re.compile(r"[^A-Za-z0-9.-]+")


def spdx_safe(value: str) -> str:
    """Reduce a string to what an SPDX identifier may contain."""
    return _ID_SAFE.sub("-", value).strip("-")


def license_ref(component_key: str) -> str:
    """Identifier for a grant that is real but matches no SPDX reference text."""
    return f"LicenseRef-{spdx_safe(component_key)}"


def expressible(license_id: str) -> bool:
    """Check identifiers against the pinned expression grammar.

    The fetched SPDX list may be newer than license-expression's bundled list;
    newer IDs must travel as LicenseRefs until the grammar supports them.
    """
    if license_id.startswith("LicenseRef-"):
        return True
    try:
        return not _spdx_licensing().validate(license_id).errors
    except (ExpressionError, ValueError, TypeError):
        return False


def grammar_names(identifiers) -> bool:
    return all(expressible(item) or is_exception(item) for item in identifiers)


def _explained(text: str, groups) -> float:
    """How much of a text its matched `alternatives` groups account for, each
    counted once and the text's copyright notices left out."""
    candidate = _ngrams(_normalize(_without_notices(text)))
    if not candidate:
        return 1.0
    entries = corpus()[1]
    return sum(
        max(len(entries[key].grams & candidate) for key in group) / len(candidate)
        for group in groups
    )


def _without_notices(text: str) -> str:
    notices = [_normalize(notice) for notice in read_notices(text)]
    return "\n".join(
        line
        for line in text.splitlines()
        if not (words := _normalize(line))
        or not any(
            notice[index : index + len(words)] == words
            for notice in notices
            for index in range(len(notice) - len(words) + 1)
        )
    )


_NEWER = "newer than the SPDX grammar this document ships with"


def why_unnamed(evidence) -> str:
    """Why a component's carried grants travel verbatim, as lower-case clauses."""
    carried = _reading(evidence).carried
    reasons = []
    if any(grant.unmatched for grant in carried):
        reasons.append("matched no text in the SPDX reference corpus")
    reasons += [
        f"{_either(group)}, and the text alone does not say which applies"
        for group in sorted({group for grant in carried for group in grant.tied})
    ]
    reasons += [
        # Round down so a below-threshold result cannot display as meeting the threshold.
        "what matched in the SPDX reference corpus accounts for only "
        f"{math.floor(grant.explained * 100)}% of {grant.text.path}, its "
        "copyright notices aside"
        for grant in carried
        if grant.partial
    ]
    declared = sorted(
        {key for grant in carried if grant.declared for key in grant.newer}
    )
    matched = sorted(
        {key for grant in carried if not grant.declared for key in grant.newer}
        - set(declared)
    )
    if declared:
        reasons.append(f"declares {', '.join(declared)}, {_NEWER}")
    if matched:
        reasons.append(f"matched {', '.join(matched)}, {_NEWER}")
    return "; ".join(reasons)


def why_unresolved(evidence) -> str:
    """Explain NOASSERTION: either a reference pool without a grant, or
    grants whose identifiers could not form an expression.
    """
    if grants := grant_texts(evidence):
        paths = ", ".join(sorted(item.path for item in grants))
        return f"its grants ({paths}) yield no SPDX expression"
    pooled = pool_names(item.path for item in pool_texts(evidence))
    return (
        f"only REUSE pool texts ship ({pooled}); no grant states which of them "
        "are the component's terms"
    )


def _listed() -> _Listed:
    if _LISTED is None:
        raise CorpusError("SPDX license list data has not been loaded")
    return _LISTED


def listed_exception(identifier: str) -> bool:
    return identifier in _listed().exceptions


def exception_evidence(component_key: str, expression: str) -> list[LicenseEvidence]:
    """The SPDX reference text for each exception an expression names."""
    texts = _listed().exception_texts
    found = []
    for key in sorted(set(identifiers_in(expression)) & _listed().exceptions):
        if key not in texts:
            raise CorpusError(
                f"json/exceptions/{key}.json is missing from the SPDX license list "
                "data; re-configure the build so deps/third_party fetches it."
            )
        found.append(
            evidence_for(
                component_key, SPDX_REFERENCE, f"{key}.txt", texts[key], "grant"
            )
        )
    return found


def why_unlicensed(expressions) -> str | None:
    """Explain a vendored file's REUSE declaration failure, if any.

    Every license operand needs its own reference text: MIT cannot supply the
    terms for an adjacent LicenseRef. Exceptions need a listed exception ID.
    """
    entries = corpus()[1]
    licences: set[str] = set()
    exceptions: set[str] = set()
    for item in expressions:
        try:
            parsed = (
                _spdx_licensing().parse(str(item), validate=False)
                if item.is_valid
                else None
            )
        except (ExpressionError, ValueError, TypeError):
            parsed = None
        if parsed is None:
            return f"{str(item)!r} does not parse"
        for symbol in _spdx_licensing().license_symbols(parsed, decompose=False):
            if isinstance(symbol, LicenseWithExceptionSymbol):
                licences.add(symbol.license_symbol.key)
                exceptions.add(symbol.exception_symbol.key)
            else:
                licences.add(symbol.key)
    listed = _listed()
    # `+` is SPDX's or-later operator, not part of the id. Each slot is checked
    # against its own list: after WITH, SPDX takes only a listed exception.
    for ids, one, many in (
        (
            [key for key in licences if key in listed.exceptions],
            "is an SPDX exception, not a licence",
            "are SPDX exceptions, not licences",
        ),
        (
            [
                key
                for key in licences
                if key.removesuffix("+") not in listed.licences
                and key not in listed.exceptions
                and not key.startswith(("LicenseRef-", "DocumentRef-"))
            ],
            "is not an SPDX identifier",
            "are not SPDX identifiers",
        ),
        (
            [key for key in exceptions if key not in listed.exceptions],
            "is not an SPDX exception",
            "are not SPDX exceptions",
        ),
    ):
        if ids:
            return f"{', '.join(sorted(ids))} " + (one if len(ids) == 1 else many)
    textless = sorted(
        key for key in licences if key not in entries or not entries[key].text
    )
    if textless or not licences:
        return "no current SPDX-listed licence with a reference text" + (
            f" for {', '.join(textless)}" if textless else ""
        )
    return None


@dataclass(frozen=True)
class _Grant:
    """A grant's supported/newer IDs, tied alternatives, and text coverage."""

    text: LicenseEvidence
    named: tuple[str, ...] = ()
    newer: tuple[str, ...] = ()
    tied: tuple[tuple[str, ...], ...] = ()
    explained: float = 0.0

    @property
    def declared(self) -> bool:
        """A reference text for an id REUSE declares, not the component's own."""
        return self.text.origin == SPDX_REFERENCE

    @property
    def unmatched(self) -> bool:
        return not self.declared and not self.text.matches

    @property
    def partial(self) -> bool:
        return bool(self.text.matches) and self.explained < _EXPLAINED_THRESHOLD


@dataclass(frozen=True)
class _Reading:
    named: list[str]
    carried: list[_Grant]
    # Carried grants whose ids the conclusion names too.
    also_named: list[LicenseEvidence]


def _read_grant(item: LicenseEvidence) -> _Grant:
    if item.origin == SPDX_REFERENCE:
        # The declared ID identifies this text; matching it to its own corpus is circular.
        ids, tied, explained = [item.identified] if item.identified else [], (), 1.0
    else:
        groups = alternatives(item.matches)
        names = {group: _name_for(group) for group in groups}
        ids = [plain for plain in names.values() if plain]
        tied = tuple(group for group, plain in names.items() if not plain)
        explained = _explained(item.text, groups)
    return _Grant(
        item,
        tuple(key for key in ids if grammar_names([key])),
        tuple(key for key in ids if not grammar_names([key])),
        tied,
        explained,
    )


def _reading(evidence) -> _Reading:
    """Separate named grants from terms that must travel verbatim.

    If a REUSE declaration contains an unsupported ID, carry all of its texts
    together to preserve OR/WITH semantics.
    """
    read = [_read_grant(item) for item in grant_texts(evidence)]
    whole = any(grant.declared and grant.newer for grant in read)
    carried = [
        grant
        for grant in read
        if (whole and grant.declared)
        or (
            not grant.declared
            and (grant.newer or grant.tied or grant.unmatched or grant.partial)
        )
    ]

    def names(grant: _Grant) -> tuple[str, ...]:
        return () if whole and grant.declared else grant.named

    return _Reading(
        sorted({key for grant in read for key in names(grant)}),
        carried,
        [grant.text for grant in carried if names(grant)],
    )


def declared_terms(
    component_key: str, expressions
) -> tuple[str, list[LicenseEvidence]]:
    """Preserve REUSE expressions and supply their reference texts.

    Reconstructing from texts would turn OR into AND. Unsupported declarations
    travel whole as a LicenseRef, including exception texts.
    """
    declared = combine(sorted(expressions))
    evidence = canonical_evidence(component_key, set(expressions))
    if declared == NOASSERTION:
        return declared, evidence
    exceptions = exception_evidence(component_key, declared)
    if _reading(evidence + exceptions).carried:
        return license_ref(component_key), evidence + exceptions
    return declared, evidence


def expression(evidence: list[LicenseEvidence], component_key: str) -> tuple[str, str]:
    """Conclude matched IDs plus a LicenseRef for any unresolved grant text."""
    reading = _reading(evidence)
    identified = reading.named + (
        [license_ref(component_key)] if reading.carried else []
    )
    concluded = _combine(identified) if identified else NOASSERTION

    # Declared and concluded both derive from the component's grant text.
    # Do not use its REUSE tag: OpenXR's COPYING.adoc is tagged CC-BY-4.0
    # for the documentation, not for the SDK it describes.
    return concluded, concluded


def _combine(expressions: list[str]) -> str:
    """Join with AND through the SPDX grammar rather than by string join."""
    try:
        combined = combine_expressions(
            sorted(expressions), relation="AND", licensing=_spdx_licensing()
        )
    except (ExpressionError, ValueError, TypeError):
        return "NOASSERTION"
    # `is None`, not truthiness: a parsed expression raises on __bool__.
    return "NOASSERTION" if combined is None else str(combined)


def normalized_expression(expression: str) -> str:
    """Normalize an SPDX expression, returning NOASSERTION when parsing fails."""
    try:
        return str(_spdx_licensing().parse(expression, validate=False))
    except (ExpressionError, ValueError, TypeError):
        return "NOASSERTION"


def read_spdx_tag(text: str) -> str | None:
    """The REUSE tag a file states about itself, if any."""
    match = _SPDX_TAG.search(text)
    return match.group("expression").strip() if match else None


_NOTICE_YEARS = re.compile(r"\b(\d{4})(?:\s*[-\u2013]\s*(\d{4}))?\b")


def fold_notices(notices) -> list[str]:
    """Remove notices subsumed by another notice for the same holder.

    Do not merge disjoint year ranges: that would claim years no input states.
    """
    parsed = []
    for notice in notices:
        years: set[int] = set()
        for start, end in _NOTICE_YEARS.findall(notice):
            years.update(range(int(start), int(end or start) + 1))
        holder = re.sub(r"[^A-Za-z0-9]+", " ", _NOTICE_YEARS.sub("", notice))
        parsed.append((holder.strip().lower(), frozenset(years), notice))

    return sorted(
        {
            notice
            for holder, years, notice in parsed
            if not any(
                other == holder and years < covered for other, covered, _ in parsed
            )
        }
    )


# Require a copyright marker or year, allowing an unbounded prose lead-in
# such as "Box collision code (...) is Copyright 2016 ...".
_PLAIN_COPYRIGHT = re.compile(
    r"^.*?(?P<notice>Copyright\b\s*(?:\(c\)|©|\d{4})[^\n\r]*)$",
    re.MULTILINE | re.IGNORECASE,
)
# Connectives that indicate a notice was cut off mid-clause.
_DANGLING_WORDS = frozenset(
    {"and", "or", "is", "was", "are", "were", "by", "under", "the", "a"}
)
# Distinguish explanatory prose from holder names in Debian Copyright fields.
_PROSE = re.compile(
    r"\b(derived|contributed|reserved|portions?|redistribut\w+|written|"
    r"modified|see|provided|permission|granted|warrant\w*|software|hereby|"
    r"this|these|those|some|all rights)\b",
    re.IGNORECASE,
)
_YEARS_AND_MARKS = re.compile(r"\(c\)|©|\d{4}(?:\s*-\s*\d{2,4})?|[,;]", re.I)
# Lines of a licence body that mention copyright without stating one.
_NOT_A_NOTICE = re.compile(
    r"\bshall be\b|\bowner or entity\b|\bbe liable\b|\bmeans?\b|"
    r"\bsubject to\b|\bdefined as\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(
    r"\[yyyy\]|<year>|\[name of copyright owner\]|<copyright", re.I
)


def normalized_tag(text: str) -> str | None:
    """A file's own REUSE tag, normalised, or None if it names nothing usable."""
    tag = read_spdx_tag(text)
    if tag is None:
        return None
    expression = normalized_expression(tag)
    # A LicenseRef here has no text to define it with, as for any other file.
    if expression == NOASSERTION or license_refs(expression):
        return None
    # Parsing without validation accepts unknown IDs; validate before emitting.
    if not all(map(expressible, _spdx_licensing().license_keys(expression))):
        return None
    return expression


def notice_is_the_licence_authors(notice: str, identified: list[str]) -> bool:
    """Identify notices belonging to the license author, not the component.

    Compare against the corpus rather than maintaining a steward list
    (for example FSF in GPL, or Sam Hocevar in WTFPL).
    """
    entries = corpus()[1]
    collapsed = _steward_form(notice)
    return any(
        (reference := entries.get(license_id))
        and collapsed in _steward_form(reference.text)
        for license_id in identified
    )


def _steward_form(text: str) -> str:
    """Folded the way copies of one steward's line differ: the GPL's reads
    `(C)` and `http://` as upstreams ship it, `©` and `https://` in the list."""
    folded = " ".join(text.split()).lower().replace("©", "(c)")
    return folded.replace("https://", "http://")


def read_notices(text: str) -> list[str]:
    """Read all concrete copyright notices, excluding unfilled reference templates."""
    lines = text.splitlines()
    found: list[str] = _debian_copyright_fields(lines)
    for index, line in enumerate(lines):
        match = _PLAIN_COPYRIGHT.match(line)
        if not match:
            continue
        notice = " ".join(match.group("notice").split()).rstrip("*/ ")
        # Holders may continue below a year-only line or trailing comma (Debian, qhull, libccd).
        step = index
        while step + 1 < len(lines) and step - index < 3:
            if not (notice.endswith(",") or not _names_a_holder(notice)):
                break
            ahead = step + 1
            # One blank line may sit between the years and the holder; qhull
            # writes `Qhull, Copyright (c) 1993-2020`, a blank, then the holder.
            if not lines[ahead].strip() and not _names_a_holder(notice):
                ahead += 1
            if ahead >= len(lines):
                break
            following = " ".join(lines[ahead].split())
            # Stop before a new notice, field, or license clause is mistaken for a holder.
            if (
                not following
                or _PLAIN_COPYRIGHT.match(following)
                or ":" in following
                or _PROSE.search(following)
                or _NOT_A_NOTICE.search(following)
            ):
                break
            step = ahead
            notice = f"{notice} {following}".strip().rstrip("*/ ")
        notice = _without_dangling(notice)
        if _PLACEHOLDER.search(notice) or _NOT_A_NOTICE.search(notice):
            continue
        if not _names_a_holder(notice) and not re.search(r"\d{4}", notice):
            continue
        if notice not in found:
            found.append(notice)
    return found


_NO_HOLDER = frozenset({"none", "n/a", "unknown"})


def _copyright_fields(lines: list[str]) -> list[list[list[str]]]:
    """Each Debian `Copyright:` field's paragraphs, each a list of its lines.

    The field runs on over indented lines and ` .` stands for a blank one,
    which separates paragraphs. Debian writes `None` where a package states no
    holder, and what follows that is commentary, not a holder.
    """
    fields: list[list[list[str]]] = []
    for index, line in enumerate(lines):
        if not line.lower().startswith("copyright:"):
            continue
        raw = [line[len("copyright:") :]]
        for following in lines[index + 1 :]:
            if not following[:1].isspace() or not following.strip():
                break
            raw.append(following)
        paragraphs: list[list[str]] = [[]]
        for value in (" ".join(item.split()) for item in raw):
            if value.lower() in _NO_HOLDER:
                break
            if value == ".":
                paragraphs.append([])
            elif value:
                paragraphs[-1].append(value)
        fields.append([paragraph for paragraph in paragraphs if paragraph])
    return fields


def _debian_copyright_fields(lines: list[str]) -> list[str]:
    """Read bare holders from Debian Copyright fields without requiring the word Copyright."""
    found: list[str] = []
    for field in _copyright_fields(lines):
        values = [value for paragraph in field for value in paragraph]
        for position, value in enumerate(values):
            holder = value.rstrip(".")
            # Full notices are handled by read_notices; stop here when bare holders end.
            if _PLAIN_COPYRIGHT.match(holder) or _PROSE.search(holder):
                break
            if position and len(holder) > 60:
                break
            if _names_a_holder(f"Copyright {holder}") and holder not in found:
                found.append(holder)
    return found


def dep5_copyright(text: str) -> list[str] | None:
    """Read DEP5 Copyright fields, or return None without an opening Format field.

    Ignore quoted notices in license prose and packaging-only stanzas.
    """
    lines = text.splitlines()
    first = next((line for line in lines if line.strip()), "")
    if not first.lower().startswith("format:"):
        return None
    return [
        notice
        for stanza in _stanzas(lines)
        if not _packaging_only(stanza)
        for field in _copyright_fields(stanza)
        for paragraph in field
        for notice in _notices_in(paragraph)
    ]


_OPENS_NOTICE = re.compile(r"(?:copyright\b|©|\(c\))", re.IGNORECASE)
_LEADING_YEAR = re.compile(r"\d{4}")
# Markers, years and punctuation only: a notice that names no holder yet.
_NO_HOLDER_YET = re.compile(r"^(?:copyright\b|\(c\)|©|\d{4}|[\s,\-–;/.])*$", re.I)


def _notices_in(paragraph: list[str]) -> list[str]:
    """A field paragraph's lines, grouped into the notices they state.

    `Copyright`, © or (c) starts a notice, and so does a year -- unless the
    notice above names no holder yet, or is a sentence left unfinished -- and
    a bare name after a bare name. Any other line continues the notice above.
    """

    def marked(text: str) -> bool:
        return bool(_OPENS_NOTICE.match(text) or _LEADING_YEAR.match(text))

    def bare(text: str) -> bool:
        return not ("\n" in text or marked(text) or _PROSE.search(text))

    def unfinished(current: str) -> bool:
        first = current.split("\n", 1)[0]
        return (
            not marked(first)
            and bool(_PROSE.search(first))
            and not current.endswith(".")
        )

    def opens(line: str, current: str) -> bool:
        if _OPENS_NOTICE.match(line):
            return True
        if _LEADING_YEAR.match(line):
            return not (_NO_HOLDER_YET.match(current) or unfinished(current))
        if current.endswith(",") and line[:1].islower():
            return False
        return bare(current) and bare(line)

    notices: list[str] = []
    for line in paragraph:
        if not notices or opens(line, notices[-1]):
            notices.append(line)
        else:
            notices[-1] += "\n" + line
    return notices


def _stanzas(lines: list[str]) -> list[list[str]]:
    stanzas: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            stanzas[-1].append(line)
        elif stanzas[-1]:
            stanzas.append([])
    return [stanza for stanza in stanzas if stanza]


def _packaging_only(stanza: list[str]) -> bool:
    """Identify stanzas limited to debian/; their holders are packagers.

    A header without Files is not packaging-only.
    """
    patterns: list[str] = []
    for index, line in enumerate(stanza):
        if line.lower().startswith("files:"):
            patterns += line[len("files:") :].split()
            for following in stanza[index + 1 :]:
                if not following[:1].isspace():
                    break
                patterns += following.split()
    patterns = [item.removeprefix("./") for item in patterns]
    return bool(patterns) and all(
        item == "debian" or item.startswith("debian/") for item in patterns
    )


def _without_dangling(notice: str) -> str:
    """Drop the connectives a notice ends on because its line broke mid-clause.

    By token, not by regex: a repeated or nested match over the whole string is
    quadratic on a long line that ends in many of them.
    """
    words = notice.split()
    while words and words[-1].rstrip(",;").lower() in _DANGLING_WORDS:
        words.pop()
    return " ".join(words).rstrip("*/,; ")


def _names_a_holder(notice: str) -> bool:
    rest = _YEARS_AND_MARKS.sub(" ", notice[len("Copyright") :])
    return bool(re.search(r"[A-Za-z]{2}", rest))


def read_copyright(text: str) -> str | None:
    """The notice a file states about itself, beside its REUSE tag."""
    match = _FILE_COPYRIGHT.search(text)
    return " ".join(match.group("notice").split()).rstrip("*/ ") if match else None


def combine(expressions: list[str]) -> str:
    return _combine(expressions)


def canonical_evidence(
    component_key: str, expressions: set[str]
) -> list[LicenseEvidence]:
    """Fetch reference texts for vendored declarations from the pinned SPDX list.

    Do not use the repository's REUSE pool: its texts can carry our own notices.
    """
    found: list[LicenseEvidence] = []
    seen: set[str] = set()
    entries = corpus()[1]
    for expression in sorted(expressions):
        try:
            parsed = _spdx_licensing().parse(expression, validate=False)
        except (ExpressionError, ValueError, TypeError):
            # The per-file build gate reports malformed declarations with their source paths.
            continue
        for token in _spdx_licensing().license_keys(parsed):
            reference = entries.get(token)
            if token in seen or reference is None or not reference.text:
                continue
            seen.add(token)
            found.append(
                evidence_for(
                    component_key,
                    SPDX_REFERENCE,
                    f"{token}.txt",
                    reference.text,
                    "grant",
                )
            )
    return found


class ExpressionReadError(Exception):
    """An expression in a document cannot be read, so what it names is unknown."""


def shared_refs(components) -> dict[str, str]:
    """Share a LicenseRef among components with byte-identical terms.

    Resolve IDs before rendering so notices, reports, and SPDX agree.
    """
    by_text: dict[str, list[str]] = {}
    for component in components:
        terms = verbatim_terms(component.evidence)
        if terms:
            by_text.setdefault(terms, []).append(component.key)

    renamed: dict[str, str] = {}
    for text, keys in by_text.items():
        if len(keys) < 2:
            continue
        canonical = shared_ref(text)
        for key in keys:
            renamed[license_ref(key)] = canonical
    return renamed


SHARED_REF = "LicenseRef-shared-text-"


def shared_ref(text: str) -> str:
    return SHARED_REF + sha256_text(text)[:12]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rename_refs(expression: str, renamed: dict[str, str]) -> str:
    """Replace whole identifiers: LicenseRef-alpha must not alter LicenseRef-alpha-extra."""
    for old_id, new_id in renamed.items():
        expression = re.sub(
            rf"(?<![\w.-]){re.escape(old_id)}(?![\w.-])", new_id, expression
        )
    return expression


def is_exception(identifier: str) -> bool:
    """Check whether an identifier names an SPDX license exception."""
    try:
        parsed = _spdx_licensing().parse(identifier, validate=False)
    except (ExpressionError, ValueError, TypeError):
        return False
    return identifier in {str(item) for item in getattr(parsed, "symbols", ())} and any(
        getattr(item, "is_exception", False) for item in parsed.symbols
    )


def operands_of(expression: str) -> list[str]:
    """Split expressions into operands that can be AND-joined without changing meaning.

    Keep OR choices whole; splitting them would require both alternatives.
    """
    try:
        parsed = _spdx_licensing().parse(expression, validate=False)
    except (ExpressionError, ValueError, TypeError):
        return [expression]
    if " OR " in str(parsed).upper():
        return [f"({parsed})"]
    return sorted(str(key) for key in _spdx_licensing().license_symbols(parsed))


def identifiers_in(expression: str) -> list[str]:
    """Every licence identifier an expression names, LicenseRef included."""
    try:
        parsed = _spdx_licensing().parse(expression, validate=False)
    except (ExpressionError, ValueError, TypeError) as error:
        raise ExpressionReadError(f"{expression!r}: {error}") from error
    return sorted(_spdx_licensing().license_keys(parsed))


def license_refs(*expressions: str) -> list[str]:
    """The LicenseRef- identifiers an expression uses, per the SPDX grammar."""
    found: set[str] = set()
    for expression in expressions:
        try:
            parsed = _spdx_licensing().parse(expression, validate=False)
        except (ExpressionError, ValueError, TypeError) as error:
            # Malformed expressions must fail closed; skipping one could hide undefined references.
            raise ExpressionReadError(f"{expression!r}: {error}") from error
        found.update(
            key
            for key in _spdx_licensing().license_keys(parsed)
            if key.startswith("LicenseRef-")
        )
    return sorted(found)


@lru_cache(maxsize=1)
def _spdx_licensing():
    """Building this parses ScanCode's licence database; do it once."""
    return get_spdx_licensing()
