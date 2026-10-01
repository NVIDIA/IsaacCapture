# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the license texts a component ships, and name what they are.

Nothing here declares a license. Texts are discovered in the component's own
files and matched against the SPDX reference corpus; a text that matches nothing
is still packaged and reported as unidentified, never guessed at.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from dataclasses import dataclass

from license_expression import (
    ExpressionError,
    combine_expressions,
    get_spdx_licensing,
)
from pathlib import Path


# Root-level names that carry a grant, and the ones that carry notices instead.
# Both are packaged; only the first kind is matched against the corpus.
# The word has to stand alone: BSD-LICENSE and LICENSE.md are grants, while
# licenses.md and accessingLicenses.md are indexes that merely mention them.
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
# ...and has to account for a real share of the file it was found in. Large
# agreements (the CUDA EULA, for one) reproduce whole OSS licenses in an
# appendix; those reach full containment at a few percent coverage, while a
# genuine match sits well above it -- a Debian copyright file naming several
# licenses around the one matched is the low end, at a quarter.
_COVERAGE_THRESHOLD = 0.20
# Two licenses are one family when either text is nearly inside the other.
_FAMILY_THRESHOLD = 0.90
# REUSE-IgnoreStart
_SPDX_TAG = re.compile(r"SPDX-License-Identifier:\s*(?P<expression>[^\n\r*/#]+)")
_FILE_COPYRIGHT = re.compile(r"SPDX-FileCopyrightText:\s*(?P<notice>[^\n\r]+)")
# REUSE-IgnoreEnd
# Heading, then everything up to the next heading or a blank-line run.
# The heading has to BE about licensing, not merely start with the word --
# "# License List Data" is a project title, not a grant.
_README_LICENSE_SECTION = re.compile(
    r"^#{1,6}\s*(licen[cs]e|licen[cs]es|licensing|licen[cs]e information|copyright|legal)"
    r"\s*$(?P<body>.*?)(?=^#{1,6}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class LicenseEvidence:
    """A license text that was read out of the component, not assumed."""

    component: str
    origin: str  # "component-file" | "archive-member" | "readme-section" | "build-host"
    path: str
    sha256: str
    size: int
    text: str
    # "grant" is the component's own license, "notice" an attribution file, and
    # "pool" a REUSE LICENSES/ entry that applies to individual files only.
    kind: str
    identified: str | None = None
    matches: tuple[tuple[str, float, float], ...] = ()
    spdx_tag: str | None = None

    def as_json(self) -> dict:
        return {
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


_DIGITS = re.compile(r"\d+")


def _normalize(text: str) -> list[str]:
    """Reduce a license to comparable boilerplate.

    Years and clause numbers are the part that differs between two copies of one
    license, so digits come out. Copyright lines stay: BSD clause 3 and MIT's
    notice-retention clause both contain the word, and upstream wrapping differs
    from SPDX's, so dropping those lines loses real text asymmetrically.
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
    """One SPDX licence, normalized for comparison and kept verbatim.

    The text is retained because it is the only authoritative copy of a licence
    this build has: a component that states an identifier and ships no file of
    its own has to be packaged with the text that identifier names.
    """

    name: str
    grams: frozenset[str]
    text: str = ""

    @property
    def size(self) -> int:
        return len(self.grams)


class CorpusError(Exception):
    """The SPDX reference texts this build fetched are missing or unreadable."""


_CORPUS: tuple[str, dict[str, Reference]] | None = None
_CORPUS_META: dict[str, str] = {}
# Parsing and n-gramming the whole SPDX list costs about half a second. Keyed by
# directory and only ever read within one process, so a re-load is free.
_PARSED: dict[str, tuple[str, dict[str, Reference]]] = {}


def corpus_provenance() -> dict[str, str]:
    """Which reference texts an identification was made against."""
    return dict(_CORPUS_META)


def load_corpus(json_dir: Path) -> str:
    """Load SPDX reference texts from the checkout deps/third_party fetched.

    `json_dir` is the `json/` directory of spdx/license-list-data. Deprecated
    identifiers are skipped: naming a component with one would be reporting a
    license SPDX has withdrawn.
    """
    global _CORPUS
    details = json_dir / "details"
    index_file = json_dir / "licenses.json"
    if not details.is_dir() or not index_file.is_file():
        raise CorpusError(
            f"SPDX license list data not found at {json_dir}. It is fetched by the "
            "build (deps/third_party); configure and build, or pass --license-data."
        )

    cached = _PARSED.get(str(details.resolve()))
    if cached is not None:
        version, entries = cached
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

        version = json.loads(index_file.read_text(encoding="utf-8")).get(
            "licenseListVersion", "unknown"
        )
        _PARSED[str(details.resolve())] = (version, entries)

    _CORPUS = (version, entries)
    _CORPUS_META.clear()
    _CORPUS_META.update(
        {
            "license_list_version": version,
            "source": str(json_dir),
            "commit": _checkout_commit(json_dir.parent),
            "licenses": str(len(entries)),
        }
    )
    return version


def _checkout_commit(source_dir: Path) -> str:
    """The commit deps/third_party pinned, read back from the checkout itself."""

    from .discovery import repo_head

    return repo_head(source_dir) or "NOASSERTION"


def corpus() -> tuple[str, dict[str, Reference]]:
    """SPDX reference texts as (list version, {id: (name, n-grams)})."""
    if _CORPUS is None:
        raise CorpusError(
            "SPDX license list data has not been loaded; it is fetched by the build "
            "into _deps/license-list-data-src (see deps/third_party)"
        )
    return _CORPUS


def identify(text: str) -> list[tuple[str, float, float]]:
    """Name the licenses a text contains, as (id, containment, coverage).

    Containment says how much of a canonical license is present; coverage says
    how much of this file that accounts for. Both must clear their threshold, so
    a license quoted in an appendix is not mistaken for the file's own terms. A
    file really can carry two licenses -- dual-licensed sources do -- so this
    returns every match that survives the family resolution below.

    Every reference is compared. The thresholds do bound how long a reference
    can be relative to the file, so matches could be skipped on length -- but
    that ties correctness to the accept conditions never growing, and it fails
    by quietly not identifying a license. Keep the comparison exhaustive.
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
    """Within a family of near-identical licenses, keep only the best fit.

    Whole families differ by one clause: BSD-2 inside BSD-3, Apache-2.0 inside
    ECL-2.0, CC-BY-4.0 inside CC-BY-NC-4.0. Every one of those clears the
    containment threshold against a file that is really the other, so the
    discriminator has to be which reference explains the whole file *and nothing
    more* -- the highest Jaccard similarity. Preferring the longer text instead
    would report Apache-2.0 code as ECL-2.0, and CC-BY-4.0 as its
    non-commercial variant.
    """
    dropped: set[str] = set()
    for left in scored:
        for right in scored:
            if left == right or right in dropped:
                continue
            shared = len(entries[left].grams & entries[right].grams)
            related = shared / min(entries[left].size, entries[right].size)
            if related >= _FAMILY_THRESHOLD and scored[right][2] > scored[left][2]:
                dropped.add(left)
                break
    return dropped


def _read(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip() or b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", "replace")


def _evidence(
    component: str, origin: str, display: str, text: str, kind: str
) -> LicenseEvidence:
    matchable = kind == "grant"
    tag = _SPDX_TAG.search(text[:4096]) if matchable else None
    matches = tuple(identify(text)) if matchable else ()
    identified = " AND ".join(license_id for license_id, _, _ in matches) or None
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


def _classify(name: str) -> str | None:
    if Path(name).suffix.lower() in _SKIP_SUFFIXES:
        return None
    if _GRANT_NAMES.search(name):
        return "grant"
    if _NOTICE_NAMES.search(name):
        return "notice"
    return None


def discover_in_tree(
    component: str, root: Path, display_prefix: str
) -> list[LicenseEvidence]:
    """Every license and notice file a source tree carries, at its root or in LICENSES/."""
    found: list[LicenseEvidence] = []
    candidates: list[tuple[Path, str]] = []

    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        if entry.is_file() and (kind := _classify(entry.name)):
            candidates.append((entry, kind))
        elif entry.is_dir() and entry.name.upper() in {"LICENSES", "LICENSE"}:
            candidates.extend(
                (item, "pool") for item in sorted(entry.rglob("*")) if item.is_file()
            )

    for path, kind in candidates:
        text = _read(path)
        if text is None:
            continue
        display = f"{display_prefix}/{path.relative_to(root).as_posix()}"
        found.append(_evidence(component, "component-file", display, text, kind))

    if not any(item.kind == "grant" for item in found):
        found.extend(_readme_fallback(component, root, display_prefix))
    return found


def _readme_fallback(
    component: str, root: Path, display_prefix: str
) -> list[LicenseEvidence]:
    """Some upstreams state the grant in a README section and ship no license file."""
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        if not entry.is_file() or not _README_NAMES.match(entry.name):
            continue
        text = _read(entry)
        if text is None:
            continue
        section = _README_LICENSE_SECTION.search(text)
        if section is None or not section.group("body").strip():
            continue
        # The section, not the whole README. Hashing the file would publish its
        # build instructions as the terms that shipped, and would bury a real
        # licence quoted there under enough prose to fall below the coverage
        # threshold and degrade to a LicenseRef.
        display = f"{display_prefix}/{entry.name}"
        return [
            _evidence(component, "readme-section", display, section.group(0), "grant")
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
    """Whether the SPDX grammar we ship can validate this identifier.

    The SPDX license list the build fetches moves ahead of the list bundled in
    `license-expression`, so the corpus can name an ID the grammar cannot parse.
    Matching against the newer corpus is right; naming an ID that a consumer's
    validator -- which ships the older list -- will reject is not.
    """
    if license_id.startswith("LicenseRef-"):
        return True
    try:
        return not _spdx_licensing().validate(license_id).errors
    except (ExpressionError, ValueError, TypeError):
        return False


def expression(
    evidence: list[LicenseEvidence], component_key: str | None = None
) -> tuple[str, str]:
    """Concluded and declared expressions for one component.

    Concluded names what was matched in a text that ships with the artifact. A
    grant that matches nothing is not dropped and not guessed at: it becomes a
    LicenseRef whose text travels in the document, which says "these are the
    terms we shipped" without claiming to know which license they are.

    An ID the shipped grammar cannot express is treated the same way, and the
    whole expression falls back rather than just the offending term: dropping
    one ID from `A AND B` would understate the terms.
    """
    grants = [item for item in evidence if item.kind == "grant"]
    identified = sorted(
        {license_id for item in grants for license_id, _, _ in item.matches}
    )
    if identified and component_key and not all(map(expressible, identified)):
        identified = [license_ref(component_key)]
    if not identified and grants and component_key:
        identified = [license_ref(component_key)]
    concluded = _combine(identified) if identified else "NOASSERTION"

    # A tag speaks for the package only where the file carrying it is a licence
    # in its own right. OpenXR's COPYING.adoc is prose about which licences the
    # project uses, tagged CC-BY-4.0 for the prose and matching no licence text;
    # promoting it declared a linked SDK under a documentation licence. Where no
    # tag qualifies the field stays unasserted rather than echoing the
    # conclusion, which would assert a declaration nobody made and hide the one
    # thing the field is for -- supplier and collector disagreeing.
    tags = sorted({item.spdx_tag for item in grants if item.spdx_tag and item.matches})
    declared = _combine(tags) if tags else NOASSERTION
    return concluded, declared


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
    """Put an expression through the SPDX grammar before it reaches a document.

    Upstream tags are written by hand and arrive with lowercase operators and
    odd spacing; emitting those unchanged would put an unparseable expression in
    an SBOM. Anything the grammar rejects is reported as NOASSERTION rather than
    passed through.
    """
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
    """Drop a notice another already covers.

    Files written in different years state one holder over different year ranges,
    and listing every one names that holder several times over. Merging the
    ranges instead would claim a year no file claims, so only a notice whose
    holder matches another's and whose years that other already contains is
    dropped -- it says nothing the one kept does not.
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


_PLAIN_COPYRIGHT = re.compile(
    r"^[^\w\n]*(?P<notice>Copyright\b[^\n\r]*\d{4}[^\n\r]*|"
    r"Copyright\b[^\n\r]*)$",
    re.MULTILINE | re.IGNORECASE,
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
    return expression


def read_notice(text: str) -> str | None:
    """A concrete copyright line stated in a licence text, if it carries one.

    A reference text carries the template instead -- `Copyright [yyyy] [name of
    copyright owner]` -- which names nobody, so it is not a notice and recording
    it would claim one where the file states none.
    """
    for match in _PLAIN_COPYRIGHT.finditer(text):
        notice = " ".join(match.group("notice").split()).rstrip("*/ ")
        if _PLACEHOLDER.search(notice) or not re.search(r"\d{4}", notice):
            continue
        return notice
    return None


def read_copyright(text: str) -> str | None:
    """The notice a file states about itself, beside its REUSE tag."""
    match = _FILE_COPYRIGHT.search(text)
    return " ".join(match.group("notice").split()).rstrip("*/ ") if match else None


def combine(expressions: list[str]) -> str:
    """Public form of the AND-combiner, for callers assembling an expression."""
    return _combine(expressions)


def canonical_evidence(
    component_key: str, expressions: set[str]
) -> list[LicenseEvidence]:
    """The SPDX reference text for the ids a vendored file declares about itself.

    Such a component ships no licence file, so there is nothing of its own to
    package verbatim and the text its identifier names is the only honest
    substitute. It comes from the list this build pinned and fetched -- never
    from this repository's own REUSE pool, whose copies exist to license *this*
    project and carry its notices, not the component's.
    """
    found: list[LicenseEvidence] = []
    seen: set[str] = set()
    entries = corpus()[1]
    for expression in sorted(expressions):
        for token in _spdx_licensing().license_keys(
            _spdx_licensing().parse(expression, validate=False)
        ):
            reference = entries.get(token)
            if token in seen or reference is None or not reference.text:
                continue
            seen.add(token)
            found.append(
                _evidence(
                    component_key,
                    "spdx-reference",
                    f"{token}.txt",
                    reference.text,
                    "grant",
                )
            )
    return found


def license_refs(*expressions: str) -> list[str]:
    """The LicenseRef- identifiers an expression uses, per the SPDX grammar."""
    found: set[str] = set()
    for expression in expressions:
        try:
            parsed = _spdx_licensing().parse(expression, validate=False)
        except (ExpressionError, ValueError, TypeError):
            continue
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
