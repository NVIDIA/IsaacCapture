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
# ...and the matches together have to account for most of the file before the
# set of them is called an identification. A Debian copyright file aggregates
# many stanzas, and generic BSD boilerplate clears containment against variants
# the file never mentions; two such matches covering half the text name the file
# no better than its own words do.
_EXPLAINED_THRESHOLD = 0.75
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
    # "grant" is the component's own license, "notice" an attribution file,
    # "pool" a REUSE LICENSES/ entry that applies to individual files only, and
    # "nested" terms stated for one subdirectory of a larger component.
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


# What counts as a text that discharges the packaging obligation. A REUSE pool
# names no single expression for the component and is still a real text, so it
# counts. The build gate, the document, the report and the verifier all ask this
# question; they ask it here.
OBLIGATION_KINDS = frozenset({"grant", "pool"})


def satisfies_obligation(kind: str) -> bool:
    """For callers holding the JSON form, where there is no dataclass to filter."""
    return kind in OBLIGATION_KINDS


def obligation_texts(evidence) -> list[LicenseEvidence]:
    return [item for item in evidence if satisfies_obligation(item.kind)]


def verbatim_terms(evidence) -> str:
    """The body a LicenseRef carries for a component: every text it read.

    One component can state its terms across several files, so this is a join
    and not a file. The build records its digest and the verifier compares
    against that, which only works while both sides join the same way -- so they
    both call this.
    """
    return "\n\n".join(item.text for item in obligation_texts(evidence))


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
_CORPUS_META: dict[str, object] = {}
# Parsing and n-gramming the whole SPDX list costs about half a second. Keyed by
# directory and only ever read within one process, so a re-load is free.
_PARSED: dict[str, tuple[str, dict[str, Reference]]] = {}


def corpus_provenance() -> dict[str, object]:
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
            "licenses": len(entries),
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

    Compare every reference; do not prune by length. Pruning ties correctness to
    the accept conditions never growing, and fails by not identifying a license.
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


SPDX_REFERENCE = "spdx-reference"


def evidence_for(
    component: str, origin: str, display: str, text: str, kind: str
) -> LicenseEvidence:
    # A reference text is not matched: scoring it against the corpus it came
    # from returns a perfect score, which reads as a measurement of the
    # component's own file and is a measurement of nothing.
    matchable = kind == "grant" and origin != SPDX_REFERENCE
    tag = _SPDX_TAG.search(text[:4096]) if matchable else None
    matches = tuple(identify(text)) if matchable else ()
    identified = " AND ".join(license_id for license_id, _, _ in matches) or None
    if origin == SPDX_REFERENCE:
        # Named by the identifier it was fetched for, with no score behind it.
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


def classify(name: str) -> str | None:
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
        if entry.is_file() and (kind := classify(entry.name)):
            candidates.append((entry, kind))
        elif entry.is_dir() and entry.name.upper() in {"LICENSES", "LICENSE"}:
            candidates.extend(
                (item, "pool") for item in sorted(entry.rglob("*")) if item.is_file()
            )

    # A subdirectory can carry its own terms and its own holder -- a vendored
    # asset set inside a larger project. Collected as "nested" so it neither
    # discharges the component's packaging obligation nor changes its licence,
    # and kept only where that directory reached the wheel; see `nested_for`.
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if (
            path.is_file()
            and path.parent != root
            and path.parent.name.upper() not in {"LICENSES", "LICENSE"}
            and (classify(path.name) or _NOTICE_NAMES.match(path.name))
        ):
            candidates.append((path, "nested"))

    for path, kind in candidates:
        text = _read(path)
        if text is None:
            continue
        display = f"{display_prefix}/{path.relative_to(root).as_posix()}"
        found.append(evidence_for(component, "component-file", display, text, kind))
    # "nested" never satisfies the grant test below; that is the point.

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
        # The fragment is load-bearing: the digest below is of the section, not
        # of the file, and a path naming the whole file beside it is a pair a
        # verifier cannot reproduce.
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


def explains_the_text(matches) -> bool:
    """Whether the matched licences account for most of what the file says.

    Not a test of whether an identification is right -- a dual-licence file
    matches correctly and scores low -- only of whether it explains the file,
    which is what the document has to say when it cannot name one.
    """
    return sum(coverage for _, _, coverage in matches) >= _EXPLAINED_THRESHOLD


def why_unnamed(grants: list[LicenseEvidence]) -> str:
    """Why terms travel verbatim rather than under an SPDX identifier."""
    if not any(item.matches for item in grants):
        return "Matched no text in the SPDX reference corpus"
    partial = [item for item in grants if not explains_the_text(item.matches)]
    if partial:
        share = max(
            sum(coverage for _, _, coverage in item.matches) for item in partial
        )
        return (
            "What matched in the SPDX reference corpus accounts for only "
            f"{share:.0%} of this text, so the matches name part of it and not "
            "the whole"
        )
    # Matched, but under an id the grammar shipped with this document cannot
    # validate, so it cannot be named here.
    return "Matched an identifier newer than the SPDX grammar this document ships with"


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
        # A reference text is named by the identifier it was fetched for; there
        # is no match to read, because matching it would be circular.
        | {
            item.identified
            for item in grants
            if item.origin == SPDX_REFERENCE and item.identified
        }
    )
    if identified and component_key and not all(map(expressible, identified)):
        identified = [license_ref(component_key)]
    if not identified and grants and component_key:
        identified = [license_ref(component_key)]
    concluded = _combine(identified) if identified else "NOASSERTION"

    # Declared is the same expression, because the text it was read from is the
    # component's own LICENSE -- shipping that file is how authors declare a
    # licence, so one artifact answers both questions. It is not an echo: a
    # component that ships no grant declares nothing and gets NOASSERTION.
    #
    # Never from a REUSE tag on a grant file. OpenXR's COPYING.adoc is prose
    # about which licences the project uses, tagged CC-BY-4.0 for the prose, and
    # promoting that declared a linked Apache-2.0 SDK under a documentation
    # licence.
    declared = concluded if grants else NOASSERTION
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


# A notice is `Copyright` plus a marker that prose does not carry: a (c), a ©,
# or a year. It need not start the line and the lead-in is not bounded: a NOTICE
# writes `Box collision code (engine_collision_box.c) is Copyright 2016 ...`.
_PLAIN_COPYRIGHT = re.compile(
    r"^.*?(?P<notice>Copyright\b\s*(?:\(c\)|©|\d{4})[^\n\r]*)$",
    re.MULTILINE | re.IGNORECASE,
)
# A clause broken by the end of a line, left dangling by taking the line alone.
_DANGLING = re.compile(r"(?:[,;]?\s+(?:and|or|is|was|are|were|by|under|the|a))+$", re.I)
# Years, year ranges and the markers around them: what is left is the holder.
# A sentence, as against a name: the Debian `Copyright:` field holds both.
_PROSE = re.compile(
    r"\b(derived|contributed|reserved|portions?|redistribut\w+|written|"
    r"modified|see|provided|this|these|some|all rights)\b",
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
    # And the grammar has to know every id in it. `normalized_expression` parses
    # without validating, so a tag naming nothing real reached the document and
    # failed its own conformance check -- the same reason a conclusion is tested.
    if not all(map(expressible, _spdx_licensing().license_keys(expression))):
        return None
    return expression


def notice_is_the_licence_authors(notice: str, identified: list[str]) -> bool:
    """Whether this notice belongs to the licence rather than to the component.

    A COPYING that *is* the reference text carries the steward's own line -- the
    FSF's in every GPL, Sam Hocevar's in WTFPL -- so a component shipping one
    verbatim would be credited to whoever wrote the licence. The corpus already
    holds those texts, so the question is answered by looking rather than by
    keeping a list of stewards.
    """
    entries = corpus()[1]
    collapsed = " ".join(notice.split())
    return any(
        (reference := entries.get(license_id))
        and collapsed in " ".join(reference.text.split())
        for license_id in identified
    )


def read_notices(text: str) -> list[str]:
    """Every concrete copyright notice a licence text states.

    All of them: glfw names two holders and miniz names two, and under MIT, BSD
    and Zlib the notice is the obligation, so keeping the first is a notice file
    that does not discharge it. A reference text carries the template instead --
    `Copyright [yyyy] [name of copyright owner]` -- which names nobody.
    """
    lines = text.splitlines()
    found: list[str] = _debian_copyright_fields(lines)
    for index, line in enumerate(lines):
        match = _PLAIN_COPYRIGHT.match(line)
        if not match:
            continue
        notice = " ".join(match.group("notice").split()).rstrip("*/ ")
        # The holder can sit on the lines below: after a trailing comma (libccd
        # names a department, a faculty and a university), or after nothing at
        # all (Debian copyright files and qhull put the years on one line and the
        # holder on the next). Keep reading until a holder appears.
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
            # A new notice, or a field of its own, is not this one's holder.
            if not following or _PLAIN_COPYRIGHT.match(following) or ":" in following:
                break
            step = ahead
            notice = f"{notice} {following}".strip().rstrip("*/ ")
        notice = _DANGLING.sub("", notice).rstrip("*/,; ")
        if _PLACEHOLDER.search(notice) or _NOT_A_NOTICE.search(notice):
            continue
        if not _names_a_holder(notice) and not re.search(r"\d{4}", notice):
            continue
        if notice not in found:
            found.append(notice)
    return found


def _debian_copyright_fields(lines: list[str]) -> list[str]:
    """Holders a Debian copyright file states without the word `Copyright`.

    `Copyright:` is a field there, and its value may be a bare name -- libmd
    credits Poul-Henning Kamp, Colin Plumb and Steve Reid that way. Nothing else
    in the file names them, so a parser looking for the word finds nobody.
    """
    found: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("Copyright:"):
            continue
        # Debian writes the field either way, and the same-line form is the
        # commoner by far. `Copyright: Jane Doe` matches no notice pattern --
        # the colon breaks it -- so without this such a package credits nobody.
        inline = line[len("Copyright:") :].strip()
        values = ([f" {inline}"] if inline else []) + list(lines[index + 1 :])
        for position, value in enumerate(values):
            if not value[:1].isspace() or not value.strip():
                break
            holder = " ".join(value.split()).rstrip(".")
            # A field whose value opens with a notice proper is read by the scan
            # below, prose and all; only a field naming a bare holder is read
            # here, and only for as long as it keeps naming one.
            if _PLAIN_COPYRIGHT.match(holder) or _PROSE.search(holder):
                break
            # Debian writes `None` where a package states no holder.
            if holder.lower() in {"none", "n/a", "unknown"}:
                break
            if position and len(holder) > 60:
                break
            if _names_a_holder(f"Copyright {holder}") and holder not in found:
                found.append(holder)
    return found


def _names_a_holder(notice: str) -> bool:
    """Whether anything is left once the years and their punctuation come out."""
    rest = _YEARS_AND_MARKS.sub(" ", notice[len("Copyright") :])
    return bool(re.search(r"[A-Za-z]{2}", rest))


def read_notice(text: str) -> str | None:
    """The first notice a licence text states, where only one is wanted."""
    notices = read_notices(text)
    return notices[0] if notices else None


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
        try:
            parsed = _spdx_licensing().parse(expression, validate=False)
        except (ExpressionError, ValueError, TypeError):
            # A header tag is written by hand and can say anything -- "Public
            # domain or MIT License" is not an expression. One that will not
            # parse names no reference text; the component is then reported with
            # no licence evidence, which is the gap, rather than stopping here.
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
    """Rename map putting components with byte-identical terms on one LicenseRef.

    A LicenseRef identifies a text, not a component, so two builds of one SDK
    under the same EULA share an id. The map is produced here, before anything
    renders, so the document, the report and the packaged notices all name the
    same id -- they are cross-referenced, and a text shipped under one name in
    the wheel's notices and another in the wheel's SBOM reconciles with nothing.
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
        canonical = f"LicenseRef-shared-text-{sha256_text(text)[:12]}"
        for key in keys:
            renamed[license_ref(key)] = canonical
    return renamed


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rename_refs(expression: str, renamed: dict[str, str]) -> str:
    """Substitute whole identifiers, never substrings.

    `LicenseRef-alpha` is a prefix of `LicenseRef-alpha-extra`, so a plain
    replace turns the longer id into one nothing defines and orphans its entry.
    """
    for old_id, new_id in renamed.items():
        expression = re.sub(
            rf"(?<![\w.-]){re.escape(old_id)}(?![\w.-])", new_id, expression
        )
    return expression


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
            # Walking past it hid every reference inside: a document could carry
            # `LicenseRef-never-defined AND (((` and be reported as defining
            # everything it uses.
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
