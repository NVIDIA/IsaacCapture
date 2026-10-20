# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Licenses are identified from the text a component ships, or not at all."""

from __future__ import annotations

import math

import pytest
import synth
from sbom import discovery, document, report
from sbom import licensing


def test_corpus_loads():
    """The whole SPDX list is available, so no subset has to be maintained."""
    version, entries = licensing.corpus()

    assert version.startswith("3.")
    assert len(entries) > 400
    assert {"MIT", "Apache-2.0", "BSD-3-Clause", "Zlib", "Qhull"} <= set(entries)


def test_deprecated_identifiers_are_not_offered():
    """Naming a component with a withdrawn identifier would be reporting a
    license SPDX no longer recognises."""
    _, entries = licensing.corpus()

    assert "GPL-2.0" not in entries, "superseded by GPL-2.0-only/-or-later"
    assert "GPL-2.0-only" in entries


def test_an_extended_license_is_not_mistaken_for_the_one_it_contains():
    """ECL-2.0 is Apache-2.0 plus a clause, and CC-BY-NC-4.0 is CC-BY-4.0 plus
    one. Both clear containment against the shorter licence's own text, so the
    file has to be resolved to the licence that explains all of it."""
    apache = [
        license_id
        for license_id, _, _ in licensing.identify(synth.spdx_text("Apache-2.0"))
    ]
    creative = [
        license_id
        for license_id, _, _ in licensing.identify(synth.spdx_text("CC-BY-4.0"))
    ]

    assert apache == ["Apache-2.0"]
    assert creative == ["CC-BY-4.0"]


def test_the_extended_license_still_wins_when_it_is_the_real_one():
    """The resolution has to work in both directions, or it is just a bias."""
    matches = [
        license_id
        for license_id, _, _ in licensing.identify(synth.spdx_text("ECL-2.0"))
    ]

    assert matches == ["ECL-2.0"]


def test_identifies_a_canonical_text():
    matches = licensing.identify(synth.spdx_text("MIT"))

    assert [license_id for license_id, _, _ in matches] == ["MIT"]


def test_identifies_a_text_wrapped_in_a_copyright_line():
    text = "Copyright (c) 2019 Some Person\n\n" + synth.spdx_text("Zlib")

    assert [license_id for license_id, _, _ in licensing.identify(text)] == ["Zlib"]


def test_a_longer_license_subsumes_the_shorter_one_it_contains():
    """A BSD-3 file contains the BSD-2 text; only BSD-3 should be reported."""
    matches = licensing.identify(synth.spdx_text("BSD-3-Clause"))

    assert [license_id for license_id, _, _ in matches] == ["BSD-3-Clause"]


def test_a_dual_licensed_file_reports_both():
    text = synth.spdx_text("Apache-2.0") + "\n\n" + synth.spdx_text("CC-BY-4.0")

    identified = {license_id for license_id, _, _ in licensing.identify(text)}

    assert identified == {"Apache-2.0", "CC-BY-4.0"}


def _varied_prose(word_count: int) -> str:
    """Filler whose wording does not repeat, the way real agreements read."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return " ".join(
        "term"
        + alphabet[index % 26]
        + alphabet[(index // 26) % 26]
        + alphabet[(index // 676) % 26]
        for index in range(word_count)
    )


def test_a_license_quoted_in_an_appendix_is_not_the_document_s_license():
    """Large agreements reproduce whole OSS licenses; that is not their license."""
    agreement = (
        "PROPRIETARY AGREEMENT\n\n"
        + _varied_prose(6000)
        + "\n\nAPPENDIX: third-party notices\n\n"
        + synth.spdx_text("BSL-1.0")
    )

    assert licensing.identify(agreement) == []


def test_a_real_license_survives_surrounding_packaging_boilerplate():
    """A Debian copyright file is mostly the license, wrapped in packaging prose."""
    wrapped = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        "Upstream-Name: thing\n\n"
        + synth.spdx_text("Zlib")
        + "\n\n"
        + _varied_prose(200)
    )

    assert [license_id for license_id, _, _ in licensing.identify(wrapped)] == ["Zlib"]


def _grant(key: str, name: str, text: str) -> licensing.LicenseEvidence:
    return licensing.evidence_for(key, "component-file", f"{key}/{name}", text, "grant")


def test_unmatched_grant_becomes_a_license_ref_not_a_guess():
    evidence = [_grant("demo", "LICENSE", synth.PROPRIETARY_LICENSE)]

    concluded, _ = licensing.expression(evidence, "demo")

    assert concluded == "LicenseRef-demo"


def test_a_matched_grant_does_not_hide_an_unmatched_one():
    mit = _grant("demo", "LICENSE", synth.spdx_text("MIT"))
    eula = _grant("demo", "LICENSE-EULA.txt", synth.PROPRIETARY_LICENSE)

    assert licensing.expression([mit, eula], "demo") == (
        "LicenseRef-demo AND MIT",
        "LicenseRef-demo AND MIT",
    )
    assert licensing.carried_texts([mit, eula]) == [eula]
    assert licensing.verbatim_terms([mit, eula]) == synth.PROPRIETARY_LICENSE
    assert licensing.why_unnamed([mit, eula]) == (
        "matched no text in the SPDX reference corpus"
    )


_GPL_TIE = (
    "matches the reference text of GPL-3.0-only or GPL-3.0-or-later, and the "
    "text alone does not say which applies"
)
_NEWER = "newer than the SPDX grammar this document ships with"


def test_a_text_two_ids_share_is_carried_rather_than_named_as_both():
    gpl = _grant("gpl", "COPYING", synth.spdx_text("GPL-3.0-only"))
    mit = _grant("gpl", "LICENSE-MIT", synth.spdx_text("MIT"))
    eula = _grant("gpl", "EULA", synth.PROPRIETARY_LICENSE)

    assert licensing.expression([gpl], "gpl")[0] == "LicenseRef-gpl"
    assert licensing.expression([gpl, mit], "gpl")[0] == "LicenseRef-gpl AND MIT"
    assert licensing.carried_texts([gpl, mit]) == [gpl]
    assert licensing.why_unnamed([gpl, mit]) == _GPL_TIE
    assert licensing.why_unnamed([eula, gpl]) == (
        f"matched no text in the SPDX reference corpus; {_GPL_TIE}"
    )


@pytest.mark.parametrize("license_id", ["MPL-2.0", "OFL-1.1"])
def test_a_tie_with_a_plain_id_names_the_plain_id(license_id):
    text = _grant("font", "LICENSE", synth.spdx_text(license_id))
    assert len(licensing.alternatives(text.matches)[0]) > 1, "not a tie any more"

    assert licensing.expression([text], "font")[0] == license_id
    assert text.identified == license_id


def test_a_tie_the_plain_id_names_is_not_given_as_a_reason():
    both = _grant(
        "c",
        "COPYING",
        synth.spdx_text("MPL-2.0") + "\n\n" + synth.spdx_text("LGPL-2.1-only"),
    )

    assert licensing.expression([both], "c")[0] == "LicenseRef-c AND MPL-2.0"
    assert licensing.why_unnamed([both]) == (
        "matches the reference text of LGPL-2.1-only or LGPL-2.1-or-later, and "
        "the text alone does not say which applies"
    )


def test_a_tied_text_says_how_little_of_it_the_reference_text_is():
    padded = _grant(
        "c", "COPYING", synth.spdx_text("LGPL-2.1-only") + _varied_prose(3000)
    )
    share = licensing._read_grant(padded).explained
    # Counted once it explains too little; counted per id it would not.
    assert licensing._EXPLAINED_THRESHOLD / 2 < share < licensing._EXPLAINED_THRESHOLD

    assert licensing.why_unnamed([padded]) == (
        "matches the reference text of LGPL-2.1-only or LGPL-2.1-or-later, and "
        "the text alone does not say which applies; what matched in the SPDX "
        f"reference corpus accounts for only {math.floor(share * 100)}% of c/COPYING, "
        "its copyright notices aside"
    )


def _share(text: licensing.LicenseEvidence) -> str:
    return f"{math.floor(licensing._read_grant(text).explained * 100)}%"


def test_a_mostly_unmatched_grant_is_named_and_carried():
    """The rest of the text may be terms no id names."""
    whole = _grant("demo", "LICENSE", synth.spdx_text("MIT"))
    padded = _grant("demo", "COPYING", synth.spdx_text("MIT") + _varied_prose(300))
    zlib = _grant("demo", "COPYING.zlib", synth.spdx_text("Zlib") + _varied_prose(250))

    assert licensing.expression([whole], "demo")[0] == "MIT"
    assert licensing.expression([padded, whole], "demo")[0] == (
        "LicenseRef-demo AND MIT"
    )
    assert licensing.carried_texts([padded, whole]) == [padded]
    assert licensing.why_unnamed([padded, zlib]) == (
        "what matched in the SPDX reference corpus accounts for only "
        f"{_share(padded)} of demo/COPYING, its copyright notices aside; what matched "
        f"in the SPDX reference corpus accounts for only {_share(zlib)} of "
        "demo/COPYING.zlib, its copyright notices aside"
    )


_FMT_EXCEPTION = (
    "As an exception, if, as a result of your compiling your source code, portions "
    "of this Software are embedded into a machine-executable object form of such "
    "source code, you may redistribute such embedded portions in such object form "
    "without including the above copyright and permission notices."
)


@pytest.mark.parametrize(
    "shape, share, concluded",
    [
        # fmt's own shape: named at 0.75, carried at 0.80.
        ("exception", (0.75, 0.80), "LicenseRef-demo AND MIT"),
        # Two thirds of it: carried at 0.85, named at 0.80.
        ("part-exception", (0.80, 0.85), "MIT"),
        # A holder list is not terms, however long.
        ("holders", (0.99, 1.01), "MIT"),
    ],
)
def test_how_much_of_a_grant_must_match_for_its_ids_to_say_it_all(
    shape, share, concluded
):
    mit = synth.spdx_text("MIT")
    text = {
        "exception": f"{mit}\n\n--- Optional exception to the license ---\n\n"
        + _FMT_EXCEPTION,
        "part-exception": mit + "\n\n" + " ".join(_FMT_EXCEPTION.split()[:34]),
        "holders": "\n".join(
            f"Copyright (c) 2015 Holder Number{index} <h{index}@example.org>"
            for index in range(8)
        )
        + "\n\n"
        + mit,
    }[shape]
    grant = _grant("demo", "LICENSE", text)
    assert share[0] < licensing._read_grant(grant).explained < share[1]

    assert licensing.expression([grant], "demo")[0] == concluded


@pytest.fixture
def mit_is_newer(monkeypatch):
    """MIT as an id the grammar this document ships with cannot express."""
    expressible = licensing.expressible
    monkeypatch.setattr(
        licensing, "expressible", lambda key: key != "MIT" and expressible(key)
    )


def _reference(key: str, license_id: str) -> licensing.LicenseEvidence:
    return licensing.evidence_for(
        key,
        licensing.SPDX_REFERENCE,
        f"{license_id}.txt",
        synth.spdx_text(license_id),
        "grant",
    )


@pytest.mark.parametrize(
    "texts, concluded, carried, reason",
    [
        (
            ["MIT", "Zlib"],
            "LicenseRef-demo AND Zlib",
            [0],
            f"matched MIT, {_NEWER}",
        ),
        (
            ["MIT", None],
            "LicenseRef-demo",
            [0, 1],
            f"matched no text in the SPDX reference corpus; matched MIT, {_NEWER}",
        ),
        (
            ["MIT", "GPL-3.0-only"],
            "LicenseRef-demo",
            [0, 1],
            f"{_GPL_TIE}; matched MIT, {_NEWER}",
        ),
        # A reference text is declared, not unmatched, and its id is not
        # restated as matched.
        (
            ["reference", "MIT", "Zlib"],
            "LicenseRef-demo AND Zlib",
            [0, 1],
            f"declares MIT, {_NEWER}",
        ),
    ],
    ids=["beside-named", "beside-unmatched", "beside-tie", "declared"],
)
def test_an_id_the_grammar_cannot_name_carries_only_its_texts(
    mit_is_newer, texts, concluded, carried, reason
):
    evidence = [
        _reference("demo", "MIT")
        if source == "reference"
        else _grant("demo", f"LICENSE-{index}", synth.PROPRIETARY_LICENSE)
        if source is None
        else _grant("demo", f"LICENSE-{index}", synth.spdx_text(source))
        for index, source in enumerate(texts)
    ]

    assert licensing.expression(evidence, "demo")[0] == concluded
    assert licensing.carried_texts(evidence) == [evidence[i] for i in carried]
    assert licensing.why_unnamed(evidence) == reason


def test_a_declaration_the_grammar_cannot_name_travels_whole(mit_is_newer):
    """An OR cannot be kept in part, so every text it declares is carried."""
    declared, evidence = licensing.declared_terms("vendored:x", {"MIT OR Zlib"})
    assert declared == "LicenseRef-vendored-x"
    assert [item.identified for item in licensing.carried_texts(evidence)] == [
        "MIT",
        "Zlib",
    ]
    declared, evidence = licensing.declared_terms("vendored:y", {"Zlib OR BSL-1.0"})
    assert declared == "Zlib OR BSL-1.0"
    assert licensing.carried_texts(evidence) == []


@pytest.mark.parametrize(
    "carried, total, subject",
    [
        (1, 1, "the shipped grant is"),
        (2, 2, "its 2 shipped grants are"),
        (1, 2, "1 of its 2 shipped grants is"),
        (2, 3, "2 of its 3 shipped grants are"),
    ],
)
def test_the_report_says_how_many_grants_are_carried(carried, total, subject):
    assert report._carried_subject(carried, total) == subject


def test_the_report_says_whether_a_carried_grant_is_named_too(mit_is_newer):
    padded = _grant("demo", "COPYING", synth.spdx_text("Zlib") + _varied_prose(250))
    eula = _grant("demo", "EULA", synth.PROPRIETARY_LICENSE)
    _, evidence = licensing.declared_terms("vendored:x", {"MIT OR Zlib"})

    assert report._carried_reason([padded]) == (
        "the shipped grant is carried verbatim as well as named: what matched in "
        f"the SPDX reference corpus accounts for only {_share(padded)} of "
        "demo/COPYING, its copyright notices aside"
    )
    assert report._carried_reason([padded, eula]).startswith(
        "its 2 shipped grants are carried verbatim, 1 of them as well as named: "
    )
    # A declaration carried whole names none of its ids.
    assert report._carried_reason(evidence) == (
        "its 2 shipped grants are carried verbatim rather than named: declares "
        f"MIT, {_NEWER}"
    )


def test_no_grant_at_all_is_noassertion():
    assert licensing.expression([], "demo") == ("NOASSERTION", "NOASSERTION")


def test_finds_a_prefixed_license_file(tmp_path):
    (tmp_path / "BSD-LICENSE").write_text(
        synth.spdx_text("BSD-3-Clause"), encoding="utf-8"
    )

    evidence = licensing.discover_in_tree("ccd", tmp_path, "ccd-src")

    assert [item.identified for item in evidence] == ["BSD-3-Clause"]


def test_a_reuse_pool_does_not_become_the_project_license(tmp_path):
    """LICENSES/ holds texts for individual files, not the project's own grant."""
    (tmp_path / "LICENSE").write_text(synth.spdx_text("Apache-2.0"), encoding="utf-8")
    pool = tmp_path / "LICENSES"
    pool.mkdir()
    (pool / "MIT.txt").write_text(synth.spdx_text("MIT"), encoding="utf-8")
    (pool / "Zlib.txt").write_text(synth.spdx_text("Zlib"), encoding="utf-8")

    evidence = licensing.discover_in_tree("openxr", tmp_path, "openxr-src")
    concluded, _ = licensing.expression(evidence, "openxr")

    assert concluded == "Apache-2.0"
    assert sum(1 for item in evidence if item.kind == "pool") == 2, (
        "pool texts still ship"
    )


def test_falls_back_to_a_readme_section_when_no_license_file_exists(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Thing\n\nSome prose.\n\n## Licence\nPublic domain or MIT License.\n",
        encoding="utf-8",
    )

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")

    assert [item.origin for item in evidence] == ["readme-section"]
    assert licensing.expression(evidence, "thing")[0] == "LicenseRef-thing"


def test_notices_are_packaged_but_do_not_set_the_expression(tmp_path):
    (tmp_path / "LICENSE").write_text(synth.spdx_text("Apache-2.0"), encoding="utf-8")
    (tmp_path / "NOTICE").write_text(
        "This product includes software from X.\n", encoding="utf-8"
    )

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")

    assert {item.kind for item in evidence} == {"grant", "notice"}
    assert licensing.expression(evidence, "thing")[0] == "Apache-2.0"


def test_shipping_a_license_file_is_how_authors_declare_one(tmp_path):
    """Declared is not an echo of the conclusion: it is read from the same file,
    because shipping that file is the declaration."""
    (tmp_path / "LICENSE").write_text(synth.spdx_text("MIT"), encoding="utf-8")

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")

    assert licensing.expression(evidence, "thing") == ("MIT", "MIT")


def test_a_component_with_no_grant_declares_nothing(tmp_path):
    (tmp_path / "README.md").write_text("nothing to see", encoding="utf-8")

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")

    assert licensing.expression(evidence, "thing") == ("NOASSERTION", "NOASSERTION")


def test_a_tag_on_a_documentation_file_is_not_the_package_declaration(tmp_path):
    """OpenXR's COPYING.adoc is prose about which licences the project uses,
    tagged CC-BY-4.0 for the prose. Promoting it declared an Apache-2.0 SDK
    under a documentation licence. The prose matches nothing, so it is carried
    as the component's LicenseRef."""
    (tmp_path / "LICENSE").write_text(synth.spdx_text("Apache-2.0"), encoding="utf-8")
    # REUSE-IgnoreStart
    (tmp_path / "COPYING").write_text(
        "SPDX-License-Identifier: CC-BY-4.0\n\nThis project uses several licences.\n",
        encoding="utf-8",
    )
    # REUSE-IgnoreEnd

    concluded, declared = licensing.expression(
        licensing.discover_in_tree("thing", tmp_path, "thing-src"), "thing"
    )

    assert concluded == "Apache-2.0 AND LicenseRef-thing"
    assert "CC-BY-4.0" not in declared


@pytest.mark.parametrize(
    "files, concluded",
    [
        ({"LICENSE": ("Apache-2.0", "CC-BY-4.0")}, "Apache-2.0 AND CC-BY-4.0"),
        (
            {"LICENSE-MIT": ("MIT",), "LICENSE-APACHE": ("Apache-2.0",)},
            "Apache-2.0 AND MIT",
        ),
    ],
)
def test_two_identified_licenses_combine(files, concluded):
    """Two identified licences must AND together, not collapse to NOASSERTION.

    The combiner returns a parsed expression object, and asking such an object
    for its truth value raises rather than answering -- so this path has to be
    exercised with more than one licence.
    """
    evidence = [
        _grant("dual", name, "\n\n".join(map(synth.spdx_text, ids)))
        for name, ids in files.items()
    ]

    assert licensing.expression(evidence, "dual")[0] == concluded
    # Nothing carried, so every grant is the terms the build evidence states.
    assert licensing.verbatim_terms(evidence) == "\n\n".join(
        item.text for item in evidence
    )


def test_a_bare_license_file_covers_its_directory(tmp_path):
    synth.write(tmp_path / "LICENSE", synth.spdx_text("MIT"))
    synth.write(tmp_path / "third/.license", "SPDX-FileCopyrightText: 2020 A\n")
    # Named like a sidecar, with no file beside it to annotate.
    synth.write(tmp_path / "third/stb.license", synth.spdx_text("MIT"))

    nested = [
        item
        for item in licensing.discover_in_tree("thing", tmp_path, "thing-src")
        if item.kind == "nested"
    ]

    assert [(item.path, item.scope) for item in nested] == [
        ("thing-src/third/.license", "thing-src/third"),
        ("thing-src/third/stb.license", "thing-src/third"),
    ]


def test_a_text_carrying_two_licences_lists_both_not_a_choice():
    dual = _grant(
        "dual",
        "LICENSE",
        synth.spdx_text("Apache-2.0") + "\n\n" + synth.spdx_text("CC-BY-4.0"),
    )

    assert dual.identified == "Apache-2.0 AND CC-BY-4.0"
    component = discovery.Component(
        key="dual",
        name="dual",
        kind="source-tree",
        supplier="NOASSERTION",
        homepage="NOASSERTION",
        purl="pkg:generic/dual",
        version="1",
        download_location="NOASSERTION",
        source_info="",
        evidence=(dual,),
    )
    rendered = document._attribution_texts(component, synth.DIST_INFO)[0]
    assert " identified as Apache-2.0 (" in rendered
    assert "; CC-BY-4.0 (" in rendered
    assert "reference text of" not in rendered


def test_a_choice_is_never_joined_to_a_licence_with_and():
    gpl = [("GPL-3.0-only", 1.0, 0.6), ("GPL-3.0-or-later", 1.0, 0.6)]
    alternatives = "matches the reference text of GPL-3.0-only or GPL-3.0-or-later"

    assert licensing.matched_prose(gpl) == alternatives
    assert (
        licensing.matched_prose([("Apache-2.0", 1.0, 0.2), *gpl, ("MIT", 1.0, 0.1)])
        == f"Apache-2.0 AND MIT; {alternatives}"
    )


def test_an_index_of_licenses_is_not_a_grant(tmp_path):
    """A cone-mode sparse checkout keeps root files, so a repository that
    merely catalogues licences can present names that look like grants."""
    (tmp_path / "licenses.md").write_text("| MIT | Apache-2.0 |\n", encoding="utf-8")
    (tmp_path / "accessingLicenses.md").write_text(
        "how to fetch them\n", encoding="utf-8"
    )

    assert licensing.discover_in_tree("catalogue", tmp_path, "catalogue-src") == []


def test_a_project_title_is_not_a_license_section(tmp_path):
    """`# License List Data` heads a README; it does not grant anything."""
    (tmp_path / "README.md").write_text(
        "# License List Data\n\nGenerated data.\n", encoding="utf-8"
    )

    assert licensing.discover_in_tree("catalogue", tmp_path, "catalogue-src") == []


def test_a_real_license_section_is_still_found(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Thing\n\nprose\n\n## Licence\nPublic domain or MIT License.\n",
        encoding="utf-8",
    )

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")

    assert [item.origin for item in evidence] == ["readme-section"]


def test_a_notice_another_already_covers_is_dropped():
    """One holder stated over several year ranges is one holder."""
    folded = licensing.fold_notices(
        [
            "Copyright (c) 2025 Acme Ltd.",
            "Copyright (c) 2025-2026 Acme Ltd.",
            "Copyright (c) 2026 Acme Ltd.",
        ]
    )

    assert folded == ["Copyright (c) 2025-2026 Acme Ltd."]


def test_folding_never_claims_a_year_no_file_claims():
    """Merging ranges would assert 2022; neither notice does."""
    assert licensing.fold_notices(
        ["Copyright (c) 2021 Acme Ltd.", "Copyright (c) 2023 Acme Ltd."]
    ) == ["Copyright (c) 2021 Acme Ltd.", "Copyright (c) 2023 Acme Ltd."]


def test_folding_does_not_merge_different_holders():
    folded = licensing.fold_notices(
        ["Copyright (c) 2026 Acme Ltd.", "Copyright (c) 2026 Other Corp."]
    )

    assert len(folded) == 2


def test_a_template_is_not_a_notice():
    """A reference text names nobody; recording its placeholder would."""
    assert licensing.read_notice("Copyright [yyyy] [name of copyright owner]") is None
    assert licensing.read_notice("Copyright (c) <year> <copyright holders>") is None
    assert (
        licensing.read_notice("MIT\n\nCopyright (c) 2016 Wenzel Jakob\n\nPermission")
        == "Copyright (c) 2016 Wenzel Jakob"
    )


def test_a_debian_copyright_field_names_its_holder_either_way(license_data):
    """Debian states a holder in a field, in two spellings, often with no year.

    `Copyright: Jane Doe` matches no notice pattern -- the colon breaks it -- and
    it is the commoner of the two forms, so a system library vendored from such a
    package was credited to nobody. The field also holds prose and the literal
    `None`, which are not holders.
    """
    licensing.load_corpus(license_data)

    same_line = "Copyright: Poul-Henning Kamp <phk@login.dkuug.dk>\nLicense: Beerware\n"
    indented = "Copyright:\n Colin Plumb\n Todd C. Miller\nLicense: public-domain\n"

    assert licensing.read_notices(same_line) == [
        "Poul-Henning Kamp <phk@login.dkuug.dk>"
    ]
    assert licensing.read_notices(indented) == ["Colin Plumb", "Todd C. Miller"]
    assert licensing.read_notices("Copyright: None\nLicense: MIT\n") == []
    assert (
        licensing.read_notices(
            "Copyright: This code is derived from software contributed by X\n"
        )
        == []
    )


def test_every_holder_in_a_text_is_read_not_just_the_first(license_data):
    """Under MIT, BSD and Zlib the notice is the obligation, so all of it counts.

    Holders run on past a comma, sit on the next line with nothing joining them,
    and follow a lead-in of any length.
    """
    licensing.load_corpus(license_data)

    assert licensing.read_notices(
        "Copyright (c) 2002-2006 Marcus Geelnard\n\n"
        "Copyright (c) 2006-2019 Camilla Lowy\n"
    ) == [
        "Copyright (c) 2002-2006 Marcus Geelnard",
        "Copyright (c) 2006-2019 Camilla Lowy",
    ]
    # The holder on the line below, with no comma to join them.
    assert licensing.read_notices(
        "Copyright © 1980, 1982, 1986, 1989-1994\n"
        "    The Regents of the University of California.\n"
    ) == [
        "Copyright © 1980, 1982, 1986, 1989-1994 The Regents of the University "
        "of California."
    ]
    # A lead-in longer than a name, and a clause broken by the end of its line.
    assert licensing.read_notices(
        "Box collision code (engine_collision_box.c) is Copyright 2016 S Kolev.\n"
    ) == ["Copyright 2016 S Kolev."]
    assert licensing.read_notices(
        "2010), this software is Copyright (c) 2007-2010 by B Lepilleur, and is\n"
        "released under the terms of the MIT License.\n"
    ) == ["Copyright (c) 2007-2010 by B Lepilleur"]


def test_a_choice_of_licences_is_not_published_as_both(license_data, tmp_path):
    """`A OR B` is one licence at the reader's choice, not two obligations.

    Rebuilding a component's conclusion from the reference texts its identifiers
    name loses the operator, and the roll-up that splits an expression into
    identifiers loses it again.
    """
    from sbom import discovery

    licensing.load_corpus(license_data)
    source = tmp_path / "vendored.h"
    source.write_text(
        "// SPDX-FileCopyrightText: Copyright 2020 Upstream Widgets Ltd\n"
        "// SPDX-License-Identifier: MIT OR GPL-2.0-or-later\n"
    )
    components, _ = discovery.discover_vendored(
        tmp_path, {source}, {"NVIDIA"}, tracked=()
    )
    component = next(iter(components.values()))

    assert component.license_concluded == "MIT OR GPL-2.0-or-later"
    assert component.license_declared == "MIT OR GPL-2.0-or-later"
    # And it survives the roll-up whole, where an AND-only one decomposes.
    assert licensing.operands_of("MIT OR GPL-2.0-or-later") == [
        "(MIT OR GPL-2.0-or-later)"
    ]
    assert licensing.operands_of("Apache-2.0 AND CC-BY-4.0") == [
        "Apache-2.0",
        "CC-BY-4.0",
    ]


def test_an_exception_is_not_a_licence_needing_its_own_text(license_data):
    """`X WITH <exception>` must not be unsatisfiable.

    An exception's text ships only where its expression travels as a LicenseRef,
    so demanding one for every exception blocks such a declaration for nothing.
    """
    licensing.load_corpus(license_data)

    assert licensing.is_exception("LLVM-exception")
    assert licensing.is_exception("Classpath-exception-2.0")
    assert not licensing.is_exception("Apache-2.0")


@pytest.mark.parametrize("content", [None, "{not json", '{"licenses": []}'])
def test_the_exception_list_is_required(tmp_path, content):
    """Without it, an exception newer than the grammar reads as an unknown id."""
    synth._stage_license_data(tmp_path)
    exceptions = tmp_path / "exceptions.json"
    if content is None:
        exceptions.unlink()
    else:
        exceptions.write_text(content)

    with pytest.raises(licensing.CorpusError, match=r"exceptions\.json"):
        licensing.load_corpus(tmp_path)


def test_a_machine_readable_copyright_field_is_read_as_debian_writes_it():
    """Field names ignore case; `None` (or `n/a`) states no holder, and what
    follows it in the field is commentary. A notice wrapped over lines stays
    one paragraph, its lines in order."""
    text = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n"
        "Files: a\ncopyright: 2001 Lower Case Holder\n .\n 2002 Second Holder\n"
        "License: MIT\n\n"
        "Files: b\nCopyright: None\n Copied from elsewhere\nLicense: public-domain\n\n"
        "Files: c\nCopyright: n/a\nLicense: MIT\n\n"
        "Files: d\nCopyright:\n Copyright 1989\n   The Regents.  All rights reserved.\n"
        "License: BSD-3-clause\n"
    )

    assert licensing.dep5_copyright(text) == [
        "2001 Lower Case Holder",
        "2002 Second Holder",
        "Copyright 1989\nThe Regents. All rights reserved.",
    ]


def test_the_debian_packaging_stanza_names_no_holder_of_the_library():
    """`Files: debian/*` credits whoever packaged it; `*` and the header count."""
    text = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        "Copyright: 2000 Header Holder\n\n"
        "Files: *\nCopyright: 2001 Upstream Holder\nLicense: MIT\n\n"
        "Files: debian/*\n debian/patches/fix.patch\n"
        "Copyright: 2020 Debian Packager\nLicense: MIT\n\n"
        "Files:\n debian/*\nCopyright: 2020 Second Packager\nLicense: MIT\n\n"
        "Files: debian ./debian/control\nCopyright: 2020 Third Packager\n"
        "License: MIT\n\n"
        "Files: debian/rules src/mixed.c\nCopyright: 2021 Mixed Stanza Holder\n"
        "License: MIT\n"
    )

    assert licensing.dep5_copyright(text) == [
        "2000 Header Holder",
        "2001 Upstream Holder",
        "2021 Mixed Stanza Holder",
    ]


def test_a_field_paragraph_is_split_into_the_notices_it_states():
    """A notice wraps, so a line continues the one above unless it opens a
    notice of its own -- and one line per holder is the common layout, which
    folding needs to see one at a time."""
    text = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n"
        "Files: *\nCopyright: Copyright © 1980, 1982\n"
        " The Regents of the University. All rights reserved.\n"
        " 2004-2024 Guillem Jover\n 2012 Guillem Jover\n .\n"
        " Some code is derived from software contributed to Berkeley by\n"
        " Somebody, as part of Summer of Code\n 2005 program.\n"
        "License: BSD-3-clause\n\n"
        "Files: md5.c\nCopyright:\n Colin Plumb\n Todd C. Miller\nLicense: MIT\n"
    )

    notices = licensing.dep5_copyright(text)

    assert notices == [
        "Copyright © 1980, 1982\nThe Regents of the University. All rights reserved.",
        "2004-2024 Guillem Jover",
        "2012 Guillem Jover",
        "Some code is derived from software contributed to Berkeley by\n"
        "Somebody, as part of Summer of Code\n2005 program.",
        "Colin Plumb",
        "Todd C. Miller",
    ]
    assert "2012 Guillem Jover" not in licensing.fold_notices(notices)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        # A year list wrapping onto the holder's line is one notice.
        (
            "Copyright: © 1994, 1995,\n 1996 Free Software Foundation, Inc.\n",
            ["© 1994, 1995,\n1996 Free Software Foundation, Inc."],
        ),
        # A holder whose name reads like prose does not swallow the next one.
        (
            "Copyright: 1999-2022 Python Software Foundation\n"
            " 1992-1993 Jean-loup Gailly\n",
            ["1999-2022 Python Software Foundation", "1992-1993 Jean-loup Gailly"],
        ),
        # A line ending in a comma runs on into a lower-case one.
        (
            "Copyright:\n Christian Tismer <tismer@example.org>,\n contributors\n",
            ["Christian Tismer <tismer@example.org>,\ncontributors"],
        ),
    ],
)
def test_a_notice_is_not_cut_or_merged_at_a_line_break(field, expected):
    text = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n"
        f"Files: *\n{field}License: MIT\n"
    )

    assert licensing.dep5_copyright(text) == expected
