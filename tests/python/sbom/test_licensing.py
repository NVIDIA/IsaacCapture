# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Licenses are identified from the text a component ships, or not at all."""

from __future__ import annotations

import synth
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


def test_unmatched_grant_becomes_a_license_ref_not_a_guess():
    evidence = [
        licensing.evidence_for(
            "demo", "component-file", "demo/LICENSE", synth.PROPRIETARY_LICENSE, "grant"
        )
    ]

    concluded, _ = licensing.expression(evidence, "demo")

    assert concluded == "LicenseRef-demo"


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
    under a documentation licence."""
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

    assert concluded == "Apache-2.0"
    assert "CC-BY-4.0" not in declared


def test_a_component_carrying_two_licenses_combines_them():
    """Two identified grants must AND together, not collapse to NOASSERTION.

    The combiner returns a parsed expression object, and asking such an object
    for its truth value raises rather than answering -- so this path has to be
    exercised with more than one licence.
    """
    evidence = [
        licensing.evidence_for(
            "dual",
            "component-file",
            "dual/LICENSE",
            synth.spdx_text("Apache-2.0") + "\n\n" + synth.spdx_text("CC-BY-4.0"),
            "grant",
        )
    ]

    concluded, _ = licensing.expression(evidence, "dual")

    assert concluded == "Apache-2.0 AND CC-BY-4.0"


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
