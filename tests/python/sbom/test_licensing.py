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
        licensing._evidence(
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


def test_an_upstream_spdx_tag_is_reported_as_declared(tmp_path):
    # REUSE-IgnoreStart
    (tmp_path / "LICENSE").write_text(
        "SPDX-License-Identifier: MIT\n\n" + synth.spdx_text("MIT"), encoding="utf-8"
    )
    # REUSE-IgnoreEnd

    evidence = licensing.discover_in_tree("thing", tmp_path, "thing-src")
    concluded, declared = licensing.expression(evidence, "thing")

    assert (concluded, declared) == ("MIT", "MIT")


def test_a_component_carrying_two_licenses_combines_them():
    """Two identified grants must AND together, not collapse to NOASSERTION.

    The combiner returns a parsed expression object, and asking such an object
    for its truth value raises rather than answering -- so this path has to be
    exercised with more than one licence.
    """
    evidence = [
        licensing._evidence(
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
