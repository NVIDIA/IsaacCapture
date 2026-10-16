# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise licensing checks on real temporary REUSE projects."""

import runpy
import shutil
from pathlib import Path

import pytest
from repo_paths import repo_root
from reuse.project import Project

CHECK = runpy.run_path(str(repo_root() / "scripts/check_licenses.py"))
SUBTREE = Path("src/python/isaaccapture_egopost")
# REUSE-IgnoreStart
COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved."


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(repo_root() / "LICENSES", tmp_path / "LICENSES")
    (tmp_path / SUBTREE).mkdir(parents=True)
    (tmp_path / "REUSE.toml").write_text(
        "version = 1\n[[annotations]]\n"
        f'path = "{SUBTREE}/**"\nprecedence = "closest"\n'
        'SPDX-License-Identifier = "LicenseRef-OpenMDW-1.1"\n'
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("license_id", ["Apache-2.0", "Not-A-License"])
def test_incorrect_egopost_license_fails(project, license_id):
    (project / SUBTREE / "module.py").write_text(
        f"# {COPYRIGHT}\n# SPDX-License-Identifier: {license_id}\n"
    )
    assert CHECK["main"]() == 1
    assert CHECK["main"](fix=True) == 1
    if license_id == "Apache-2.0":
        assert CHECK["main"](fix=True) == 0
        content = (project / SUBTREE / "module.py").read_text()
        assert COPYRIGHT in content
        assert "SPDX-License-Identifier: LicenseRef-OpenMDW-1.1" in content


def test_unannotated_first_party_asset_gets_sidecar(project):
    path = project / SUBTREE / "config.json"
    path.write_text("{}\n")
    assert CHECK["main"]() == 1
    assert CHECK["main"](fix=True) == 1
    assert path.read_text() == "{}\n"
    sidecar = path.with_suffix(".json.license").read_text()
    assert "NVIDIA CORPORATION & AFFILIATES" in sidecar
    assert "LicenseRef-OpenMDW-1.1" in sidecar
    assert CHECK["main"](fix=True) == 0


def test_unannotated_dependency_requires_review(project):
    path = project / "deps" / "third_party" / "vendor.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n")
    assert CHECK["main"](fix=True) == 1
    assert path.read_text() == "value = 1\n"


def test_third_party_header_and_asset_sidecar_survive(project):
    module = SUBTREE / "vendor.py"
    source = (
        "# SPDX-FileCopyrightText: 2018 Original Author\n"
        "# SPDX-License-Identifier: MIT\n"
    )
    (project / module).write_text(source)
    (project / SUBTREE / "config.json").write_text("{}\n")
    (project / SUBTREE / "config.json.license").write_text(
        "SPDX-FileCopyrightText: 2018 Original Author\nSPDX-License-Identifier: MIT\n"
    )
    assert CHECK["main"](fix=True) == 0
    infos = Project.from_directory(project).reuse_info_of(module)
    assert {str(e) for info in infos for e in info.spdx_expressions} == {"MIT"}
    assert (project / module).read_text() == source


@pytest.mark.parametrize("license_id", ["Apache-2.0", "LicenseRef-Missing"])
def test_metadata_only_change_rechecks_assets(project, license_id):
    (project / SUBTREE / "config.json").write_text("{}\n")
    metadata = project / "REUSE.toml"
    metadata.write_text(
        metadata.read_text()
        + 'SPDX-FileCopyrightText = "2020 NVIDIA CORPORATION & AFFILIATES"\n'
    )
    assert CHECK["main"]() == 0
    metadata.write_text(
        metadata.read_text().replace("LicenseRef-OpenMDW-1.1", license_id)
    )
    assert CHECK["main"]() == 1


@pytest.mark.parametrize("source", ["sidecar", "toml"])
def test_explicit_first_party_exception_survives(project, source):
    path = project / SUBTREE / "module.py"
    path.write_text(f"# {COPYRIGHT}\n# SPDX-License-Identifier: Apache-2.0\n")
    if source == "sidecar":
        path.with_suffix(".py.license").write_text(
            f"{COPYRIGHT}\nSPDX-License-Identifier: Apache-2.0\n"
        )
    else:
        metadata = project / "REUSE.toml"
        metadata.write_text(
            metadata.read_text()
            + (
                f'\n[[annotations]]\npath = "{SUBTREE}/module.py"\n'
                'precedence = "override"\n'
                f'{COPYRIGHT.split(": ", 1)[0]} = "2020 NVIDIA CORPORATION & AFFILIATES"\n'
                'SPDX-License-Identifier = "Apache-2.0"\n'
            )
        )
    assert CHECK["main"](fix=True) == 0


def test_missing_default_tag_is_added_without_changing_copyright(project):
    path = project / "module.py"
    path.write_text(f"# {COPYRIGHT}\n\nvalue = 1\n")
    before = Project.from_directory(project).reuse_info_of(path)[0].copyright_notices
    assert CHECK["main"](fix=True) == 1
    content = path.read_text()
    after = Project.from_directory(project).reuse_info_of(path)[0].copyright_notices
    assert after == before
    assert "SPDX-License-Identifier: Apache-2.0" in content
    assert "value = 1" in content
    assert CHECK["main"](fix=True) == 0


def test_unannotated_core_file_gets_apache_header(project):
    path = project / "module.py"
    path.write_text("value = 1\n")
    assert CHECK["main"](fix=True) == 1
    content = path.read_text()
    assert "NVIDIA CORPORATION & AFFILIATES" in content
    assert "SPDX-License-Identifier: Apache-2.0" in content
    assert CHECK["main"](fix=True) == 0


def test_copyright_only_header_uses_validated_egopost_default(project):
    path = project / SUBTREE / "module.py"
    path.write_text(f"# {COPYRIGHT}\nvalue = 1\n")
    assert CHECK["first_party_licenses"](
        Project.from_directory(project), path.relative_to(project)
    ) == {"LicenseRef-OpenMDW-1.1"}
    assert CHECK["main"]() == 0
    metadata = project / "REUSE.toml"
    metadata.write_text(
        metadata.read_text().replace("LicenseRef-OpenMDW-1.1", "Apache-2.0")
    )
    assert (
        CHECK["check_policy"](
            Project.from_directory(project), path.relative_to(project)
        )
        is False
    )
    assert CHECK["main"]() == 1
    assert CHECK["main"](fix=True) == 1
    assert path.read_text() == f"# {COPYRIGHT}\nvalue = 1\n"


# REUSE-IgnoreEnd
