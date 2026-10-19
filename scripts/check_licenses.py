# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check and fix Isaac Capture source headers, including all EgoPost assets."""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from reuse.global_licensing import ReuseTOML
from reuse.project import Project

EGOPOST_SUBTREE = Path("src/python/isaaccapture_egopost")
EGOPOST_LICENSE = "LicenseRef-OpenMDW-1.1"
DEFAULT_LICENSE = "Apache-2.0"
COPYRIGHT_HOLDER = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
# Same rule as check-copyright-year's `files:`; covers the SBOM suffixes (tested).
SOURCE_FILES = re.compile(
    r"\.(py|pyi|js|ts|cpp|cc|c|h|hpp|cxx|hxx|inc|cu|cuh|frag|vert|cmake|md|yaml|yml"
    r"|sh|fbs|toml|html)$"
    r"|CMakeLists\.txt$|Dockerfile(\.[A-Za-z0-9_-]+)?$"
)


def covered_paths(project: Project) -> list[Path]:
    # Check source files repository-wide; EgoPost additionally includes all assets.
    paths = (path.relative_to(project.root) for path in project.all_files())
    return sorted(
        path
        for path in paths
        if path.is_relative_to(EGOPOST_SUBTREE) or SOURCE_FILES.search(str(path))
    )


def third_party_path(path: Path) -> bool:
    # Directory exemptions follow the repository layout: third_party lives in deps.
    return path.parts[0] == "deps"


def is_egopost_default(project: Project, path: Path, info) -> bool:
    if info.source_type.value != "reuse-toml" or info.source_path != "REUSE.toml":
        return False
    annotation = ReuseTOML.from_file(project.root / "REUSE.toml").find_annotations_item(
        path
    )
    # A subtree fallback is policy, not an explicit per-file exception.
    return annotation is not None and f"{EGOPOST_SUBTREE}/**" in annotation.paths


def check_egopost_default(project: Project) -> bool:
    annotations = ReuseTOML.from_file(project.root / "REUSE.toml").annotations
    defaults = [item for item in annotations if f"{EGOPOST_SUBTREE}/**" in item.paths]
    if not defaults or any(
        {str(expr) for expr in item.spdx_expressions} != {EGOPOST_LICENSE}
        or item.precedence.value != "closest"
        for item in defaults
    ):
        print(
            f"REUSE.toml: EgoPost default must be {EGOPOST_LICENSE} with closest precedence"
        )
        return False
    return True


def first_party_licenses(project: Project, path: Path) -> set[str] | None:
    if third_party_path(path):
        return None
    infos = project.reuse_info_of(path)
    notices = {notice.name for info in infos for notice in info.copyright_notices}
    explicit_exception = any(
        info.spdx_expressions
        and info.source_type.value != "file-header"
        and not is_egopost_default(project, path, info)
        for info in infos
    )
    if (
        explicit_exception
        or not notices
        or not all("NVIDIA CORPORATION & AFFILIATES" in notice for notice in notices)
    ):
        return None
    return {str(expr) for info in infos for expr in info.spdx_expressions}


def fix_license(project: Project, path: Path) -> bool:
    if third_party_path(path):
        return False
    expected = (
        EGOPOST_LICENSE if path.is_relative_to(EGOPOST_SUBTREE) else DEFAULT_LICENSE
    )
    infos = project.reuse_info_of(path)
    notices = {notice for info in infos for notice in info.copyright_notices}
    effective = {str(expr) for info in infos for expr in info.spdx_expressions}
    expressions = first_party_licenses(project, path)
    add_copyright = not notices and effective <= {expected}
    if add_copyright or expressions == set():
        command = [
            sys.executable,
            "-m",
            "reuse",
            "annotate",
            "--fallback-dot-license",
            "-l",
            expected,
        ]
        if add_copyright:
            command += ["-c", COPYRIGHT_HOLDER, "--year", str(date.today().year)]
        subprocess.run(command + [str(path)], check=True)
        return True
    if expressions is None:
        return False
    if expected == EGOPOST_LICENSE and expressions == {"Apache-2.0"}:
        content = path.read_text()
        # Replace only the default header tag, retaining copyright and other notices.
        updated, count = re.subn(
            r"(SPDX-License-Identifier:[ \t]*)Apache-2\.0(?=[ \t]*(?:\*/|-->)?[ \t]*$)",
            rf"\g<1>{EGOPOST_LICENSE}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if count:
            path.write_text(updated)
            print(f"{path}: updated default license to {EGOPOST_LICENSE}")
            return True
    return False


def check_policy(project: Project, path: Path) -> bool:
    expressions = first_party_licenses(project, path)
    if expressions is not None and EGOPOST_LICENSE not in expressions:
        print(
            f"{path}: expected {EGOPOST_LICENSE}; document any license exception explicitly"
        )
        return False
    return True


def main(fix: bool = False) -> int:
    project = Project.from_directory(Path.cwd())
    if not check_egopost_default(project):
        return 1
    paths = covered_paths(project)
    changed = False
    if fix:
        for path in paths:
            changed = fix_license(project, path) or changed
        project = Project.from_directory(Path.cwd())
    ok = True
    # Pass complete coverage on every invocation, including metadata-only changes.
    for start in range(0, len(paths), 200):
        result = subprocess.run(
            [sys.executable, "-m", "reuse", "lint-file", "--"]
            + [str(path) for path in paths[start : start + 200]],
            check=False,
        )
        ok = result.returncode == 0 and ok
    for path in paths:
        if path.is_relative_to(EGOPOST_SUBTREE):
            ok = check_policy(project, path) and ok
    if changed:
        print(
            "License headers updated; review and stage the changes before committing."
        )
    return 0 if ok and not changed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="add default NVIDIA headers and repair default license tags",
    )
    sys.exit(main(fix=parser.parse_args().fix))
