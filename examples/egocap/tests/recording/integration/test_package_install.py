"""Install/import checks for the Ego Capture UI recording package.

Standalone now: no checkout bridge, no environment variable this package needs to import or run.
"""

import ast
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import egocap

# This file lives at tests/recording/integration/, a sibling of the real
# package tree (recording/egocap/), not inside it -- go up to egocap/
# root (3 levels: integration -> recording -> tests) then into recording/.
EGOCAP_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = EGOCAP_ROOT / "recording"

#: The four real importable packages this one "egocap" distribution ships, and where each one's
#: actual source lives -- see pyproject.toml's [tool.setuptools.packages.find] comment for why
#: there are four `where` roots with the package one level down from each.
_SHIPPED_PACKAGE_DIRS = {
    "egocap": EGOCAP_ROOT / "recording" / "egocap",
    "recording_service": EGOCAP_ROOT / "recording" / "recording_service",
    "capture_server": EGOCAP_ROOT / "capture_server" / "capture_server",
    "egocap_web": EGOCAP_ROOT / "backend" / "egocap_web",
}


def _all_submodule_names(package_name: str) -> list[str]:
    """The package itself plus every submodule, recursively -- found by walking the real source
    tree directly (not pkgutil.walk_packages on the live __path__, which misresolves
    capture_server's as an ambiguous namespace path under this project's multi-root editable
    install), so this actually covers "every importable module", not just a couple hand-picked
    ones."""
    root = _SHIPPED_PACKAGE_DIRS[package_name]
    names = [package_name]
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.name == "__init__.py":
            parts = relative.parts[:-1]
        else:
            parts = (*relative.parts[:-1], relative.stem)
        if not parts:
            continue
        names.append(".".join((package_name, *parts)))
    return names


def test_egocap_package_is_installed_and_importable():
    # The editable install covers the whole consolidated egocap project
    # (pyproject.toml's project.name is "egocap", one dist for
    # recording/capture_server together) -- importlib.metadata looks up
    # dists by that project name, not by an individual importable module's
    # name, so this checks "egocap", then that the `egocap` module
    # specifically still resolves to this exact checkout.
    dist = importlib.metadata.distribution("egocap")
    assert dist.metadata["Name"] == "egocap"
    assert dist.version == "0.1.0"
    assert egocap.__file__ is not None
    assert Path(egocap.__file__).resolve().parent == PACKAGE_ROOT / "egocap"


def test_egocap_package_imports_cleanly_with_no_environment_requirements():
    """Every importable module in all four shipped packages imports with no special environment
    set -- not just a couple of hand-picked ones."""
    modules: list[str] = []
    for package_name in _SHIPPED_PACKAGE_DIRS:
        modules.extend(_all_submodule_names(package_name))
    script = "; ".join(f"import {m}" for m in modules) + "; print('OK')"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PACKAGE_ROOT),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_missing_native_sdk_services_not_imported_anywhere():
    # None of these hardware/SDK/upload packages are declared dependencies
    # in pyproject.toml; a real import of any of them anywhere in egocap/
    # would mean an SDK slipped back in. The static AST scan below is the
    # actual guard -- whether these packages happen to be installed in this
    # environment proves nothing about egocap/'s own code.
    banned = ("depthai", "record3d", "boto3", "tkinter")
    for path in PACKAGE_ROOT.joinpath("egocap").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                top = (name or "").split(".")[0]
                assert top not in banned, f"{path} imports banned SDK {name!r}"
