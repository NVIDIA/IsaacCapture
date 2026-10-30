"""The service and the recording-owned wrapper it depends on import only their own declared dependencies.

Static (AST) checks over every production module of the service and of ``egocap/recording/``, and over this
suite's own service-API tests. Dynamic imports, relative imports that escape the package, and aliasing are
all treated as edges. Negative controls prove the scanner rejects each violation kind.

The acquisition-boundary checks below (``FORBIDDEN_PREFIXES``) only catch the specific names they name --
a module importing some other undeclared third-party or monorepo package would pass them silently. The
allowlist checks further down close that gap: ``_ALLOWED_TOP_LEVEL_MODULES`` is this file's own copy of
pyproject.toml's ``[project.dependencies]`` (top-level import names, not PyPI distribution names) plus the
stdlib and this project's own four packages -- anything else is undeclared by construction, including a
monorepo import coming back without anyone having to name it specifically.
"""

import ast
import sys
from pathlib import Path

import egocap
import recording_service

SERVICE_DIR = Path(recording_service.__file__).parent
WRAPPER_DIR = Path(egocap.__file__).parent / "recording"
TESTS_DIR = Path(__file__).parent

FORBIDDEN_PREFIXES = ("capture_server", "egocap.capture")

#: Top-level import names for every pyproject.toml [project.dependencies] entry (PyPI distribution
#: name -> import name differs for a few: PyYAML -> yaml, ruamel.yaml -> ruamel, Pillow -> PIL,
#: opencv-python -> cv2). Kept as a plain literal, not derived from pyproject.toml at import time,
#: so a dependency added there without a matching entry here fails this test instead of silently
#: widening what's allowed.
_ALLOWED_THIRD_PARTY_MODULES = frozenset(
    {
        "numpy",
        "zarr",
        "yaml",
        "ruamel",
        "loguru",
        "PIL",
        "scipy",
        "cv2",
        "websockets",
        "fastapi",
        "uvicorn",
    }
)
_ALLOWED_OWN_PACKAGES = frozenset({"egocap", "recording_service", "capture_server", "egocap_web"})
_ALLOWED_TOP_LEVEL_MODULES = (
    frozenset(sys.stdlib_module_names) | _ALLOWED_THIRD_PARTY_MODULES | _ALLOWED_OWN_PACKAGES
)


def _edges(path: Path, package: str) -> list[tuple[str, str | None]]:
    """Every (module, symbol) the file imports, including dynamic imports and relative-escape markers."""
    tree = ast.parse(path.read_text())
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > len(parts):
                    found.append(("<relative-escape>", None))
                    continue
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.extend((module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("importlib.import_module", "import_module", "__import__") and node.args:
                arg = node.args[0]
                found.append((arg.value if isinstance(arg, ast.Constant) else "<dynamic-import>", None))
    return found


def _forbidden(edges) -> list[str]:
    """Edges that reach acquisition code, or that cannot be resolved statically."""
    return [
        f"{m}.{s}" if s else m
        for m, s in edges
        if m.startswith(FORBIDDEN_PREFIXES) or m in ("<relative-escape>", "<dynamic-import>")
    ]


def _scan(files, package: str):
    """Violations across files, keyed by file name."""
    return {f.name: _forbidden(_edges(f, package)) for f in files if _forbidden(_edges(f, package))}


def _undeclared(edges) -> list[str]:
    """Top-level imports that aren't stdlib, an allowed dependency, or one of this project's own
    four packages -- catches a monorepo import or other undeclared third-party package this file
    doesn't name specifically. This does NOT itself reject the acquisition names FORBIDDEN_PREFIXES
    denies (capture_server/egocap_web/egocap are all in _ALLOWED_OWN_PACKAGES, since they're real
    packages of this project): that boundary is enforced separately, by _forbidden above."""
    out = []
    for m, s in edges:
        if m in ("<relative-escape>", "<dynamic-import>"):
            out.append(m)
            continue
        top = m.split(".", 1)[0]
        if top not in _ALLOWED_TOP_LEVEL_MODULES:
            out.append(f"{m}.{s}" if s else m)
    return out


def _scan_undeclared(files, package: str):
    """Undeclared-import violations across files, keyed by file name."""
    return {f.name: _undeclared(_edges(f, package)) for f in files if _undeclared(_edges(f, package))}


def test_service_modules_have_no_acquisition_edges():
    """The service has no capture_server or egocap.capture imports and uses only egocap.recording."""
    files = sorted(SERVICE_DIR.glob("*.py"))
    assert files
    assert _scan(files, "recording_service") == {}
    for f in files:
        for module, _symbol in _edges(f, "recording_service"):
            assert not module.startswith("egocap") or module.startswith("egocap.recording"), (f.name, module)


def test_service_api_tests_have_no_acquisition_edges():
    """The service-API tests (this suite) never import capture code either."""
    files = [f for f in sorted(TESTS_DIR.glob("*.py")) if f.name != Path(__file__).name]
    assert _scan(files, "tests") == {}


def test_wrapper_modules_have_no_acquisition_imports():
    """egocap/recording/ never imports capture_server or the retired egocap.capture package."""
    assert _scan(sorted(WRAPPER_DIR.glob("*.py")), "egocap.recording") == {}


def test_service_modules_import_only_declared_dependencies():
    """The service imports only stdlib, an allowed pyproject.toml dependency, or its own packages
    -- this is the actual "only declared dependencies" claim this module's docstring makes;
    FORBIDDEN_PREFIXES above only denies the specific acquisition names, not an arbitrary
    undeclared import (including one from the monorepo this project was extracted from)."""
    files = sorted(SERVICE_DIR.glob("*.py"))
    assert files
    assert _scan_undeclared(files, "recording_service") == {}


def test_wrapper_modules_import_only_declared_dependencies():
    """egocap/recording/ imports only stdlib, an allowed pyproject.toml dependency, or its own
    packages -- same "only declared dependencies" enforcement as the service check above."""
    assert _scan_undeclared(sorted(WRAPPER_DIR.glob("*.py")), "egocap.recording") == {}


def test_negative_controls_are_rejected(tmp_path):
    """The scanner flags each violation kind, from a service-like module and from a wrapper-like module."""
    cases = {
        "capture.py": "import capture_server.protocol\n",
        "egocap_capture.py": "from egocap.capture import ingest\n",
        "dynamic.py": "import importlib\nimportlib.import_module('capture_server.protocol')\n",
        "dynamic_unknown.py": "import importlib\nname='x'\nimportlib.import_module(name)\n",
        "relative_escape.py": "from ... import something\n",
    }
    for name, source in cases.items():
        path = tmp_path / name
        path.write_text(source)
        assert _forbidden(_edges(path, "recording_service")), name

    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("from capture_server.protocol import ControlMessage\n")
    assert _forbidden(_edges(wrapper, "egocap.recording")) == ["capture_server.protocol.ControlMessage"]


def test_undeclared_import_negative_control_is_rejected(tmp_path):
    """An import of a real, installed, but undeclared package (not a pyproject.toml dependency)
    is flagged -- this is what would catch a monorepo import coming back without anyone having
    added it to FORBIDDEN_PREFIXES by name."""
    path = tmp_path / "undeclared.py"
    # installed (it's a test dependency), but not a [project.dependencies] entry
    path.write_text("import pytest\n")
    assert _undeclared(_edges(path, "recording_service")) == ["pytest"]
