"""Builds both Ego Capture UI web frontends (``frontend/`` and ``recordings_frontend/``)
via their existing npm scripts, for local development and for test fixtures
that need real built assets. Not part of a packaging/distribution pipeline --
``uv build`` still produces a Python-only wheel; it does not bundle either
frontend.

Run directly for local development: ``uv run python build_support.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EGOCAP_ROOT = Path(__file__).resolve().parent

FRONTEND_DIR = EGOCAP_ROOT / "frontend"
RECORDINGS_FRONTEND_DIR = EGOCAP_ROOT / "recordings_frontend"

#: Both frontends are a fixed pair, not a configurable list -- a third
#: frontend is rare enough that adding one can just extend this tuple.
_FRONTEND_DIRS = (FRONTEND_DIR, RECORDINGS_FRONTEND_DIR)


def build_frontends(frontend_dirs: tuple[Path, ...] = _FRONTEND_DIRS) -> list[Path]:
    """Runs ``npm ci`` then ``npm run build`` in each of ``frontend_dirs``
    (using its committed lockfile, matching what a clean install would do),
    and returns each resulting ``dist/`` path. Defaults to both frontends;
    a caller that only needs one (e.g. a test fixture) can pass just that
    directory so it isn't rebuilding -- and isn't vulnerable to a build
    failure in -- the other one. Raises ``subprocess.CalledProcessError``
    on the first failure -- npm's own output goes straight to the terminal,
    so the real error is visible without extra wrapping. Stops at the first
    failing frontend rather than attempting the rest with a known-bad state."""
    dist_dirs = []
    for frontend_dir in frontend_dirs:
        subprocess.run(["npm", "ci"], cwd=frontend_dir, check=True)
        subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
        dist_dir = frontend_dir / "dist"
        # A build that "succeeds" without producing an index.html would
        # otherwise fail much later and confusingly, inside whatever test
        # or server tries to serve it.
        if not (dist_dir / "index.html").is_file():
            raise RuntimeError(f"{dist_dir} has no index.html after `npm run build` -- build produced no page")
        dist_dirs.append(dist_dir)
    return dist_dirs


if __name__ == "__main__":
    built = build_frontends()
    for dist_dir in built:
        print(f"built {dist_dir}", file=sys.stderr)
