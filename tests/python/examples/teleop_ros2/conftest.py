# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve `isaaccapture_examples.teleop_ros2` against the in-tree source, so a
# bare `pytest` works and the ctest ENVIRONMENT needs no example path.
#
# These files are also copied into the teleop_ros2 container
# (Dockerfile: `COPY tests/python/examples/teleop_ros2/ tests/`), where
# tests/python/repo_paths.py does not exist and the example is already installed
# into the venv by `uv sync`. So the source-tree wiring is conditional: without
# it, importing repo_paths raises and every test errors during collection.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[2]

if (_tests_python / "repo_paths.py").is_file():
    if str(_tests_python) not in sys.path:
        sys.path.insert(0, str(_tests_python))

    from repo_paths import repo_root  # noqa: E402

    # python/, not python/isaaccapture_examples/: that is a PEP 420 namespace. Do
    # not add an __init__.py to make an import work -- it breaks the installed
    # wheel's ability to share the namespace.
    sys.path.insert(0, str(repo_root() / "examples" / "teleop_ros2" / "python"))
