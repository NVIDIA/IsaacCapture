# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import isaaccapture_examples.jmex from this checkout rather than any installed
# copy, so a bare `pytest` works and the ctest entry needs no example path.
#
# The path entry is the example's namespace root, python/. Do not give
# isaaccapture_examples/ an __init__.py to make an import work -- see
# examples/README.md.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "jmex" / "python"))
