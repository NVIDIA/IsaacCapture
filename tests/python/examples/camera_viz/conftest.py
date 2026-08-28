# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve `isaaccapture_examples.camera_viz` against the in-tree source, and with
# it the codec .so built in place by codec/build.sh.
#
# python/, not python/isaaccapture_examples/: that is a PEP 420 namespace. Do not
# add an __init__.py to make an import work -- it breaks the installed wheel's
# ability to share the namespace.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "camera_viz" / "python"))
