<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Capture examples

Each subdirectory is a self-contained example. Read one, run it, or copy it into
your own project — the last of those is the reason for the layout below.

## Running an example

Build Isaac Capture from source, then set up a venv from the repository root
exactly as in [step 4 of Build from source][install-step]: create it on the
build's Python, activate it, and install `isaaccapture` pinned to `X.Y+local`
with `--reinstall-package isaaccapture`. Rerun that install after every
rebuild to replace the installed wheel. `uv pip install` finds a repo-root
`.venv` even when it is inactive and `python` does not, so activate it in every
terminal you run an example from.

```bash
uv pip install -e ./examples/<name>
python -m isaaccapture_examples.<name>
```

Every example runs this way except camera_viz and vehicle_teleop, whose
wrapper scripts call `.venv/bin/python` inside the example directory; their
READMEs set up that venv. An example whose `pyproject.toml` points
`[tool.uv.sources]` at this checkout installs an editable build instead of the
wheel; [step 4][editable-examples] says how to rebuild it.

[install-step]: https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html#install-the-isaaccapture-pip-package
[editable-examples]: https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html#build-from-source-editable-examples

An example with one obvious entry point puts it in `__main__.py`, so
`python -m isaaccapture_examples.<name>` runs it. An example that is several
co-equal demos has no `__main__.py` at all — each is a submodule, run as
`python -m isaaccapture_examples.<name>.<mod>`, and the README lists them.
Promoting one of several peers to the default only makes it look privileged.

## Layout

```
examples/<name>/
├── README.md            — what it does, and the command to run it
├── pyproject.toml       — the distribution, named isaaccapture-examples-<name>
└── python/
    └── isaaccapture_examples/
        └── <name>/
            ├── __init__.py
            ├── __main__.py   — only if there is one obvious entry point
            └── ...
```

`pyproject.toml` sits at the example root, so `uv pip install ./examples/<name>`
works for every example without knowing anything about its internals.

`python/` is the namespace root. **`isaaccapture_examples/` has no
`__init__.py`** and must never get one: it is a [PEP 420][pep420] namespace
shared by every example distribution, and giving one distribution ownership of
it makes the others collide or vanish when two are installed together.

[pep420]: https://peps.python.org/pep-0420/

## Importing within an example

Modules in the same example import each other **relatively**:

```python
from .pipeline import Frame        # yes
from pipeline import Frame         # no
```

A bare `from pipeline import Frame` resolves only because the directory of the
script you invoked lands on `sys.path`. It breaks under `python -m`, breaks the
moment someone copies the file into a package of their own, and claims a
generic top-level name — `pipeline`, `common`, `messages` — for the whole
interpreter. Do not paper over it with a `try: from .x / except ImportError:
from x` guard; that doubles every import site to tolerate an invocation that
should simply not be used.

## What this does not cover

An example with no importable Python is left alone. `rebot` is the case: its
only `.py` is a root-run `/dev/mem` helper that a systemd unit installs to
`/opt/` and executes with the system interpreter, and its `pyproject.toml`
exists to resolve a CLI dependency. Packaging that would break the thing that
runs it. The rule below is about examples you `import`.

## Adding an example

Copy the layout above and match an existing example — `deviceio_live_view` is
the smallest complete one.

- Name the distribution `isaaccapture-examples-<name>`, mirroring the import
  path, so nothing claims a bare top-level name in `site-packages`.
- With hatchling, use `only-include` + `sources`, **not** `packages`:

  ```toml
  [tool.hatch.build.targets.wheel]
  only-include = ["python/isaaccapture_examples/<name>"]
  sources = ["python"]
  ```

  `packages` keeps only the last path component, so it builds without complaint
  and produces a wheel rooted at a bare top-level `<name>/`.
- Keep `[tool.uv.sources]` last in the file. `install_python_example()` strips
  it from the installed copy by matching up to the next `[`, so a block after it
  loses its comments.
- If the example ships in the install tree, add
  `install_python_example(DESTINATION examples/<name>)` to its `CMakeLists.txt`.
