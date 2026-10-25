<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Thanks for your interest in contributing! This document outlines how to build, test, and submit changes.

## Code Style

- Use meaningful, descriptive names for variables, functions, and types in all languages.
- Write short docstrings for public modules, classes, and functions.
- Write clear, easy-to-read, and maintainable code.
- Keep code warnings and linter errors to a minimum.
- In general, prefer clarity over clever tricks, and keep the codebase friendly for contributors.

**C++**
- Use C++20.
- Format C++ with `clang-format`.

**Python**
- Follow PEP 8 for style.
- Use type annotations (PEP 484) and prefer formatted string literals (PEP 701, f-strings).

## License

- Contributions use the license of the component/file being changed: Apache-2.0 by default, OpenMDW-1.1 for EgoPost, with explicit third-party exceptions preserved. The DCO below applies to each contribution.

## Pre-commit

We use [pre-commit](https://pre-commit.com/) to run Python linters and formatters (including
[Ruff](https://docs.astral.sh/ruff/)) before commits. If you are using uv like us, install the
tools and hooks as follows:

```bash
uv tool install ruff
uv tool install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

After that, pre-commit runs automatically on `git commit`. To run on all files manually:

```bash
SKIP=check-copyright-year pre-commit run --all-files
```

To run only Ruff (check + fix and format):

```bash
ruff check . --fix
ruff format .
```

### License headers

Pre-commit automatically checks and updates license headers with [REUSE](https://reuse.software/).
It adds NVIDIA headers to unannotated first-party files and selects the license by
file location, preserving existing attribution and explicit third-party exceptions.
Review and stage any automatic edits, then commit again.

## CI
GitHub Actions builds the default targets and runs tests.

## Pull Requests
1. Create a feature branch
2. Ensure builds pass locally and in CI
3. Describe motivation, changes, and testing in the PR
4. Link related issues

### Code ownership

[`.github/CODEOWNERS`](.github/CODEOWNERS) maps subsystems to reviewers, with
repository maintainers covering unmatched paths. Keep related source, bindings,
and tests assigned together when adding or moving files. All files under `docs/`
share the documentation owners; plugin infrastructure, integrations, and tests
share the plugin owners. Examples have their own entries. Rules are ordered
from broad defaults to specific overrides; the last matching rule wins.

Owners must have repository write access. Update the map as maintainers change,
keeping a backup reviewer for each area.
Multiple owners on a line are alternatives, not a requirement for both to approve.

GitHub requests owner reviews for ready pull requests once the file is on their
base branch. Requiring owner approval before merge is a separate repository
ruleset or branch-protection setting; adding this file does not enable it.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
