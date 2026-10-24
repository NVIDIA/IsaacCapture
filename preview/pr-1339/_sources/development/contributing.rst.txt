.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Contributing
============

We welcome contributions. Please see the repository's `CONTRIBUTING.md <https://github.com/NVIDIA/IsaacCapture/blob/main/CONTRIBUTING.md>`_ for:

- Code of conduct and how to contribute
- Development setup and coding standards
- Pull request process

Previewing documentation changes
--------------------------------

Local build
~~~~~~~~~~~

Build the docs locally to catch broken links and rendering issues before opening
a pull request:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make current-docs

The output is written to ``docs/build/current/``. Open ``index.html`` in a
browser to inspect it. Sphinx is run with ``-W --keep-going``, so warnings are
treated as errors — fix them locally before pushing.

PR preview on GitHub Pages
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every PR preview is published to a single canonical location:

.. code-block:: text

   https://nvidia.github.io/IsaacCapture/preview/pr-<N>/

PRs changing ``docs/`` publish automatically after a successful build, including
PRs from forks. GitHub may first require a maintainer to approve a fork's workflow
run. The PR does not need to be merged.

``Build docs and deploy production`` builds the PR changes without publishing
privileges. ``Publish docs PR preview`` then uses trusted code from ``main`` to
validate and publish that run's artifacts. A single bot comment links to the
preview and is updated after subsequent successful builds.

Same-repository web-client-only PRs retain automatic previews at
``preview/pr-<N>/client/``. Fork PRs without docs changes require a maintainer retry
for a web-client preview.

A maintainer with write access can retry publication of the current successful
build by commenting:

.. code-block:: text

   /preview-docs

This command can publish docs or web-client artifacts, including for PRs without
``docs/`` changes. It reuses built artifacts and never executes PR code in the
publisher. The command receives 👀 while deploying and 👍 after publication.

Previews are removed when PRs close; a scheduled sweep also removes stale previews.
