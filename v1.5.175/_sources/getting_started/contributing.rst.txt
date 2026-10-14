.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Contributing
============

We welcome contributions. Please see the repository's `CONTRIBUTING.md <https://github.com/NVIDIA/IsaacTeleop/blob/main/CONTRIBUTING.md>`_ for:

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

   https://nvidia.github.io/IsaacTeleop/preview/pr-<N>/

How the preview gets built depends on where the PR's branch lives.

PRs from a branch on ``NVIDIA/IsaacTeleop``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Build & deploy docs`` workflow runs automatically on every push to the
PR branch and publishes the preview. The preview URL is added to the workflow
run's *Summary* tab and refreshed on every push. No extra action required.

PRs from a fork
^^^^^^^^^^^^^^^

GitHub Actions on PRs from forks run with a read-only token, so the workflow
cannot push to ``gh-pages`` automatically. Instead:

1. ``Build & deploy docs`` still runs and uploads the built artifacts; only
   the deploy step is skipped.
2. When a fork PR is opened, a bot comments with instructions.
3. A maintainer (anyone with write access to ``NVIDIA/IsaacTeleop``) deploys
   the preview by commenting on the PR:

   .. code-block:: text

      /preview-docs

4. The maintainer-triggered workflow downloads the artifacts that
   ``Build & deploy docs`` already produced for that commit and pushes them
   to ``preview/pr-<N>/``, reacting to the comment with 👀 while deploying
   and 👍 once published. A follow-up comment posts the preview URL.

Because the deploy reuses the same artifacts that the build job validated,
``/preview-docs`` does not run any PR code itself. Re-comment
``/preview-docs`` after new commits land to redeploy from the latest build.

Previews are not auto-cleaned; maintainers can run the
``Cleanup docs PR previews`` workflow from the *Actions* tab to clear the
``preview/`` tree.
