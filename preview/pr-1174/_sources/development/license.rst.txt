.. _license:

License
========

Unless otherwise noted, the code in this repository is licensed under the
`Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

Known exceptions
----------------

- Certain headers from the `OpenXR SDK <https://github.com/KhronosGroup/OpenXR-SDK>`_ or related
  `extensions <https://github.com/NVIDIA/IsaacCapture/tree/main/deps/cloudxr/openxr_extensions>`_
  are imported from their original source and are licensed under the
  `MIT license <https://opensource.org/licenses/MIT>`_ or `Boost Software License 1.0 <https://www.boost.org/LICENSE_1_0.txt>`_.

- CloudXR SDK is NVIDIA's proprietary software, licensed under the `NVIDIA CloudXR License <https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE>`_.
- The documentation is built using `NVIDIA Sphinx Theme <https://pypi.org/project/nvidia-sphinx-theme/>`_.

License headers
---------------

We use `REUSE <https://reuse.software/>`_ to enforce SPDX license headers on source files.
`Pre-commit <https://github.com/NVIDIA/IsaacCapture/blob/main/.pre-commit-config.yaml>`_ runs
``reuse lint-file`` on changed files (e.g. ``.py``, ``.cpp``, ``.md``, ``.yaml``):

1. **Required fields**: Each file must have ``SPDX-FileCopyrightText`` and ``SPDX-License-Identifier``.
2. **License**: first-party files use ``Apache-2.0`` (full text in ``LICENSES/Apache-2.0.txt``),
   except EgoPost (``src/python/isaaccapture_egopost/``), which uses ``LicenseRef-OpenMDW-1.1``.
   Checked-in third-party files record upstream's terms instead; see *What stops the Release
   build* in :code-file:`scripts/sbom/README.md`.

Example header (for Python files):

.. code-block:: python
   :class: code-100col

   # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   # SPDX-License-Identifier: Apache-2.0

You can also run ``reuse spdx -o project.spdx`` to generate an SPDX report for the project.
