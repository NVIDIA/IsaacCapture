.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Wheel contents SBOM and license evidence
========================================

Every published ``isaaccapture`` wheel lists what it redistributes and carries the license
text behind each obligation. You can check both offline, without installing the wheel. The
Release build refuses to ship a wheel with a gap in either.

To check a wheel, see `Verifying a wheel`_. To learn how the evidence is worked out, or why
a build stopped, see the collector's developer README, :code-file:`scripts/sbom/README.md`.

What a wheel carries
--------------------

.. code-block:: text

   isaaccapture-<version>-<abi>-<platform>.whl
   └── isaaccapture-<version>.dist-info/
       ├── METADATA                       # License-Expression + one License-File per text below
       ├── licenses/
       │   ├── LICENSE.md                 # this project's own license
       │   ├── LICENSES/<id>.txt          # this project's license texts
       │   ├── THIRD-PARTY-NOTICES.md     # each component, its license, and where the text is
       │   └── third-party/<component>/   # the component's own license file, or a reference text
       └── sboms/
           └── <wheel filename>.spdx.json # SPDX 2.3 contents inventory

- **The SBOM** lists every file in the wheel but ``RECORD`` and itself: where it came from,
  under what license, and its digest. It describes the wheel's own contents only. Build
  tools, dependencies this wheel does not carry (the build evidence records them with
  ``"in_this_wheel": false``),
  install-time dependencies (``Requires-Dist`` stays authoritative; the SBOM records the
  requirement, never a license for it), source distributions and ``pip install .``,
  containers and the web client are not covered.
- **Each third-party component** ships its own license file or, for material checked into
  this repository, the reference text for its declared license beside its copyright
  notice.
- ``sboms/`` may also hold a CycloneDX document ``auditwheel`` writes for the libraries it
  bundled.

Evidence published with a release
---------------------------------

Each tagged GitHub release carries, for every wheel:

``<wheel>.spdx.json``
   An exact copy of the document embedded in the wheel.

``<wheel>.licenses.md`` / ``<wheel>.licenses.json``
   Each redistributed component, the license concluded for it and the file it was read
   from, how it reaches the wheel, and what the installation environment supplies instead.
   It covers this wheel only; what the build used but did not ship is in the build evidence.

``<wheel>.build-evidence.json``
   Resolved commits, declared constraints, archive digests and build options, from the
   build machine, and every component the build resolved, whether or not it reached the
   wheel.

and, once per release:

``isaaccapture-sbom-manifest.json``
   Each wheel's filename and SHA-256 and its evidence files' digests, recorded after the
   last change to the wheel, so a wheel touched since no longer matches.

Verifying a wheel
-----------------

You need ``uv`` and a clone of this repository:

.. code-block:: bash

   git clone --depth 1 https://github.com/NVIDIA/IsaacCapture
   uv run --project IsaacCapture/scripts/sbom isaaccapture-sbom check \
     --wheel isaaccapture-<version>-<abi>-<platform>.whl

To check a whole release, put its wheels and evidence files in a directory of their own,
and keep other wheels out of it:

.. code-block:: bash

   uv run --project IsaacCapture/scripts/sbom isaaccapture-sbom check-set \
     --manifest release/isaaccapture-sbom-manifest.json \
     --wheel-dir release --evidence-dir release

``check`` confirms the SBOM accounts for every file, its digests match the wheel and its
``RECORD``, each license text it cites is packaged where it says, and every
``LicenseRef-`` carries its text; add ``--build-evidence`` or ``--manifest`` to check
against the release evidence too.

The SBOM describes a wheel; it is not a signature. Integrity is the wheel's ``RECORD``,
which installers check. These releases carry no
`PEP 740 <https://peps.python.org/pep-0740/>`_ attestations.

To read the inventory, open ``.dist-info/sboms/*.spdx.json`` with any SPDX 2.3 tool. A
file's ``SPDXID`` changes only when its path does, so two releases can be diffed.
