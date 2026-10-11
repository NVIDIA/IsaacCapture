.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _cts-submit:

.. rst-class:: cts-page

Submit Your Results
===================

This page shows how to package a checked take and submit it with your integration pull request.
You can submit any verdict, as long as its report goes with it.

.. contents:: Steps
   :local:
   :depth: 1
   :backlinks: none

1. Open your integration pull request
-------------------------------------

Open the pull request that adds your device (:doc:`/device/add_device`,
:doc:`/getting_started/contributing`). The take goes with that pull request.

2. Record and check one take per suite
--------------------------------------

Record and check one take for each suite your device supports: :doc:`hand`, :doc:`full_body`, or
both. Takes are never overwritten, so submit the one you want reviewed.

3. Package the take
-------------------

Keep the four files of the take together and never rename them.

**Full body.** Open the take in the viewer:

.. code-block:: bash

   src/python/isaaccapture/cts/full_body/checker/.venv/bin/python -m full_body_cts.panel \
       ~/isaaccapture-captures/<take>/<take>-body.mcap

Open http://127.0.0.1:8080 and press **package for submission** in the **Submission** folder.
The browser showing the panel downloads ``<take>-body.<verdict>.zip``, and the panel shows its name,
size and SHA-256 prefix:

.. code-block:: text

   pico4u_2026-03-04_101530-body.retake.zip
   └── pico4u_2026-03-04_101530-body.retake/
       ├── report.json                                verdict, checks, hashes, commit
       ├── report.txt                                 the same report as text
       ├── pico4u_2026-03-04_101530-body.mcap         the recording
       ├── pico4u_2026-03-04_101530-body.labels.json  the step windows
       ├── pico4u_2026-03-04_101530-body.json         capture provenance
       └── pico4u_2026-03-04_101530-body.log          recorder log

**Hand.** There is no package button. From the repository root, write both reports into the take
directory with the checker, then archive the directory:

.. code-block:: bash

   TAKE=~/isaaccapture-captures/<take>
   NAME=$(basename "$TAKE")
   CHECK=src/python/isaaccapture/cts/hand/checker/.venv/bin/python
   "$CHECK" -m hand_cts.cli "$TAKE/$NAME-hand.mcap"        > "$TAKE/report.txt"
   "$CHECK" -m hand_cts.cli "$TAKE/$NAME-hand.mcap" --json > "$TAKE/report.json"
   VERDICT=$(awk '$1 == "verdict" {print tolower($2)}' "$TAKE/report.txt")
   tar czf "$TAKE-hand.$VERDICT.tar.gz" -C "$(dirname "$TAKE")" "$NAME"

The archive lands next to the take directory:

.. code-block:: text

   manus_2026-03-04_101530-hand.pass.tar.gz
   └── manus_2026-03-04_101530/
       ├── manus_2026-03-04_101530-hand.mcap
       ├── manus_2026-03-04_101530-hand.labels.json
       ├── manus_2026-03-04_101530-hand.json
       ├── manus_2026-03-04_101530-hand.log
       ├── report.json
       └── report.txt

4. Submit the form
------------------

Fill in the submission form with the link to your pull request and upload the archive from step 3.
The form asks you to sign in with a Google account.

.. button-link:: https://forms.gle/hLqAhNkae4bhAUX79
   :color: primary

   Open the submission form

5. Describe the take in your pull request
-----------------------------------------

Add one block per suite to the pull request description:

.. code-block:: text

   Device conformance
   Suite:    hand
   Take:     manus_2026-03-04_101530
   Verdict:  pass
   Notes:    <for fail, retake or insufficient_data: which checks failed, and why>

6. What happens next
--------------------

We re-run the checker on your take at the commit recorded with it, review the replay in the
viewer, and reply on your pull request.

.. note::

   The archive holds the performer's recorded motion and the recorder log, which shows file paths
   and other details of your workstation. The form sends it to us only; keep it out of the pull
   request.
