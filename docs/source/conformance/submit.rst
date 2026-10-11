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

Keep the four files of the take together and never rename them. Open the take in the viewer of
its suite:

.. code-block:: bash

   # Hand
   src/python/isaaccapture/cts/hand/checker/.venv/bin/python -m hand_cts.panel \
       ~/isaaccapture-cts/<take>/<take>-hand.mcap
   # Full body
   src/python/isaaccapture/cts/full_body/checker/.venv/bin/python -m full_body_cts.panel \
       ~/isaaccapture-cts/<take>/<take>-body.mcap

Open http://127.0.0.1:8080 and press **package for submission** in the **Submission** folder.
The browser showing the viewer downloads ``<take>-hand.<verdict>.zip`` or
``<take>-body.<verdict>.zip``, and the viewer shows its name, size and SHA-256 prefix:

.. code-block:: text

   glovex_2026-03-04_101530-hand.pass.zip
   └── glovex_2026-03-04_101530-hand.pass/
       ├── report.json                                verdict, checks, hashes, commit
       ├── report.txt                                 the same report as text
       ├── glovex_2026-03-04_101530-hand.mcap          the recording
       ├── glovex_2026-03-04_101530-hand.labels.json   the step windows
       ├── glovex_2026-03-04_101530-hand.json          capture provenance
       └── glovex_2026-03-04_101530-hand.log           recorder log

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
   Take:     glovex_2026-03-04_101530
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
