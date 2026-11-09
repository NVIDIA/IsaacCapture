.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _cts-submit:

Submit Your Results
===================

This page shows how to package a checked take and send it through the submission form for your
integration pull request. You can submit any verdict, as long as its report goes with it.

.. contents:: Steps
   :local:
   :depth: 1
   :backlinks: none

1. Open your integration pull request
-------------------------------------

Open the pull request that adds your device (:doc:`/device/add_device`,
:doc:`/development/contributing`). Each form submission refers to that pull request.

2. Record and check one take per suite
--------------------------------------

Record and check one take for each suite your device supports: :doc:`hand`, :doc:`full_body`, or
both. Takes are never overwritten, so submit the one you want reviewed. Record from a commit you
have pushed to the pull request, with nothing under ``src/python/isaaccapture/cts/`` changed. The
take records the commit it was recorded at, and the report whether the checker's code was
modified; we re-run the checker at that commit.

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

The viewer serves port 8080, or the next free port; open the address it prints and press
**package for submission** in the **Submission** folder. When the archive is ready, a
notification in the viewer offers ``<take>-hand.<verdict>.zip`` or ``<take>-body.<verdict>.zip``
with its size. Click it to save the archive on the computer running the browser, or right-click it
and choose **Save link as** to pick where. Under the button, the viewer shows the archive's name,
size and the first 12 characters of its SHA-256; you quote them in step 5.

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
Submit the form once for each archive. The form asks you to sign in with a Google account.

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
   Archive:  glovex_2026-03-04_101530-hand.pass.zip, sha256 <the 12 characters the viewer shows>
   Notes:    <for fail, retake or insufficient_data: which checks failed, and why>

6. What happens next
--------------------

We re-run the checker on your take at the commit recorded with it, review the replay in the
viewer, and reply on your pull request.

.. note::

   The archive holds the performer's recorded motion and the recorder log, which shows file paths
   and other details of your workstation. The form sends it to us only; keep it out of the pull
   request.
