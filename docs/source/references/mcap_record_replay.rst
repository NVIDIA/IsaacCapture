.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

MCAP Recording & Replay
========================

Isaac Capture supports recording live tracking data to `MCAP <https://mcap.dev/>`_
files and replaying them offline through the same retargeting pipeline — no
headset or OpenXR runtime required during replay.

Recording a Live Session
-------------------------

Pass an ``McapRecordingConfig`` in the ``mcap_config`` field while keeping the
default ``SessionMode.LIVE``.  ``TeleopSession`` automatically discovers
trackers from the pipeline, so you only need to provide the output filename:

.. code-block:: python

   from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig
   from isaaccapture.deviceio import McapRecordingConfig

   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       mcap_config=McapRecordingConfig("recording.mcap"),
   )

   with TeleopSession(config) as session:
       while True:
           result = session.step() # The tracker state is recorded to the MCAP file

When the context manager exits, the MCAP file is finalized and closed.

Auto-population and appending of tracker names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TeleopSession`` always auto-discovers trackers from the pipeline's DeviceIO
sources and uses each source's ``name`` as the MCAP channel name.  For
example, a pipeline with ``HeadSource(name="head")`` and
``HandsSource(name="hands")`` produces channels ``"head"`` and ``"hands"``.

Any ``(tracker, channel_name)`` pairs you pass explicitly via
``tracker_names`` are **appended** after the auto-discovered sources.  This is
useful for recording additional trackers that are not part of the pipeline:

.. code-block:: python

   from isaaccapture.deviceio import McapRecordingConfig

   extra_tracker = deviceio.HandTracker()

   mcap_config = McapRecordingConfig(
       "recording.mcap",
       [(extra_tracker, "extra_hands")],
   )

   # Final tracker list will be:
   #   [(head_source.tracker, "head"), (hands_source.tracker, "hands"),
   #    (extra_tracker, "extra_hands")]

Replaying a Recording
---------------------

Set ``mode=SessionMode.REPLAY`` and pass an ``McapReplayConfig``.  No OpenXR
session or headset connection is created — data is read directly from the MCAP
file:

.. code-block:: python

   from isaaccapture.teleop_session_manager import (
       TeleopSession,
       TeleopSessionConfig,
       SessionMode,
   )
   from isaaccapture.deviceio import McapReplayConfig

   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       mode=SessionMode.REPLAY,
       mcap_config=McapReplayConfig("recording.mcap"),
   )

   with TeleopSession(config) as session:
       while True:
           result = session.step()
           action = result["action"]
           # Process replayed action ...

Just like recording, trackers are auto-discovered from the pipeline and any
explicit ``tracker_names`` you provide are **appended** after them.  Use this
to add extra trackers that aren't part of the pipeline:

.. code-block:: python

   McapReplayConfig(
       "recording.mcap",
       [(extra_tracker, "extra_hands")],
   )

.. important::

   ``mcap_config`` is **required** when ``mode`` is ``SessionMode.REPLAY``.
   Omitting it raises ``ValueError``.

Schema compatibility
^^^^^^^^^^^^^^^^^^^^

Every recording embeds the FlatBuffer schema it was written under. Opening a replay
session compares every schema the recording carries against the schema the running
build was compiled from, and raises ``RuntimeError`` when the recorded bytes would be
misread — a recording made before a breaking schema change, for instance. The whole
recording is graded before the first frame is read, so replay never stops part-way
through for a schema reason.

That needs the list of schemas up front. It normally comes from the summary section at
the end of the file; a recording whose writer never got to flush one, which is what a
run cut short leaves behind, is scanned for its schema records instead. A recording so
damaged that neither route works is refused.

A difference that leaves every recorded field readable — a table field appended
since, or one marked ``(deprecated)`` — is reported on stderr and replayed instead.
There is no way to override the raise: a recording the running build would misread
is not replayable, and reading it anyway would hand back records whose field values
are quietly wrong. ``src/core/schema/README.md`` has the evolution rules that keep a
schema change from getting there.

Runnable Example
----------------

A complete record / replay example lives at
``examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/``:

- ``common.py`` — pipeline builders (``build_hand_pipeline()``,
  ``build_controller_pipeline()``, ``build_full_body_pipeline()``) plus the
  ``HandJoints`` retargeter and the bone tables used by the replay
  visualizers.
- ``record_hand.py`` / ``replay_hand.py`` — records the ``hands`` channel
  from a live OpenXR session and replays it with a `viser
  <https://viser.studio/>`_ visualization of both hand skeletons (joint cloud
  + skeleton).
- ``record_controller.py`` / ``replay_controller.py`` — records the
  ``controllers`` channel and replays it in viser, including a per-controller
  HUD (thumbstick, trigger, squeeze, button states).
- ``live_full_body.py`` — live viser preview of the full-body skeleton from
  an active OpenXR session.
- ``record_full_body.py`` / ``replay_full_body.py`` — records the
  ``full_body`` + ``controllers`` channels and replays the body skeleton in
  viser.

Each pipeline wires only the ``Source`` nodes it needs, so the resulting MCAP
contains exactly the matching channels.  To capture more data, add
additional source nodes (``HeadSource``, ``ControllersSource``, …) in
``common.py``.

For a live browser view of **all** human DeviceIO trackers at once (hands, head,
controllers, and full body), see ``examples/deviceio_live_view/`` and its
``README.md``:

.. code-block:: bash

   uv pip install -e ./examples/deviceio_live_view
   python -m isaaccapture_examples.deviceio_live_view --accept-eula

A C++ recorder lives at ``examples/mcap_record_replay/cpp/``:

- ``record_full_body.cpp`` — records the ``full_body`` channel by passing a
  ``core::McapRecordingConfig`` to ``DeviceIOSession::run()``. It uses the
  same channel base name as ``FullBodySource``, so the resulting file replays
  with ``replay_full_body``.

Live preview
^^^^^^^^^^^^

From the repo root:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   python -m isaaccapture_examples.mcap_record_replay.live_full_body --accept-eula
   python -m isaaccapture_examples.mcap_record_replay.live_full_body --port 8090 --accept-eula

Open the printed URL (default ``http://localhost:8080``) in a browser.  The
viewers bind every interface, since they run where the headset is and get
opened from another machine; pass ``--host 127.0.0.1`` to keep one local.

Recording
^^^^^^^^^

From the repo root:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   R="python -m isaaccapture_examples.mcap_record_replay.record_hand"
   $R              # 5 s → ./recordings/hands_<timestamp>.mcap
   $R 10           # record for 10 seconds
   $R 10 out.mcap  # custom output path

Recordings are written to ``./recordings/`` relative to where you run the
command, and the replay scripts look there when given no path.

The example never downloads a published wheel — it runs against the
``isaaccapture`` next to it, built from this checkout.  The first install
compiles the extension modules and takes a few minutes; later ones reuse the
cached build, and the install is editable, so edits under ``src/python/`` need
no rebuild.  From ``install/examples/mcap_record_replay`` (after ``cmake
--install``) it picks up the wheel you just built instead — see
:doc:`/development/build_from_source/index`.

An active OpenXR runtime / headset must be connected, just like any other
live ``TeleopSession``.  Recording also needs the CloudXR runtime natives, which
a from-source build bundles only when the CloudXR SDK tarball is available —
CMake fetches it automatically, and warns rather than fails if it cannot.  Replay
has no such requirement.

Replaying
^^^^^^^^^

Replay repeats until Ctrl+C — no headset required:

.. code-block:: bash

   R="python -m isaaccapture_examples.mcap_record_replay.replay_hand"
   $R                    # newest hands_*.mcap under ./recordings/
   $R path/to/file.mcap  # explicit file
   $R --port 8090        # change viser port

Open the printed URL (default ``http://localhost:8080``) in a browser to see the
left (green) and right (blue) hand skeletons update each frame.

The ``record_controller`` / ``replay_controller`` and ``record_full_body`` /
``replay_full_body`` module pairs use the same CLI (positional MCAP path,
``--host``, ``--port``). ``live_full_body`` accepts ``--host``,
``--port``, and the CloudXR launcher flags (including ``--accept-eula``).

``uv run`` (instead of an activated venv) needs an explicit ``--python 3.11``
— without it, ``uv`` may resolve against the system Python and fail to match
the lockfile.

SOMA body workflows
^^^^^^^^^^^^^^^^^^^

The body commands accept ``--body-schema full-body-pose`` (default) or
``--body-schema soma``. SOMA mode supports both ``SomaBodyJointRotations`` and
``SomaBodyJointPoses``. They render the native 77-joint body topology, including
available finger joints, without converting to ``FullBodyPose``. See
:doc:`/device/soma` for the transport contract.

FullBodyPose mode includes controllers by default. SOMA mode uses only the
body stream, including available finger joints. Pass ``--controllers`` to
include controller recording, or ``--no-controllers`` to disable it for
FullBodyPose. Standalone hands use the separate hand commands; these examples
do not extract or fuse them with the body.

Install the example and demo publisher into a configured development checkout:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   uv pip install -e ./examples/soma_body_publisher

Start the live viewer and connect a CloudXR client. Live preview does not read
or write an MCAP file:

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.live_full_body \
     --accept-eula --body-schema soma --soma-body-representation joint-rotations

In a second terminal, start the existing publisher after the runtime is running:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   uv run --no-sync python -m isaaccapture_examples.soma_body_publisher \
     --body-representation joint-rotations --loop

Record in another terminal, then stop the live viewer and publisher before
replaying on the same Viser port:

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.record_full_body \
     10 recordings/soma_body_joint-rotations_demo.mcap \
     --accept-eula --body-schema soma --soma-body-representation joint-rotations

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.replay_full_body \
     recordings/soma_body_joint-rotations_demo.mcap \
     --body-schema soma --soma-body-representation joint-rotations

For evaluated poses, use ``--soma-body-representation joint-poses`` on the
viewer, recorder, and replay commands, and ``--body-representation joint-poses``
on the publisher. Use a separate output file. Replay works without OpenXR.
As with the other viewers, ``--host`` and ``--port`` control browser access.

The recorder stores received SOMA payloads unchanged, without FK. Viewing joint
rotations uses ``SomaBodySource`` and its existing FK helper against a prepared
neutral model. Viewing joint poses maps directly without loading SOMA-X assets
or running FK. Unavailable joints and connecting bones are hidden.

``--soma-body-collection-id`` selects the publisher collection and MCAP channel
base. Its default, ``soma_body_demo``, matches the publisher's ``--collection-id``.
Replay reads
``<collection>/soma_body_joint_rotations_tracked`` or
``<collection>/soma_body_joint_poses_tracked``. If no file is supplied, it selects
the newest ``soma_body_<representation>_*.mcap`` under ``./recordings/``.

Native replay still advances one tracked sample per update. The viewer uses the
selected channel's average capture rate for pacing; ``--rate`` overrides it
(``0`` is unpaced). It does not resample or synchronize streams. FullBodyPose
and OpenXR hand replay retain their 60 Hz, duration-based playback.

SOMA hand workflows
^^^^^^^^^^^^^^^^^^^

The hand commands accept ``--hand-schema openxr-hand-pose`` (default) or
``--hand-schema soma``. SOMA mode supports ``SomaHandJointRotations`` and
``SomaHandJointPoses``. Each left/right collection retains its anatomical
``hand_side`` and uses the native 25-joint Wrist/finger layout. These commands
do not fuse standalone hands with a body stream.

Install the MCAP example and the existing hand publisher:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   uv pip install -e ./examples/soma_hand_publisher

Start the viewer and connect a CloudXR client:

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.live_hand \
     --accept-eula --hand-schema soma --soma-hand-representation joint-rotations

After the runtime is running, start the publisher in a second terminal:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   uv run --no-sync python -m isaaccapture_examples.soma_hand_publisher \
     --hand-representation joint-rotations --loop

Record in another terminal, then stop the viewer and publisher before replaying
on the same Viser port:

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.record_hand \
     10 recordings/soma_hand_joint-rotations_demo.mcap \
     --accept-eula --hand-schema soma --soma-hand-representation joint-rotations

   uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.replay_hand \
     recordings/soma_hand_joint-rotations_demo.mcap \
     --hand-schema soma --soma-hand-representation joint-rotations

For evaluated poses, use ``--soma-hand-representation joint-poses`` on the
viewer, recorder, and replay commands, and ``--hand-representation joint-poses``
on the publisher. Use a separate output file. Live preview does not read or
write MCAP; replay does not require OpenXR. ``--host`` and ``--port`` control
browser access.

Recording preserves received payloads without FK. Viewing rotations uses
``SomaHandSource`` FK against prepared neutral models. Joint poses map directly
without loading models. The viewer colors left green and right blue and hides
unavailable joints and bones.

``--soma-left-hand-collection-id`` and ``--soma-right-hand-collection-id`` default
to ``soma_hand_left_demo`` and ``soma_hand_right_demo``, matching the publisher's
``--left-collection-id`` and ``--right-collection-id``. Replay reads each
collection's ``soma_hand_joint_rotations_tracked`` or
``soma_hand_joint_poses_tracked`` channel. With no file argument, it selects the
newest ``soma_hand_<representation>_*.mcap`` under ``./recordings/``.

The trackers retain independent, one-sample-per-update replay. Pacing defaults
to the busier selected hand's average capture rate; ``--rate`` overrides it.
Missing or exhausted hands are hidden, and playback repeats after the longer
channel finishes. No timestamp synchronization or resampling is added.

API Reference
-------------

``SessionMode``
^^^^^^^^^^^^^^^

.. code-block:: python

   class SessionMode(Enum):
       LIVE = "live"
       REPLAY = "replay"

Determines whether ``TeleopSession`` creates a live OpenXR session or replays
from an MCAP file.

``McapRecordingConfig``
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   McapRecordingConfig(filename, tracker_names=None)

- **filename** — Path to the output MCAP file.
- **tracker_names** — Optional list of ``(tracker, channel_name)`` pairs.
  When empty, ``TeleopSession`` auto-populates from discovered sources.

``McapReplayConfig``
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   McapReplayConfig(filename, tracker_names=None)

- **filename** — Path to an existing MCAP file to replay.
- **tracker_names** — Optional list of ``(tracker, channel_name)`` pairs.
  When empty, ``TeleopSession`` auto-populates from discovered sources.
