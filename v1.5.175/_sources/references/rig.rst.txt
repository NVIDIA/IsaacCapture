.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _rig-launcher:

Rig Launcher
============

A typical Isaac Teleop setup is one or more **producer** plugins that publish
device data, and one or
more **consumer** apps (a Python ``TeleopSession`` script or a C++ example
binary) that read the streams. Such a configured set is a *rig* — the same
shape serves demos, production teleop, and data collection.
``isaacteleop.rig`` starts a rig in a single tmux window from a small YAML
file, instead of you juggling three terminals by hand:

.. code-block:: bash

   # from the Teleop repository root
   python -m isaacteleop.rig rigs/se3_tracker.yaml

The module ships in the ``isaacteleop`` wheel; the rig files are part of the
source checkout under ``rigs/`` (they reference ``install/`` binaries, so they
only make sense next to a built tree).

.. contents:: Sections
   :local:
   :depth: 1
   :backlinks: none

Prerequisites
-------------

- ``tmux`` installed (``sudo apt install tmux``).
- A built and installed Teleop tree (see :ref:`install-isaacteleop-pip-package`
  and the build reference) — the rigs reach their binaries through
  ``{install}/plugins/`` and ``{install}/examples/`` (see
  `The install prefix`_).
- The ``isaacteleop`` wheel installed in the **current** Python environment.
  tmux panes do not inherit your venv; the launcher bakes the absolute path of
  its own interpreter (and your ``PYTHONPATH``, if set) into the pane commands,
  so whatever environment you launch from is the one the rig runs in.

Run a rig
---------

.. code-block:: bash

   # from the Teleop repository root
   python -m isaacteleop.rig rigs/se3_tracker.yaml

What happens:

1. **Preflight** — the launcher verifies tmux is available, the referenced
   binaries exist under the resolved install prefix and are executable (naming
   the exact remedy if not — see `The install prefix`_),
   and the interpreter can import ``isaacteleop.cloudxr``. Nothing is created
   until preflight passes.
2. **A CloudXR runtime is ensured.** The runtime is a **host singleton**
   owned by the CloudXR service, not by a rig pane: the launcher attaches to
   the one already serving, or starts a detached service when none is (see
   :ref:`dedicated-cloudxr-runtime`). Either way it outlives the rig, so
   killing the rig leaves the headset connected.
3. **Worker panes load the CloudXR environment automatically.** Each
   producer/consumer pane runs ``source <install-dir>/run/cloudxr.env`` so
   ``XR_RUNTIME_JSON`` and friends point at the service's runtime — you never
   source it by hand. There is nothing to wait for: a pane only starts once
   the runtime is serving, so the file already exists. The install dir
   follows ``CXR_INSTALL_DIR`` (default ``~/.cloudxr``).
4. **Producer/consumer panes then run their commands automatically.** As
   soon as the CloudXR environment is loaded, each pane prints a banner
   (``[producer: ...] running: <command>``) and runs its command — no
   :kbd:`Enter` needed. When the command exits, the pane reports
   ``[rig] command exited with status N — press Enter to rerun`` and drops
   to an interactive shell with the same command pre-typed at the prompt.
   An app that reaches ``xrGetSystem`` before a headset connects waits there
   (``waiting for a system...``) and carries on by itself once you connect
   one to the printed URL — no rerun needed. If the CloudXR environment fails to
   load, the pane does *not* run the command; it prints a remedy and leaves
   the command pre-typed instead.
5. The launcher then switches you to the rig window. Run from inside tmux,
   the rig is a **new window in your current session** — your other windows
   stay put and nothing nests. Run from a plain shell, it gets a session of
   its own (named after the rig) and the launcher attaches to it.

Re-running the same rig just switches to the running one — in whichever
session it lives; it does **not** pick up edits
to the rig file. Start over with:

.. code-block:: bash

   python -m isaacteleop.rig rigs/se3_tracker.yaml --kill

which kills the rig's tmux window and every process in it (equivalent to
``tmux kill-window -t se3_tracker``, without needing to know which session
holds it). Only the rig's window: a session you launched the rig into keeps
its other windows. Killing a rig that is not running is a no-op.

The install prefix
------------------

Rig commands reference binaries as ``{install}/plugins/...`` and
``{install}/examples/...``. ``{install}`` resolves at launch time to:

1. ``$ISAAC_TELEOP_INSTALL_DIR``, if set — the knob for a tree installed with a
   non-default ``CMAKE_INSTALL_PREFIX``;
2. otherwise ``<cwd>/install``, the prefix this project's CMakeLists forces by
   default (``cwd`` is the rig's, so for the shipped rigs that is
   ``<repo>/install``).

.. code-block:: bash

   # a tree installed elsewhere, e.g. cmake --install build --prefix /opt/isaacteleop/install
   ISAAC_TELEOP_INSTALL_DIR=/opt/isaacteleop/install \
       python -m isaacteleop.rig rigs/se3_tracker.yaml

A binary that is missing under the resolved prefix names the fix for the case
you are actually in: a build tree that was configured but never installed gets
the one ``cmake --install ... --prefix ...`` command (its own
``CMAKE_INSTALL_PREFIX`` may point anywhere, so the prefix is passed
explicitly); a set ``ISAAC_TELEOP_INSTALL_DIR`` that resolves to the wrong tree
says so rather than telling you to rebuild; and every message without the
variable set mentions it, since an install tree you already have is as likely
as an unbuilt one.

The prefix is made absolute before it reaches a pane, and quoted, so a path
containing spaces works. A rig that spells the path out literally
(``install/plugins/...``) still works whenever the default prefix is the right
one — but it cannot follow the variable, which is the point of the placeholder.

The rig YAML
------------

``rigs/se3_tracker.yaml`` is the annotated exemplar — copy it to write your
own:

.. code-block:: yaml

   name: se3_tracker              # rig id AND tmux window name (letters/digits/-/_)
   description: CloudXR runtime + SE3 controller tracker plugin + pose printer
   cwd: ..                        # pane working dir, relative to this file
   params:                        # shared values, substituted into the commands below
     hand: right
     collection_id: se3_tracker   # defined ONCE, referenced by both sides below
   producers:                     # publish device data into the runtime
     - name: se3 tracker plugin (requires headset + controller)
       command: "{install}/plugins/controller_se3_tracker/controller_se3_tracker_plugin {hand} {collection_id}"
   consumers:                     # read the streams — a TeleopSession script or a C++ binary
     - name: se3 printer (requires headset)
       command: "{install}/examples/schemaio/se3_printer {collection_id}"

``rigs/full_body.yaml`` shows the other supported shape — no ``producers``
key, because its consumers read the tracking data directly from the runtime,
so there is no ``collection_id`` to rendezvous on.

Top-level keys:

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Key
     - Required
     - Meaning
   * - ``name``
     - yes
     - Rig id **and** tmux window name. Letters, digits, ``-``, ``_`` only.
   * - ``description``
     - no
     - Free text, printed when the rig is created.
   * - ``cwd``
     - no
     - Working directory for every pane and the base for relative command
       paths, resolved relative to the YAML file's directory (default: the
       YAML's directory).
   * - ``params``
     - no
     - Flat mapping of ``{placeholder}`` values shared by the commands
       below — the rig file is the single source of configuration; edit it
       (and relaunch with ``--kill``) to change them.
   * - ``producers`` / ``consumers``
     - at least one entry total
     - Lists of ``{name, command}`` entries. ``name`` is free text shown in
       the pane title (a good place for hardware prerequisites); ``command``
       is a shell string run verbatim in the pane.

Commands are plain shell strings. Two placeholders are reserved: ``{python}``
expands to the absolute path of the launching interpreter and ``{install}`` to
the install prefix (see `The install prefix`_). Every other ``{placeholder}``
must be declared under ``params`` (literal braces are written ``{{`` / ``}}``);
``python`` and ``install`` are rejected as param names.
Unknown top-level keys, unknown entry keys, and unknown placeholders are hard
errors — a typo fails loudly at load time instead of misbehaving in a pane.

.. important::

   In a rig with both producers and consumers, the two sides rendezvous on a
   shared ``collection_id``, and a mismatch is **silent no-data** by design.
   Define it once under ``params`` and reference it as ``{collection_id}`` in
   every command — then one edit in one place changes both sides together.

.. note::

   The CloudXR runtime is a service, not a rig pane. The rig makes sure one
   is serving before any pane starts, and every pane sources its environment,
   so Python consumers attach to that same runtime instead of starting their
   own.

   See :ref:`dedicated-cloudxr-runtime` for managing that service yourself.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Likely cause
   * - ``... not found or not executable — build and install first``
     - The rig references ``install/`` binaries that are not built yet; run
       the printed ``cmake`` commands.
   * - ``cannot import 'isaacteleop.cloudxr'``
     - You launched from an environment without the ``isaacteleop`` wheel;
       activate the right venv (or ``pip install install/wheels/isaacteleop-*.whl``).
   * - ``no CloudXR runtime for rig '...'``
     - Preflight could not attach to a runtime or start a service. Check
       what is serving with ``python -m isaacteleop.cloudxr.service status``.
   * - ``[cloudxr] loading ... failed``
     - The runtime is serving but its ``cloudxr.env`` could not be sourced;
       the pane does not run its command. Check the error printed above
       the message, ``source`` the file as the message says, then press
       :kbd:`Enter` — the command is already pre-typed.
   * - A pane sits at ``waiting for a system...``
     - Normal until a headset connects to the runtime's URL; the app goes on
       by itself once one does. Nothing times this out, so a pane that never
       moves is usually a device-profile mismatch — check it with
       ``python -m isaacteleop.cloudxr.service status``.
   * - ``[rig] command exited with status ...`` right after launch
     - The command failed on its own account; a late headset is no longer
       such an exit. Read the pane's output, then press :kbd:`Enter` to
       rerun — the command is already pre-typed at the prompt.
   * - Edits to the rig file seem ignored
     - The rig was already running; relaunch after
       ``python -m isaacteleop.rig <rig.yaml> --kill``.
