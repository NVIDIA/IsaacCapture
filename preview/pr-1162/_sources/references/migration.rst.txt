.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Migrating to ``isaaccapture``
=============================

In 1.6 the Python import package and the distribution were both renamed from
``isaacteleop`` to ``isaaccapture``. ``import isaacteleop`` keeps working for now
through an alias shipped inside the ``isaaccapture`` wheel, which **will be
removed in 1.9**.

Depend on ``isaaccapture`` and pin ``isaaccapture<1.9`` if you need the alias
to stay available while you migrate imports.

In short
--------

#. Upgrade with ``pip install -U isaaccapture[...]`` -- see
   :ref:`migration-upgrading`.
#. Rewrite ``isaacteleop`` to ``isaaccapture`` in imports, ``python -m`` commands
   and dependency pins; ``isaacteleop_examples`` to ``isaaccapture_examples``.

.. _migration-upgrading:

Upgrading an existing install
-----------------------------

Installing ``isaaccapture`` replaces the old ``isaacteleop`` distribution with
a small transition package that contains no code. Pip removes the legacy
implementation automatically:

.. code-block:: console

   $ pip install -U "isaaccapture[cloudxr]"

The same command works on a fresh environment. Both import names resolve to
``isaaccapture``. Existing ``pip install -U isaacteleop[...]`` commands also
install ``isaaccapture`` and forward the requested extras.

Imports
-------

.. code-block:: diff

   -import isaacteleop
   -from isaacteleop.teleop_session_manager import DeviceState
   +import isaaccapture
   +from isaaccapture.teleop_session_manager import DeviceState

Every submodule keeps its name; only the top-level package changes. Both names
resolve to the same module objects until 1.9, so a partially migrated process
shares state, singletons and extension registrations rather than loading two
copies.

The same rename applies to ``python -m`` entry points, for example
``python -m isaaccapture.cloudxr.service run``.

.. warning::

   ``isaacteleop_examples`` became ``isaaccapture_examples``, and the alias
   does **not** cover it -- it serves ``isaacteleop`` alone. Every
   ``python -m isaacteleop_examples.*`` command fails with a bare
   ``ModuleNotFoundError`` and no notice attached. The example distributions
   renamed with it: ``isaacteleop-examples-*`` to ``isaaccapture-examples-*``.

What importing the old name does
--------------------------------

Until 1.9 the old name emits a standard ``DeprecationWarning`` on first import.
Python's warning filters control its visibility; use ``-W default`` to see it.
With ``-W error::DeprecationWarning``, the import raises until migrated or filtered.

Installing
----------

.. code-block:: diff

   -pip install isaacteleop[cloudxr]
   +pip install isaaccapture[cloudxr]

Update dependency pins and extras to the new name:

.. code-block:: diff

   -isaacteleop[retargeters,ui,cloudxr]~=1.5.0
   +isaaccapture[retargeters,ui,cloudxr]~=1.6.0

The ``isaaccapture`` wheel ships the ``isaacteleop`` alias inside it, so
``import isaacteleop`` keeps working after the requirement is renamed. Pip
installs the transition dependency automatically.

A ``sys.meta_path`` finder is invisible to a type checker, so everything under
``isaacteleop`` is typed ``Any`` and **pyright reports every submodule import as
unresolved** (``reportMissingImports``). The alias promises runtime identity only.

Logging
-------

Loggers are named after the module they live in, and the alias hands back the
real module, so the whole hierarchy moved to ``isaaccapture.*`` even for code
that still imports the old name. No alias can reach a string key, so a
``dictConfig`` or ``getLogger`` call naming ``isaacteleop`` silently stops
matching -- the logger is created, and nothing ever logs to it:

.. code-block:: diff

   -logging.getLogger("isaacteleop.cloudxr").setLevel(logging.DEBUG)
   +logging.getLogger("isaaccapture.cloudxr").setLevel(logging.DEBUG)
