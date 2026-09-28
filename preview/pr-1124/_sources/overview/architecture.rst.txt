.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Architecture
============

Isaac Teleop is the unified framework for high-fidelity egocentric and robotics data collection. It
streamlines device integration, standardizes human demo data collection, and fosters device and data
interoperability.

.. figure:: ../_static/isaac-teleop-architecture.svg
   :alt: Isaac Teleop Architecture

   **Figure:** Isaac Teleop high-level architecture

The core components of Isaac Teleop are:

Unified Device Interface
------------------------

- Support for major XR headsets, including Apple Vision Pro, Pico, Quest, and others
- Seamless integration of USB and Bluetooth peripherals (e.g., gloves, pedals, body trackers)
- Extensible interface enabling vendors to add custom or proprietary devices
- Consistent timestamping for multi-device streams, synchronized via unified device input control loop

Retargeting Interface
---------------------

- Tensor in, tensor out, GPU acceleration ready
- Reuse schema from the data interface
- Handle both a single data point, or an entire trajectory

Data Interface
--------------

- Standardized `data schema <https://github.com/NVIDIA/IsaacTeleop/tree/main/src/core/schema/fbs>`_ defined in the FlatBuffers (fbs) format.
- Data recording & playback with `mcap <https://mcap.dev/>`_
- Dataset interoperability with `LeRobot <https://github.com/huggingface/lerobot>`_

Visualization (Televiz)
-----------------------

Televiz (``isaacteleop.viz``) is a lightweight compositor module for visualizing what the operator
sees — camera and sensor feeds, plus 3D rendered content — in an XR headset or a desktop window.

- Composites multiple sources into one view: 2D camera/sensor planes (``QuadLayer``) and full-view stereo RGBD (``ProjectionLayer``) for 3D rendered content
- Per-eye stereo rendering and 3D placement in XR; the same API drives windowed and offscreen output
- Zero-copy submission of GPU frames straight from CuPy, PyTorch, or any CUDA memory object
- Shares one OpenXR session with the teleop device trackers, so rendering and tracking can run over a single CloudXR connection

See :doc:`../getting_started/televiz` for the module API and :doc:`../references/camera_streaming` for
the reference camera-streaming sample.
