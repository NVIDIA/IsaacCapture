<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Welcome to Isaac Capture

[![Default: Apache-2.0](https://img.shields.io/badge/Default-Apache--2.0-blue.svg)](development/license.html) [![EgoPost: OpenMDW-1.1](https://img.shields.io/badge/EgoPost-OpenMDW--1.1-blue.svg)](https://github.com/NVIDIA/IsaacCapture/blob/main/LICENSES/LicenseRef-OpenMDW-1.1.txt)

**NVIDIA Isaac Capture** is a shared library for teleoperation and egocentric data
capture. It brings device plugins, retargeters, and data utilities together so
applications can collect and use human demonstration data across robot and
simulation environments.

**Isaac Teleop** is the robot-in-the-loop component within Isaac Capture. It
provides teleoperation sessions and workflows for controlling robots and
collecting demonstrations. The **Ego Pipeline** covers human and egocentric data
capture and post-processing. Both build on the shared capture infrastructure.

See [Architecture](overview/architecture.md) for the component relationships and
[Ecosystem](overview/ecosystem.md) for devices and integrations. Source code and Python
packages are distributed through [GitHub](https://github.com/NVIDIA/IsaacCapture)
and [PyPI](https://pypi.org/project/isaaccapture/).

#### TIP
To run Isaac Teleop, follow [Quick Start](teleop/quick_start.md) for installation,
headset setup, and your first session. Existing users can consult
[Migrating to isaaccapture](development/migration.md) for the Python package migration.

## Table of Contents

## Overview

* [Architecture](overview/architecture.md)
* [Isaac Teleop](overview/isaac_teleop.md)
* [Ecosystem](overview/ecosystem.md)

## Isaac Teleop

* [Quick Start](teleop/quick_start.md)
  * [1. Check out code base (for examples)](teleop/quick_start.md#check-out-code-base-for-examples)
  * [2. Install the `isaaccapture` pip package](teleop/quick_start.md#install-the-isaaccapture-pip-package)
  * [3. Configure CloudXR (optional)](teleop/quick_start.md#configure-cloudxr-optional)
  * [4. Whitelist ports for Firewall](teleop/quick_start.md#whitelist-ports-for-firewall)
  * [5. Connect an XR headset](teleop/quick_start.md#connect-an-xr-headset)
  * [6. Run a teleop example](teleop/quick_start.md#run-a-teleop-example)
  * [Next steps](teleop/quick_start.md#next-steps)
  * [More Information](teleop/quick_start.md#more-information)
* [Synthetic XR Teleoperation](teleop/synthetic_xr.md)
  * [Franka simulation example](teleop/synthetic_xr.md#franka-simulation-example)
  * [One shared robot-control path](teleop/synthetic_xr.md#one-shared-robot-control-path)
  * [Try the example](teleop/synthetic_xr.md#try-the-example)
  * [Bring your own task logic](teleop/synthetic_xr.md#bring-your-own-task-logic)
* [OVXR installation](getting_started/ovxr.md)
  * [Install XR support](getting_started/ovxr.md#install-xr-support)
  * [Optional packages and licenses](getting_started/ovxr.md#optional-packages-and-licenses)
  * [Optional camera panels](getting_started/ovxr.md#optional-camera-panels)
  * [Optional desktop mirror](getting_started/ovxr.md#optional-desktop-mirror)
  * [Use from another application](getting_started/ovxr.md#use-from-another-application)
  * [Application-specific OVXR versions](getting_started/ovxr.md#application-specific-ovxr-versions)
  * [Session lifetime](getting_started/ovxr.md#session-lifetime)
* [Teleop Session](teleop/teleop_session.md)
  * [Overview](teleop/teleop_session.md#overview)
  * [Quick Start](teleop/teleop_session.md#quick-start)
  * [Configuration Classes](teleop/teleop_session.md#configuration-classes)
  * [API Reference](teleop/teleop_session.md#api-reference)
  * [Examples](teleop/teleop_session.md#examples)
  * [Benefits](teleop/teleop_session.md#benefits)
  * [Advanced Features](teleop/teleop_session.md#advanced-features)
* [Teleop Control State Machine](teleop/control_state_machine.md)
  * [Overview](teleop/control_state_machine.md#overview)
  * [Default State Machine Behavior](teleop/control_state_machine.md#default-state-machine-behavior)
  * [Connecting Inputs](teleop/control_state_machine.md#connecting-inputs)
  * [Using Context in High-Level App Logic](teleop/control_state_machine.md#using-context-in-high-level-app-logic)

## Ego Post Processing

* [EgoHand](ego_post_processing/egohand.md)
  * [Setup](ego_post_processing/egohand.md#setup)
  * [Example](ego_post_processing/egohand.md#example)
  * [Outputs](ego_post_processing/egohand.md#outputs)
  * [Hand Reconstruction](ego_post_processing/egohand.md#hand-reconstruction)
  * [2D Hand Boxes](ego_post_processing/egohand.md#d-hand-boxes)
  * [Keypoints to SOMA converter](ego_post_processing/egohand.md#keypoints-to-soma-converter)
  * [Docker](ego_post_processing/egohand.md#docker)
  * [Performance](ego_post_processing/egohand.md#performance)
  * [Limitations](ego_post_processing/egohand.md#limitations)
  * [License](ego_post_processing/egohand.md#license)

## Development

* [Build from Source](development/build_from_source/index.md)
  * [Prerequisites](development/build_from_source/index.md#prerequisites)
  * [One time setup](development/build_from_source/index.md#one-time-setup)
  * [1. Clone the repository](development/build_from_source/index.md#clone-the-repository)
  * [2. CMake: Configure and build](development/build_from_source/index.md#cmake-configure-and-build)
  * [3. Running tests](development/build_from_source/index.md#running-tests)
  * [4. Install the `isaaccapture` pip package](development/build_from_source/index.md#install-the-isaaccapture-pip-package)
* [Televiz](development/televiz.md)
  * [Installation](development/televiz.md#installation)
  * [Overview](development/televiz.md#overview)
  * [Display modes](development/televiz.md#display-modes)
  * [Quick start](development/televiz.md#quick-start)
  * [Session configuration](development/televiz.md#session-configuration)
  * [Layers](development/televiz.md#layers)
  * [Frame loop](development/televiz.md#frame-loop)
  * [Performance and diagnostics](development/televiz.md#performance-and-diagnostics)
  * [Session state](development/televiz.md#session-state)
  * [Sharing the XR session](development/televiz.md#sharing-the-xr-session)
  * [API reference](development/televiz.md#api-reference)
  * [C++ API](development/televiz.md#c-api)
  * [More information](development/televiz.md#more-information)
* [Migrating to `isaaccapture`](development/migration.md)
  * [In short](development/migration.md#in-short)
  * [Upgrading an existing install](development/migration.md#upgrading-an-existing-install)
  * [Imports](development/migration.md#imports)
  * [What importing the old name does](development/migration.md#what-importing-the-old-name-does)
  * [Installing](development/migration.md#installing)
  * [Logging](development/migration.md#logging)
  * [The asset cache](development/migration.md#the-asset-cache)
* [Contributing](development/contributing.md)
  * [Previewing documentation changes](development/contributing.md#previewing-documentation-changes)
* [License](development/license.md)
  * [Known exceptions](development/license.md#known-exceptions)
  * [License headers](development/license.md#license-headers)

## Integrations

* [Isaac Lab & Sim](integrations/isaac_lab_sim.md)
* [Isaac ROS](integrations/isaac_ros.md)
* [LeRobot and SO-101](integrations/lerobot/index.md)

## Devices

* [XR headsets](device/xr/index.md)
* [Hand tracking](device/hand_tracking/index.md)
  * [MANUS Gloves](device/hand_tracking/manus.md)
  * [Wuji Glove](device/hand_tracking/wuji_glove.md)
  * [Sharpa AvatarExo Glove](device/hand_tracking/sharpa_avatar.md)
  * [Haptikos Exoskeletons](device/hand_tracking/haptikos.md)
* [Body tracking](device/body_tracking/index.md)
  * [SOMA Pose Transport](device/body_tracking/soma.md)
* [Cameras](device/cameras/index.md)
  * [OAK Camera Plugin](device/cameras/oak.md)
  * [SENSING GMSL Camera](device/camera/sensing.md)
* [Joint-space devices](device/joint_space/index.md)
* [Tactile devices](device/tactile/index.md)
  * [OGLO Tactile Glove Plugin](device/tactile/oglo.md)
* [Spatial controllers](device/spatial_controllers/index.md)
  * [SpaceMouse Plugin](device/spatial_controllers/spacemouse.md)
* [Gamepads](device/gamepads/index.md)
  * [Gamepad Plugin](device/gamepad.md)
* [Spatial trackers](device/trackers/index.md)
  * [VIVE Ultimate Tracker](device/trackers/vive_ultimate_tracker.md)
* [Add a New Device](device/add_device.md)
* [Device Conformance](conformance/overview.md)

## References

* [System Requirements](references/requirements.md)
  * [Teleoperation to Robots with Input Devices](references/requirements.md#teleoperation-to-robots-with-input-devices)
  * [Teleoperation with Isaac Sim and Isaac Lab](references/requirements.md#teleoperation-with-isaac-sim-and-isaac-lab)
  * [Human-centric Data Collection](references/requirements.md#human-centric-data-collection)
* [Build System](references/build.md)
  * [Project structure](references/build.md#project-structure)
  * [CMake integration](references/build.md#cmake-integration)
  * [Using the Python wheel](references/build.md#using-the-python-wheel)
  * [Installing from source with pip (scikit-build-core)](references/build.md#installing-from-source-with-pip-scikit-build-core)
  * [Build directory layout and per-Python-version builds](references/build.md#build-directory-layout-and-per-python-version-builds)
  * [Output locations](references/build.md#output-locations)
  * [Troubleshooting](references/build.md#troubleshooting)
  * [Reference](references/build.md#reference)
* [Wheel contents SBOM and license evidence](references/sbom.md)
  * [What a wheel carries](references/sbom.md#what-a-wheel-carries)
  * [Evidence published with a release](references/sbom.md#evidence-published-with-a-release)
  * [Verifying a wheel](references/sbom.md#verifying-a-wheel)
* [Generated Tracker Code](references/generated_trackers.md)
  * [How it works](references/generated_trackers.md#how-it-works)
  * [What each entry produces](references/generated_trackers.md#what-each-entry-produces)
  * [Trackers with metadata](references/generated_trackers.md#trackers-with-metadata)
  * [Generated today](references/generated_trackers.md#generated-today)
  * [Still hand-written](references/generated_trackers.md#still-hand-written)
  * [Future work](references/generated_trackers.md#future-work)
* [Retargeting Interface](references/retargeting/index.md)
  * [Source Nodes](references/retargeting/index.md#source-nodes)
  * [Available Retargeters](references/retargeting/index.md#available-retargeters)
  * [Build a Retargeting Pipeline](references/retargeting/index.md#build-a-retargeting-pipeline)
  * [Add a New Retargeter](references/retargeting/index.md#add-a-new-retargeter)
* [Device provider monitoring](references/device_provider_monitoring.md)
  * [Reading status](references/device_provider_monitoring.md#reading-status)
  * [Identity and inventory](references/device_provider_monitoring.md#identity-and-inventory)
  * [Provider states](references/device_provider_monitoring.md#provider-states)
  * [Device states](references/device_provider_monitoring.md#device-states)
  * [Plugin status transport](references/device_provider_monitoring.md#plugin-status-transport)
  * [OpenXR ownership](references/device_provider_monitoring.md#openxr-ownership)
* [Camera Streaming](references/camera_streaming.md)
  * [Requirements](references/camera_streaming.md#requirements)
  * [Setup](references/camera_streaming.md#setup)
  * [Command overview](references/camera_streaming.md#command-overview)
  * [Web client for camera visualization](references/camera_streaming.md#web-client-for-camera-visualization)
  * [First run — a recording, no camera required](references/camera_streaming.md#first-run-a-recording-no-camera-required)
  * [Supported sources](references/camera_streaming.md#supported-sources)
  * [Running with a real camera](references/camera_streaming.md#running-with-a-real-camera)
  * [Display modes](references/camera_streaming.md#display-modes)
  * [Display surfaces](references/camera_streaming.md#display-surfaces)
  * [CloudXR runtime flags](references/camera_streaming.md#cloudxr-runtime-flags)
  * [Split mode — robot → workstation over RTP](references/camera_streaming.md#split-mode-robot-workstation-over-rtp)
  * [Configuration](references/camera_streaming.md#configuration)
  * [Status panel](references/camera_streaming.md#status-panel)
  * [Troubleshooting](references/camera_streaming.md#troubleshooting)
  * [How it works](references/camera_streaming.md#how-it-works)
  * [Sharing the XR session with TeleopSession](references/camera_streaming.md#sharing-the-xr-session-with-teleopsession)
  * [Next steps](references/camera_streaming.md#next-steps)
* [MCAP Recording & Replay](references/mcap_record_replay.md)
  * [Recording a Live Session](references/mcap_record_replay.md#recording-a-live-session)
  * [Replaying a Recording](references/mcap_record_replay.md#replaying-a-recording)
  * [Runnable Example](references/mcap_record_replay.md#runnable-example)
  * [API Reference](references/mcap_record_replay.md#api-reference)
* [CloudXR Service](references/cloudxr.md)
  * [Start the service](references/cloudxr.md#start-the-service)
  * [Manage the service](references/cloudxr.md#manage-the-service)
  * [Optional launch modes](references/cloudxr.md#optional-launch-modes)
  * [USB-local startup checks](references/cloudxr.md#usb-local-startup-checks)
  * [Re-open the client on the headset](references/cloudxr.md#re-open-the-client-on-the-headset)
  * [Load CloudXR environment variables](references/cloudxr.md#load-cloudxr-environment-variables)
  * [Run teleop examples against the service](references/cloudxr.md#run-teleop-examples-against-the-service)
  * [Use a system OpenXR runtime (no CloudXR attach)](references/cloudxr.md#use-a-system-openxr-runtime-no-cloudxr-attach)
  * [Run the service in a container or CI](references/cloudxr.md#run-the-service-in-a-container-or-ci)
  * [Connect the OOB headset after application startup](references/cloudxr.md#connect-the-oob-headset-after-application-startup)
  * [Configuration](references/cloudxr.md#configuration)
  * [Troubleshooting](references/cloudxr.md#troubleshooting)
* [Out-of-Band Teleop Control](references/oob_teleop_control.md)
  * [Quick start](references/oob_teleop_control.md#quick-start)
  * [ADB automation](references/oob_teleop_control.md#adb-automation)
  * [Architecture](references/oob_teleop_control.md#architecture)
  * [WebSocket protocol](references/oob_teleop_control.md#websocket-protocol)
  * [HTTP API](references/oob_teleop_control.md#http-api)
  * [Authentication](references/oob_teleop_control.md#authentication)
  * [Web client integration](references/oob_teleop_control.md#web-client-integration)
  * [Environment variables](references/oob_teleop_control.md#environment-variables)
  * [USB-local mode (`--setup-oob --usb-local`)](references/oob_teleop_control.md#usb-local-mode-setup-oob-usb-local)
  * [Troubleshooting](references/oob_teleop_control.md#troubleshooting)
* [Haptic Feedback](references/haptic_feedback.md)
  * [Architecture](references/haptic_feedback.md#architecture)
  * [Example](references/haptic_feedback.md#example)
  * [Running](references/haptic_feedback.md#running)
  * [Adding a new haptic device](references/haptic_feedback.md#adding-a-new-haptic-device)
  * [See also](references/haptic_feedback.md#see-also)

## Indices and tables

* [Index](genindex.md)
* [Search Page](search.md)
