<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teleop ROS 2 Agent Notes

Supported ROS 2 reference publisher for Isaac Teleop. Hosted under `examples/`
because it depends on ROS 2 and the `isaaccapture` wheel; it is not part of the
core library. Consumed by Isaac ROS Teleop.

## Docker Validation

- This reference integration needs ROS 2 plus the built `isaaccapture` wheel, which are not present in the dev container. Run/validate it inside the `examples/teleop_ros2/Dockerfile` image (the same path CI's `test-teleop-ros2` job uses), not directly on the host.
- To exercise it without live XR hardware, replay an MCAP fixture: build the image, run the installed `teleop_ros2_mcap_generator` to write a fixture, then run `python -m isaaccapture_examples.teleop_ros2` with `-p mode:=<mode> -p mcap_replay_path:=<file>` and check topics (e.g. via `python -m isaaccapture_examples.teleop_ros2.integration_tests.teleop_ros2_topic_verifier`). Replay mode does not launch CloudXR, so no GPU/NGC runtime is required. Share the fixture across containers with `-v /tmp:/tmp` and `--network host` for ROS 2 discovery.
- The image build disables some CI gates (`-DENABLE_CLANG_FORMAT_CHECK=OFF`, `-DBUILD_TESTING=OFF`), so a green Docker build does not mean C++ formatting or ctest pass. Validate those separately.
- When creating temporary Docker images for `examples/teleop_ros2` validation, remove them before finishing the task unless the user explicitly asks to keep them.

## Source Layout

- In source code files under this package, preserve the existing grouped/sorted organization for helpers, message builders, classes, and member functions: scan the surrounding order before inserting, and do not place helpers near call sites when the existing section is sorted.
- In Python integration test verifier code, do not use bare `assert` for runtime validation; Python optimization can disable it, so raise explicit exceptions from validators.
- Keep ROS publisher methods at one abstraction level: call one output-specific pure
  builder, then publish or broadcast its result. Compose payload construction and
  transport encoding inside the message-builder module, not in the node.
- Do not extract a one-use mapping helper merely to shorten a builder; keep the
  complete output mapping together unless the helper is reused or represents an
  independently tested contract.

## Rename consistency

- When renaming a symbol or concept, update semantically coupled type names,
  fields, variables, constructor keywords, consumers, and tests in the same pass
  so old and new vocabulary do not coexist.
