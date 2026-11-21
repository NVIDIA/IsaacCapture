---
name: cmake-structure
description: >-
  Isaac Capture CMake target layout and #include conventions. Use whenever you
  create or edit a CMakeLists.txt, add or move a module/library/executable/test
  target, set up target_include_directories or target_link_libraries, decide
  where a header file goes (private vs public inc/), write or reorder #include
  directives in C/C++ sources, or restructure directories under src/ or deps/.
  Also use when reviewing diffs that touch CMakeLists.txt, header placement,
  include paths, or the overall project/directory structure.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Capture CMake & include structure

The rules, recommended structure, repo conventions, and review checklist live in
the repo's single source of truth:
**[`cmake/cmake-structure.md`](../../../cmake/cmake-structure.md)**.

Read that file and apply it before changing CMake files, header placement, or
include paths — for both authoring and review.
