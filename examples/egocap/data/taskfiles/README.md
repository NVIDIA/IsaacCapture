<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Task catalogs

CSV task catalogs served read-only by the web backend at `GET /api/task-files` (names) and
`GET /api/task-files/{name}` (parsed task definitions, in file order). Only the files in this directory are
served; the backend never reads another location. Deployments must ship this directory with the application.

Two layouts are supported:

- **Studio catalogs** (the shipped `Kitchen.csv` is a small neutral example): a title record, a blank record, then
  the headers `Studio, Scenario, Task, Description`.
- **Mid Train**: headers on the first record; required `Task Name` and `Task Description`, optional
  `Primitive(s)` (split on `+`), `Props`, `Left Hand Grasp` and `Right Hand Grasp`. Task names are normalised to
  `lower_snake`.

Other columns (operators, sessions, completion tracking) are ignored. A task file lists available tasks; it does not
select one.

Set `EGOCAP_TASK_FILES_DIR` to serve a different directory (for example the one the recorder reads its task library
from); unset, the API reads this one. The summary file `Summary.csv`, hidden files and editor leftovers are never
listed. Errors: 400 for a malformed name, 404 for a catalog that is not listed, 500 (with the reason) for a catalog on
the server that is oversized or malformed, and 503 when the directory cannot be read.
