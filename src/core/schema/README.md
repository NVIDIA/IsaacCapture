<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Schemas and their golden binary form

[`fbs/`](fbs) holds the FlatBuffer schemas, this module's language-neutral contract.
Each binding generates its own code from them.

[`golden/`](golden) holds one `.bfbs` per schema, as those schemas compiled at the last
agreed revision. These are byte-for-byte the bytes `McapTrackerChannels` embeds in every
recording, so an existing `.mcap` on disk carries one of them.

They are deliberately **not** in LFS, unlike the binary assets in `.gitattributes`. The
test suite reads them, and a clone that skipped `git lfs pull` would hand the test a
pointer file to parse as a schema — reported as an unparseable schema rather than a
missing fetch. At a couple of KB each, changing only when a schema does, plain git
objects cost nothing.

The `[schema_conform]` cases in `schema_tests` compare what the schemas compile to *now*
against the goldens, and fail when a change would make existing recordings unreadable.
That is the CI-time half of the check `McapTrackerViewers` performs at replay time; both
call `check_schema_compat()`, so they agree on what "unreadable" means. Note that the
comparison reads the `.bfbs` out of the build tree, so `ctest` on its own re-checks
whatever was built last: **build before you run it**, or a breaking edit reads as green.

## A schema that is not recorded

[`fbs/camera_protocol.fbs`](fbs/camera_protocol.fbs) is the conversation between the camera
plugin and the readers of its frames ([`camera_protocol`](../camera_protocol)). No recording
carries it. It is kept here, with a golden, because it is a contract of the same kind: the
two ends can come from different releases, and a change the comparison rejects would make
them misread each other.

It is not a recorded schema in disguise. It has no `Record` root and does not use
`DeviceDataTimestamp`: its times are the host's monotonic clock, and its lease ids, slot
indices and attached descriptors mean something between two processes on one machine only.
A recorder that stores camera frames maps those times in its own schema.

## Adding a field

Append it to a table with a fresh `id`, and leave it optional. The test passes, and no
golden needs to change — a golden that is *older* than the current schema is exactly what
it is for.

`(required)` is the exception the test reports separately: the recorded messages carry no
such field, so `Verifier::VerifyBuffer` rejects them at replay.

## When the test fails

It is telling you that recordings made before your change can no longer be read
correctly. Reach for one of these, roughly in order of preference:

- **Append instead of edit.** A new *optional* table field with a fresh `id` is safe.
- **Deprecate instead of delete.** Mark the field `(deprecated)` rather than removing it,
  so the ids after it keep their slots.
- **Never change a `struct`.** Structs are stored inline with no vtable, so their size
  is baked into every layout enclosing them. Resizing one silently misreads any struct
  it is nested in and any vector of it, and reads past the end of the recorded value
  where it is a table field. Add a new optional *table* field instead.
- **Renaming a `root_type` breaks recordings even though the bytes do not change.** The
  recording carries the old name and `McapTrackerViewers` matches it exactly, so a rename
  is as breaking as a layout change and needs the same decision.

## Refreshing the goldens

Only once the break is intentional and accepted:

```bash
cmake --build build --target schema_golden_update
```

That copies the freshly built `.bfbs` over [`golden/`](golden); it adds and overwrites but
never deletes, so `git rm` the golden of a schema you removed. Commit the result together
with the schema change, and say in the commit message what stops reading.
