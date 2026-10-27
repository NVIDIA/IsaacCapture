<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor

A local Viser application for editing glove tactile metadata. It accepts URDF and
schema JSON files, preserves incomplete drafts, and exports checked numerical JSON
with an optional SDK binary. All runtime and device interactions are read-only.

## Prepare the existing Python environment

Use your existing Isaac Capture environment with Python 3.11–3.13. Viser provides
the browser interface; yourdfpy loads URDF and computes forward kinematics;
trimesh and rtree provide surface queries; NumPy supplies numerical operations.
These libraries are declared in Isaac Capture's shared UI requirements and package
extra. The editor does not create an environment, install packages, find another
interpreter, or build an SDK.

From the repository root, using the Python environment you selected:

```bash
cd scripts/tactile_metadata
python -m pip install -r ../../src/core/python/requirements-ui.txt
python -m tactile_metadata_editor.cli gui --urdf model/left/linkerhand_FFG_left.urdf --sides left
```

Open [the local editor](http://127.0.0.1:8080). Use `--port` to choose another port.
Offline editing does not import Isaac Capture or start OpenXR. SDK checking,
binary export and actual live reception require an SDK already importable in the
same environment. Live reception also requires a compatible runtime and HMD
system; see the [tactile device documentation](../../docs/source/device/tactile/index.rst).

## Create a layout

1. In **Model**, choose the wrist link (`base_link` in the supplied left model).
   Enter allowed links, for example `base_link,index_distal`, and apply them.
   Inspect the resource list and joint tree. All URDF links use fixed opacity
   0.6; highlighting changes color. Package mappings are a JSON object of package
   names to local directories, used to resolve `package://` resources without
   network access.
2. Use **Pick target** and **Pick on q0 model** to pick `Wrist O`, `Wrist F`
   toward the forearm, and `Wrist D` toward the back of the hand. Preview and
   confirm the wrist reference. Both hands use the same right-handed axes.
3. In **Segments**, adjust the free-joint slider or numeric input. Fixed and
   mimic joints appear in the tree; mimic joints follow their source. Confirm
   **Set shared binding pose q0**. Existing bind poses and weights remain
   numerically unchanged when q0 or the wrist reference changes.
4. Add `WRIST` with the wrist link; its bind pose is identity. Add, for example,
   `INDEX_DISTAL` with `index_distal`. New non-wrist segments start at the URDF
   link frame when calibrated. Pick a segment center or use the native gizmo and
   inspector to adjust its frame. URDF frames are initial estimates: calibrate
   their axes to the HandJoint convention before production use. Associating an
   imported segment with a link preserves its numerical bind pose.
5. In **Layout**, select the intended connected surface component. Set rows and
   columns to `4`, and enter the new taxels' segment row list, for example `[1]`.
   Pick the first center `O`, first-row last center `C`, and first-column last
   center `R`. Preview the grid. Spans and spacing are reported in meters; spans
   can be edited along the picked directions.
6. Choose scan order, reversal, optional excluded `[row,column]` cells and a
   start output index. `-1` finds an unused contiguous interval. Failed
   projections remain red candidates; excluded cells produce no candidate.
   Apply or cancel. Applying creates independent taxels and clears the transient
   generation parameters. For a single row use O/C; for a single column use O/R.
   For one point, pick O and then **Single tangent** to provide the tangent.
7. Click draft points, toggle with Ctrl/Cmd, enable rectangle selection, or enter
   an output-index list/range such as `0:15,20`. Selection defaults to visible
   points; **Select through mesh** includes hidden points. Inspect lists in
   pages of 25. Reproject, move, flip normals or twist tangents as needed.
   Inspector fields replace only the edited component; mixed values are reported.
8. In **Binding**, edit the four slots or assign one to four segment rows with
   equal initial weights. Repeated indices and slot order are retained. A slider
   or numeric weight edit replaces its slot, then divides all four values by
   their sum independently for each selected taxel. Invalid edits preserve the
   rejected row and report the reason. Loading, recording, checking, saving and
   exporting never normalize raw weights.
9. Preview motion with URDF controls. Return to q0 to resume surface picking.
   Use **Preview / apply output indices** to review explicit index changes.
   Moving or deleting taxels does not compact output indices; formal export
   requires exactly `0..N-1`. Segment reordering remaps valid slot references,
   including zero-weight slots. Deleted references remain invalid until repaired.

Use the supplied right model in a separate document:

```bash
python -m tactile_metadata_editor.cli gui --urdf model/right/linkerhand_FFG_right.urdf --sides right
```

Its wrist link is `hand_base_link`. For a real dual-hand URDF, choose `--sides
both`, then assign distinct wrist links and each hand's allowed surfaces. The
editor does not mirror a single-hand model. **New document** unlocks the model
hand selection after handling unsaved drafts.

## Save, reopen, check and export

**Export → Save all metadata** writes schema metadata values only, using the same
JSON as backups and export. Model, calibration and source settings stay in memory;
reopening restores values and array order only. Incomplete values remain saveable,
with validation errors shown in the interface. If any save fails, no side is
marked saved. Only metadata-value changes mark the document unsaved.
**Info issue → Locate** selects the relevant hand/entity in the shared draft
inspector. Advanced schema repair can replace incomplete arrays, with a temporary
backup before replacement.

```bash
python -m tactile_metadata_editor.cli gui --metadata left.metadata.json --sides left
python -m tactile_metadata_editor.cli check left.metadata.json
python -m tactile_metadata_editor.cli check left.metadata.json --sdk
python -m tactile_metadata_editor.cli export left.metadata.json --sides left --format sdk --out package
python -m tactile_metadata_editor.cli export left.metadata.json right.metadata.json --sides both --out pair
```

Choose **Assign JSON to** and **Open JSON**, or provide `--sides left/right/both`
on the CLI. Pure checking needs no side; export requires one file per specified
hand, left then right for `both`.

The protocol JSON follows [tactile_metadata.fbs](../../src/core/schema/fbs/tactile_metadata.fbs):
`geometry_mapping_type: "LBS"`, nested `geometry_mapping` arrays, and
`segment_source_type: "HandSkeletonRef"` with nested `segment_source.joint_ids`.
Pose quaternions are XYZW; influence structs contain `values` arrays. There is no
additional metadata format. Taxel array row order expresses output order.
Exports contain only metadata JSON and optional SDK binaries; check results and
revisions appear in the interface or CLI output. All sides must succeed before
the new or empty destination is published. GUI exports capture a snapshot;
later edits do not change that snapshot.

## Live reception and the SDK-free demo

Launch with `gui --demo` to use a synthetic independent layout and hand/force
stream. In **Live**, configure each side's tactile collection (default `tactile`),
HandTracker output and optional JointState collection, then connect. Apply source
configuration in the current session and reconnect. JointState mappings specify
a source name, URDF target, scale and offset; the panel reports raw values,
converted values, limits and validity. Demo JointState names come from mappings
present when connecting.

Only enabled JointState drives the live URDF; otherwise it is hidden. HandPose
supplies the tracked skeleton and current segment transforms relative to WRIST.
The live skeleton and taxels use the tracked wrist frame. The draft gizmo always
uses the calibrated q0 frame, independently of the animated drawing.

**Play / pause** freezes the live drawing while receipt continues. Draft forms,
wire markers and manipulators remain editable. Resume uses the latest frame and
current draft. Returning to EDIT shows q0 without stopping receipt. Interrupted
streams retain the last geometry and mark forces invalid rather than zero.
Force colors cover 0–30 N by default and saturate at the selected maximum.
Multi-segment LBS supplies position, without inventing a mixed rigid orientation
or a world-space force arrow.

Visualization defaults to **Received** metadata. Switching to **Draft** changes
only the drawing. **Record metadata groups** first backs up the draft to the
system temporary directory, logs its path, then replaces the selected groups
verbatim. Partial replacements may introduce count/reference errors, which remain
editable. Metadata/force alignment is the producer's and user's responsibility:
forces always use protocol row order. Source switching establishes no alignment.

## Implementation and verification

`document.py` owns raw arrays, IDs, explicit edits, validation and transactional
State revisions. `geometry.py` owns local resource resolution, FK, frame math,
projection and LBS. Its link attachments are derived caches, not metadata edits.
`storage.py` owns schema JSON, atomic saves, backups and package publication.
`sdk.py` imports SDK classes only inside children started with `sys.executable`:
checks use `LBS`, `HandSkeletonRef`, `TactileMetadata.to_bytes()`; live reception
uses `HandTracker`, `TactileTracker`, optional `JointStateTracker`,
`OpenXRSession` and `DeviceIOSession.run/update`. Views are copied before the
bounded local pipe, and only the latest frame is retained. `cli.py` and `ui.py`
share document, geometry and storage operations. The UI uses native Viser
controls and persistent scene nodes; document edits run on one event queue.

Local development checks target this package only:

```bash
ruff format --check tactile_metadata_editor
ruff check tactile_metadata_editor
python -m compileall -q tactile_metadata_editor
```

The inspection environment lacked the GUI/geometry libraries and SDK. Syntax,
lint, schema round trips, weight/reordering operations, rollback, atomic export,
and SDK-unavailable handling were checked locally. Viser call signatures were
checked against cached 1.1.1 source. Browser interaction, numerical mesh/LBS
execution, successful SDK encoding and actual hardware reception still require
verification in the configured environment; these checks are not claimed passed.
