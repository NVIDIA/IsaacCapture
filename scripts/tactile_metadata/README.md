<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor

Tactile Metadata Editor is a local Viser application for creating, editing, and
previewing glove tactile metadata. It loads URDF models and
[`TactileMetadata`](../../src/core/schema/fbs/tactile_metadata.fbs) JSON files.
All edits remain local; live connections only receive Isaac Capture data.

## Set up and launch

Use an existing Isaac Capture environment with Python 3.11 or newer. Install
the shared UI requirements into that environment:

```bash
cd scripts/tactile_metadata
python -m pip install -r ../../src/core/python/requirements-ui.txt
python -m tactile_metadata_editor.cli
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). To use another port:

```bash
python -m tactile_metadata_editor.cli --port 8090
```

Metadata, side, and URDF inputs are optional:

```bash
python -m tactile_metadata_editor.cli \
  --metadata ~/Downloads/right.metadata.json \
  --side right \
  --urdf "$(pwd)/model/right/linkerhand_FFG_right.urdf"
```

`--side` defaults to `left`. Paths beginning with `~` are expanded.

Offline editing uses Viser, NumPy, yourdfpy, trimesh, and rtree. Live reception
also requires the Isaac Capture SDK, a compatible OpenXR runtime, and an
available HMD system in the same Python environment.

## Data and coordinates

The editor maintains independent left- and right-hand documents. **Toggle Side**
selects the active document; each side has its own metadata path and data.
General uses a separate right-docked menu above the Editor/Preview menu. The
Selected Object panel is always present as a floating window on the left.
The **Mode** section uses bold, mutually exclusive **Edit** and **Preview**
checkbox labels; one mode is always selected.
Focused numeric fields preserve partial decimal input such as `1.` until the
next digit is entered or the field loses focus.

Three pose datasets are displayed:

- **Editing** contains editable segment and taxel binding poses plus Edit-mode
  URDF joint values.
- **Saved** is the last loaded or saved metadata snapshot. Enable **Show Saved
  Binding Poses** to render it at opacity 0.3.
- **Preview** contains current segment poses and Preview-mode URDF joint values.
  Preview taxel poses are derived from the selected binding metadata.

All UI values and scene interactions use visualization-space coordinates.
**Segments Alignment** maps metadata wrist coordinates into visualization space.
Loading applies that transform; saving applies its inverse. Changing the
alignment transforms the visible editor values without changing the underlying
metadata coordinates.

A three-axis indicator in the top-right shows visualization space relative to
the camera.

## Edit metadata

The **Edit** mode contains three tabs.

### URDF

1. Enter an absolute **URDF Path** and select **Load URDF**.
2. Use **URDF Alignment** to translate, rotate, and scale the model.
3. Expand **URDF Joints** to adjust each movable joint.

URDF joints are direct sliders. Segment and Taxel lists start collapsed, and
their displayed indices start at 0.

URDF configuration is session-only and is not saved in tactile metadata. The
model renders at opacity 0.2.

### Segments

- Use **Segments Alignment** to align all metadata poses.
- Add a segment from a HandJoint and XYZ position, or select **Add to URDF
  Surface** and click the model.
- Segments are flat-shaded cubes with side length 0.004 m. Smooth color
  gradients and cast/received shadows are disabled.
- Expand **All Segments** to edit HandJoint, position, and XYZ rotation.

HandJoint values must be unique. Taxel bindings use HandJoint names internally,
so removing or renaming a referenced segment leaves the affected taxel field
invalid and outlined in red.

### Taxels

Set Rows, Columns, Area Width, Area Height, Center Position, and Rotation.
**Capture URDF Surface Pose** copies a clicked surface position and orientation
into those fields without creating taxels. **Capture Segment Pose** copies the
complete pose of a clicked Editing segment. **Add Taxels** then creates the
row-major planar grid and copies the four Segment/Weight fields above the
capture buttons into every new Taxel. Taxels are squares with side length
0.002 m. A taxel with any validation error uses `ERROR_COLOR`;
`SELECTED_COLOR` overrides it while that taxel is selected.

To bind taxels:

1. Select **Start Binding to Segments** and select taxels in the scene.
2. Select **Select Segments** and choose one to four segments.
3. Select **Complete**. The first Segment receives weight one; later Segments
   receive weight zero.

At most one binding-step button is enabled at a time.

Expand **All Taxels** to edit 50 rows at a time; **Previous Page** and **Next
Page** switch pages without changing global Taxel indices. Each visible row
contains pose, four segment slots, and four weights. `None` is an editor-only
inactive slot with weight zero. Selecting the first active segment sets its
weight to one; later segment selections start at zero. Editing a weight changes
only that slot. Validation reports rows whose weights do not sum to one.

## Scene selection

- Click an object to select it.
- Shift-click adds objects of the same type.
- Shift-drag selects a screen rectangle and temporarily disables camera dragging.
- A selection cannot contain both segments and taxels. If an initial rectangle
  contains both, taxels take priority.
- Clicking empty scene space clears the selection.

Selected objects keep `SELECTED_COLOR`. Related objects use the separate
`SIDE_HIGHLIGHT_COLOR`: selecting segments highlights taxels with positive
weights to those HandJoints, while selecting taxels highlights their
positive-weight segments. Zero-weight slots are ignored.

The **Selected Object** panel edits position and rotation, deletes selected
objects, and scales multi-selection positions about their center. Edit mode
changes Editing data; Preview mode changes Preview segments. Preview taxels are
derived and cannot be selected. With no selection, the panel remains visible,
shows Type `None` and Count `0`, and disables its input fields.
Exactly one selected Segment reveals its HandJoint editor; exactly one selected
Taxel reveals its four Segment/Weight binding slots. These editors reuse one
control set rather than creating controls for every Taxel.

The selected-object 3-axis control supports axis-only translation. Dragging an
axis moves every selected object by the same delta. Single selections use the
object orientation; multi-selections use visualization-space axes. Rotation,
plane-translation, and scale handles are disabled.

Action and validation errors appear in the fixed **Info** area at the bottom of
the 3D viewport. It cannot be moved.

## Save and load

Enter **Metadata Path**, then use **Load** or **Save**. Saving validates the
current document. If errors remain, the editor asks whether to continue; invalid
drafts can still be written.

**New** resets the active side and initializes all OpenXR HandJoint segments in
the neutral hand pose. The left-hand defaults are mirrored before conversion to
visualization space.

The JSON contains only schema fields:

- `geometry_mapping_type` and `geometry_mapping`
- `segment_source_type` and `segment_source`

Editor HandJoint references are converted to `taxel_segment_indices` when
saving and restored when loading. Pose orientations are XYZW quaternions, and
taxel array order is the tactile force order. URDF paths, alignments, joint
values, selections, and live settings are not persisted.

The tool does not provide SDK binary export, a separate project format, or
standalone check/export CLI commands.

## Preview and live data

Use **Reset Values** to restore Preview segment poses and URDF joint values from
Editing.
Choose the mutually exclusive checkbox in **Binding Metadata Source**:

- **Editor** uses the current editor bindings and is selected by default.
- **Live** is disabled until metadata is received from the tactile stream.

The Preview Static section lists Segments before URDF Joints.

Preview provides four independent apply operations:

- **Apply Segments** replaces HandJoint/segment poses.
- **Apply Taxels** replaces Editing taxels with exactly the valid derived taxels
  currently rendered in Preview; invalid rows are omitted.
- **Apply URDF** replaces URDF joint values.
- **Apply ALL** applies Segments and URDF plus the same valid rendered taxel
  subset used by Apply Taxels.

Each Preview taxel is evaluated independently. Invalid rows are reported and
omitted from the scene while valid taxels continue to render. Live force rows
remain matched by their original taxel indices. Invalid taxels are reported in
Info and omitted from Apply Taxels and Apply ALL; no Apply button is disabled.

For live reception:

1. Set **Tactile Collection ID**.
2. Optionally enable **Use JointState to Drive URDF**, set its collection ID,
   and provide mappings:

   ```json
   {"source_joint": ["urdf_joint", 1.0, 0.0]}
   ```

3. Select **Connect Live**.

HandPose replaces Preview segments with only its valid joints, retained in
HandJoint order, and is the sole source of Live Preview segment identities and
poses. Invalid joints are absent. If WRIST is invalid, Preview contains no live
segments because a wrist-relative transform cannot be computed. Selected
binding metadata affects only taxel deformation; it never seeds Preview
segments. Enabled JointState mappings update Preview URDF joints. Live frames
overwrite manual Static values while connected.
Tactile force magnitude maps from blue at 0 N to red at 30 N, with values above
30 N clamped. Missing or inactive force data uses the default taxel color.

## Verify changes

From `scripts/tactile_metadata`:

```bash
ruff format --check tactile_metadata_editor
ruff check tactile_metadata_editor
python -m compileall -q tactile_metadata_editor
```
