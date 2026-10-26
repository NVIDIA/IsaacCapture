# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient range generation and camera-aware taxel selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import Candidate, FrozenMap, Issue, Skinning, Taxel, freeze
from .state import _index, replace_device


@dataclass(frozen=True)
class BatchSpec:
    device_id: str
    origin: tuple
    normal: tuple
    rows: int = 1
    cols: int = 1
    column_end: tuple | None = None
    row_end: tuple | None = None
    tangent: tuple | None = None
    column_span: float | None = None
    row_span: float | None = None
    scan: str = "row-major"
    reverse_rows: bool = False
    reverse_cols: bool = False
    excluded: tuple = ()
    allowed_links: tuple = ()
    component_seeds: FrozenMap = FrozenMap()
    start_index: int | None = None
    segment_row: int = 0
    depth: float = 0.005
    tangent_angle: float = 0.0
    flip_normal: bool = False

    def __post_init__(self):
        for name in (
            "origin", "normal", "column_end", "row_end", "tangent",
            "excluded", "allowed_links",
        ):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "component_seeds", FrozenMap(self.component_seeds))


def scan_cells(rows, cols, scan="row-major", reverse_rows=False, reverse_cols=False):
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
        raise ValueError("Rows and columns must be positive integers")
    if scan not in ("row-major", "column-major", "row-snake", "column-snake"):
        raise ValueError("Unknown scan order")
    row_axis, col_axis = list(range(rows)), list(range(cols))
    if reverse_rows:
        row_axis.reverse()
    if reverse_cols:
        col_axis.reverse()
    row_outer = scan in ("row-major", "row-snake")
    outer, inner = (row_axis, col_axis) if row_outer else (col_axis, row_axis)
    cells = []
    for index, first in enumerate(outer):
        traversal = (
            list(reversed(inner)) if scan.endswith("snake") and index % 2 else inner
        )
        cells.extend(
            (first, second) if row_outer else (second, first)
            for second in traversal
        )
    return tuple(cells)


def grid_samples(spec):
    import numpy as np

    from .geometry import unit

    cells = scan_cells(
        spec.rows, spec.cols, spec.scan, spec.reverse_rows, spec.reverse_cols
    )
    origin = np.asarray(spec.origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Range origin must be finite")
    normal = unit(spec.normal) * (-1 if spec.flip_normal else 1)

    def axis(endpoint, span, name):
        if endpoint is None:
            raise ValueError(f"Pick the {name} end center")
        displacement = np.asarray(endpoint, dtype=np.float64) - origin
        displacement -= normal * np.dot(normal, displacement)
        direction = unit(displacement)
        length = float(np.linalg.norm(displacement)) if span is None else float(span)
        if not np.isfinite(length) or length < 1e-6:
            raise ValueError(f"{name} span must be at least one micrometer")
        return direction, length

    if spec.cols > 1:
        column, width = axis(spec.column_end, spec.column_span, "column")
    else:
        column, width = None, 0.0
    if spec.rows > 1:
        row, height = axis(spec.row_end, spec.row_span, "row")
    else:
        row, height = None, 0.0
    if column is None and row is None:
        if spec.tangent is None:
            raise ValueError("A single taxel requires a tangent direction")
        tangent = np.asarray(spec.tangent)
        column = unit(tangent - normal * np.dot(normal, tangent))
        row = np.cross(normal, column)
    elif column is None:
        column = unit(np.cross(row, normal))
    elif row is None:
        row = unit(np.cross(normal, column))
    if abs(np.dot(column, row)) > 1 - 1e-6:
        raise ValueError("Range directions are collinear")
    pitch_c = width / (spec.cols - 1) if spec.cols > 1 else 0.0
    pitch_r = height / (spec.rows - 1) if spec.rows > 1 else 0.0
    excluded = set(tuple(cell) for cell in spec.excluded)
    if any(cell not in cells for cell in excluded):
        raise ValueError("Excluded cells lie outside the selected range")
    points = tuple(
        (cell, origin + cell[1] * pitch_c * column + cell[0] * pitch_r * row)
        for cell in cells if cell not in excluded
    )
    return points, column, normal, (pitch_c, pitch_r)


def _free_start(device, count):
    occupied = {
        index
        for taxel in device.taxels
        if (index := _index(taxel.taxel_index)) is not None
    }
    # Raw orphan rows remain occupied until explicitly edited, never overwritten.
    for field, rows in (
        ("indices", device.skinning.indices), ("weights", device.skinning.weights)
    ):
        deleted = device.skinning.deleted_rows.get(field, FrozenMap())
        occupied.update(
            index for index in range(len(rows)) if str(index) not in deleted
        )
    start = 0
    while any(index in occupied for index in range(start, start + count)):
        start += 1
    return start


def _append_taxels(snapshot, device_id, taxels, segment_row, label):
    device = snapshot.document.device(device_id)
    if type(segment_row) is not int or not 0 <= segment_row < len(device.segments):
        raise ValueError("Select an existing segment row for new taxels")
    indices, weights = list(device.skinning.indices), list(device.skinning.weights)
    provenance = {
        field: dict(rows.items())
        for field, rows in device.skinning.deleted_rows.items()
    }
    existing = {
        index
        for item in device.taxels
        if (index := _index(item.taxel_index)) is not None
    }
    for taxel in taxels:
        index = taxel.taxel_index
        if type(index) is not int or index < 0:
            raise ValueError("A new taxel requires a nonnegative integer output index")
        if index in existing:
            raise ValueError("A new taxel conflicts with an existing output index")
        for field, rows in (("indices", indices), ("weights", weights)):
            if index < len(rows) and str(index) not in provenance.get(field, {}):
                raise ValueError(
                    "A new index would overwrite a raw skinning row; "
                    "select a free range"
                )
        existing.add(index)
        while len(indices) <= index:
            indices.append(None)
        while len(weights) <= index:
            weights.append(None)
        indices[index] = (segment_row,) * 4
        weights[index] = (1.0, 0.0, 0.0, 0.0)
        for field in tuple(provenance):
            provenance[field].pop(str(index), None)
            if not provenance[field]:
                del provenance[field]
    updated = replace(
        device, taxels=(*device.taxels, *taxels),
        skinning=Skinning(indices, weights, provenance),
    )
    issues = tuple(
        Issue(
            "error", "projection_failed", str(item.generation_issue),
            device_id, item.id, "bind_pose",
        )
        for item in taxels if item.generation_issue is not None
    )
    return Candidate(
        snapshot.revision, replace_device(snapshot.document, updated), label, issues
    )


def generate_batch(spec, snapshot, model):
    import numpy as np

    from .geometry import taxel_pose
    from .profiles import profile_for

    device = snapshot.document.device(spec.device_id)
    model = model.with_config(snapshot.document.model)
    if not spec.allowed_links or not spec.component_seeds:
        raise ValueError("Choose the permitted links and seed surface components")
    points, tangent, normal, _ = grid_samples(spec)
    if not points:
        raise ValueError("The range has no included cells")
    start = (
        _free_start(device, len(points))
        if spec.start_index is None else spec.start_index
    )
    if type(start) is not int or start < 0:
        raise ValueError("Starting output index must be a nonnegative integer")
    if model.resource_issues():
        raise ValueError("Reload changed resources before surface generation")
    root = profile_for(device).root_pose(
        device, model, snapshot.document.bind_configuration
    )
    surface = model.surface_snapshot(
        snapshot.document.bind_configuration, root_pose=root
    )
    taxels = []
    for offset, (_, seed) in enumerate(points):
        try:
            hit = surface.project(
                seed, normal, depth=spec.depth,
                allowed_links=spec.allowed_links,
                component_seeds=spec.component_seeds,
                flip_normal=spec.flip_normal,
            )
            pose = taxel_pose(hit.position, hit.normal, tangent, spec.tangent_angle)
            attachment, error = hit.attachment(), None
        except (ValueError, np.linalg.LinAlgError) as failure:
            pose, attachment, error = None, None, str(failure)
        taxels.append(Taxel(
            taxel_index=start + offset, bind_pose=pose,
            surface_attachment=attachment, generation_issue=error,
            provenance="surface",
        ))
    return _append_taxels(
        snapshot, device.id, tuple(taxels), spec.segment_row, "Add taxel range"
    )


def place_taxel(hit, spec, snapshot):
    from .geometry import taxel_pose

    device = snapshot.document.device(spec.device_id)
    start = _free_start(device, 1) if spec.start_index is None else spec.start_index
    normal = tuple(-value for value in hit.normal) if spec.flip_normal else hit.normal
    pose = taxel_pose(hit.position, normal, spec.tangent, spec.tangent_angle)
    taxel = Taxel(
        taxel_index=start, bind_pose=pose,
        surface_attachment=hit.attachment(), provenance="surface",
    )
    return _append_taxels(
        snapshot, device.id, (taxel,), spec.segment_row, "Place taxel"
    )


def reproject_taxels(
    snapshot, device_id, entity_ids, model, *, allowed_links,
    component_seeds, depth=0.005, flip_normal=False,
):
    from .geometry import pose_matrix, taxel_pose
    from .profiles import profile_for

    device = snapshot.document.device(device_id)
    selected = set(entity_ids)
    if not selected <= {item.id for item in device.taxels}:
        raise KeyError("A selected taxel no longer exists")
    configured = model.with_config(snapshot.document.model)
    if configured.resource_issues():
        raise ValueError("Reload changed model resources before reprojection")
    root = profile_for(device).root_pose(
        device, configured, snapshot.document.bind_configuration
    )
    surface = configured.surface_snapshot(
        snapshot.document.bind_configuration, root_pose=root
    )
    taxels, issues = [], []
    for taxel in device.taxels:
        if taxel.id not in selected:
            taxels.append(taxel)
            continue
        try:
            pose = pose_matrix(taxel.bind_pose)
            hit = surface.project(
                pose[:3, 3], pose[:3, 2], depth=depth,
                allowed_links=allowed_links,
                component_seeds=component_seeds,
                flip_normal=flip_normal,
            )
            taxels.append(replace(
                taxel,
                bind_pose=taxel_pose(hit.position, hit.normal, pose[:3, 0]),
                surface_attachment=hit.attachment(), generation_issue=None,
                provenance="surface",
            ))
        except (ValueError, TypeError) as error:
            taxels.append(replace(taxel, generation_issue=str(error)))
            issues.append(Issue(
                "error", "reprojection_failed", str(error),
                device_id, taxel.id, "bind_pose",
            ))
    updated = replace_device(snapshot.document, replace(device, taxels=tuple(taxels)))
    return Candidate(snapshot.revision, updated, "Reproject taxels", tuple(issues))


def select_indices(device, text):
    ranges = []
    for item in text.replace(",", " ").split():
        if "-" in item:
            first, last = item.split("-", 1)
            first, last = int(first), int(last)
            if first < 0 or last < first:
                raise ValueError("Invalid inclusive index range")
            ranges.append((first, last))
        else:
            index = int(item)
            if index < 0:
                raise ValueError("Taxel indices must be nonnegative")
            ranges.append((index, index))
    return tuple(
        item.id for item in device.taxels
        if (index := _index(item.taxel_index)) is not None
        and any(first <= index <= last for first, last in ranges)
    )


def pick_taxel(origin, direction, centers, radius):
    import numpy as np

    from .geometry import unit

    direction = unit(direction)
    best, distance = None, float("inf")
    for entity_id, position in centers.items():
        offset = np.asarray(origin) - position
        projection = float(offset @ direction)
        discriminant = (
            projection * projection - float(offset @ offset) + radius * radius
        )
        if discriminant < 0:
            continue
        roots = (
            -projection - np.sqrt(discriminant),
            -projection + np.sqrt(discriminant),
        )
        valid = [value for value in roots if value >= 0]
        if valid and min(valid) < distance:
            best, distance = entity_id, min(valid)
    return best


def rectangle_select(centers, camera_pose, fov, aspect, screen_min, screen_max,
                     *, surface=None, through=False):
    import numpy as np

    from .geometry import transform_point

    if not 0 < fov < np.pi or aspect <= 0:
        raise ValueError("A valid camera projection is required")
    inverse = np.linalg.inv(camera_pose)
    selected = []
    lo, hi = np.minimum(screen_min, screen_max), np.maximum(screen_min, screen_max)
    for entity_id, position in centers.items():
        point = transform_point(inverse, position)
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        screen = (0.5 + point[0] / (2 * point[2] * np.tan(fov / 2) * aspect),
                  0.5 + point[1] / (2 * point[2] * np.tan(fov / 2)))
        if not np.all(np.asarray(screen) >= lo) or not np.all(np.asarray(screen) <= hi):
            continue
        if not through and surface is not None:
            origin = np.asarray(camera_pose)[:3, 3]
            direction = np.asarray(position) - origin
            distance = np.linalg.norm(direction)
            if distance > 1e-9 and surface.ray_hits(
                origin, direction, max_distance=max(0, distance - 1e-5)
            ):
                continue
        selected.append(entity_id)
    return tuple(selected)
