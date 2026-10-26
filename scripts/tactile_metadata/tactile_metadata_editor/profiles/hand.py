# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HandSkeletonRef semantics and wrist-relative model attachments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..domain import FrozenMap, Issue

# src/core/schema/fbs/hand.fbs, in OpenXR order; NUM_JOINTS is excluded.
JOINT_NAMES = (
    "PALM",
    "WRIST",
    "THUMB_METACARPAL",
    "THUMB_PROXIMAL",
    "THUMB_DISTAL",
    "THUMB_TIP",
    "INDEX_METACARPAL",
    "INDEX_PROXIMAL",
    "INDEX_INTERMEDIATE",
    "INDEX_DISTAL",
    "INDEX_TIP",
    "MIDDLE_METACARPAL",
    "MIDDLE_PROXIMAL",
    "MIDDLE_INTERMEDIATE",
    "MIDDLE_DISTAL",
    "MIDDLE_TIP",
    "RING_METACARPAL",
    "RING_PROXIMAL",
    "RING_INTERMEDIATE",
    "RING_DISTAL",
    "RING_TIP",
    "LITTLE_METACARPAL",
    "LITTLE_PROXIMAL",
    "LITTLE_INTERMEDIATE",
    "LITTLE_DISTAL",
    "LITTLE_TIP",
)
JOINT_VALUES = {name: index for index, name in enumerate(JOINT_NAMES)}


@dataclass(frozen=True)
class ResolvedHand:
    root_pose: object
    segment_poses: tuple
    issues: tuple[Issue, ...]


def wrist_mount(model, q0, link, origin, forearm, dorsal):
    import numpy as np

    from ..geometry import matrix_pose, unit

    z = unit(np.asarray(forearm) - origin)
    dorsal = np.asarray(dorsal) - origin
    y = unit(dorsal - z * (z @ dorsal))
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((np.cross(y, z), y, z))
    matrix[:3, 3] = origin
    local = np.linalg.inv(model.link_pose(link, q0)) @ matrix
    return FrozenMap({"link": link, "pose": matrix_pose(local)})


class HandProfile:
    joint_names = JOINT_NAMES

    def source_payload(self, segments):
        names = []
        for segment in segments:
            if (
                not isinstance(segment.semantic, str)
                or segment.semantic not in JOINT_VALUES
            ):
                raise ValueError(f"Unknown HandJoint: {segment.semantic!r}")
            names.append(segment.semantic)
        if len(set(names)) != len(names):
            raise ValueError("HandJoint definitions must be unique")
        return {"type": "HandSkeletonRef", "joint_ids": names}

    def validate(self, device):
        issues = []
        seen = set()
        for segment in device.segments:
            semantic = segment.semantic
            if not isinstance(semantic, str) or semantic not in JOINT_VALUES:
                code, reason = "joint_name", "Unknown or missing HandJoint name"
            elif semantic in seen:
                code, reason = "joint_duplicate", "HandJoint is defined more than once"
            else:
                seen.add(semantic)
                continue
            issues.append(
                Issue("error", code, reason, device.id, segment.id, "semantic")
            )
        for index, segment in enumerate(device.segments):
            if segment.semantic != "WRIST" or index >= len(device.segment_bind_poses):
                continue
            pose = device.segment_bind_poses[index]
            if isinstance(pose, tuple) and len(pose) == 7:
                try:
                    identity = (
                        all(abs(value) <= 1e-7 for value in pose[:6])
                        and abs(abs(pose[6]) - 1) <= 1e-5
                    )
                except (TypeError, ValueError):
                    continue
                if not identity:
                    issues.append(Issue(
                        "error", "wrist_identity", "WRIST bind pose must be identity",
                        device.id, segment.id, "segment_bind_poses",
                    ))
        if device.side not in (None, "left", "right"):
            issues.append(Issue(
                "error", "hand_side", "Hand side must be left or right",
                device.id, field="side",
            ))
        return tuple(issues)

    def root_pose(self, device, model, q):
        from ..geometry import pose_matrix

        mount = device.wrist_mount
        if mount is None or "link" not in mount or "pose" not in mount:
            raise ValueError("Select a wrist link and calibrate its reference frame")
        return model.link_pose(mount["link"], q) @ pose_matrix(mount["pose"])

    def resolve(self, device, model, q, *, require_attachments=False):
        import numpy as np

        from ..geometry import immutable_array, matrix_pose, pose_matrix

        issues = list(self.validate(device))
        try:
            root = self.root_pose(device, model, q)
        except (ValueError, TypeError, KeyError) as error:
            issue = Issue(
                "error", "wrist_mount", str(error), device.id, field="wrist_mount"
            )
            return ResolvedHand(None, (), (*issues, issue))
        poses = []
        for row, segment in enumerate(device.segments):
            try:
                if segment.frame_ref == "root":
                    if require_attachments and segment.semantic != "WRIST":
                        raise ValueError(
                            "Model motion preview requires a linked segment"
                        )
                    pose = device.segment_bind_poses[row]
                    pose_matrix(pose)
                elif segment.frame_ref.startswith("link:"):
                    if segment.mount_pose is None:
                        raise ValueError("Restore the segment's model attachment at q0")
                    matrix = (
                        np.linalg.inv(root)
                        @ model.link_pose(segment.frame_ref[5:], q)
                        @ pose_matrix(segment.mount_pose)
                    )
                    pose = matrix_pose(matrix)
                else:
                    raise ValueError("Unknown segment frame reference")
                poses.append(pose)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                poses.append(None)
                issues.append(Issue(
                    "error", "segment_pose", str(error), device.id,
                    segment.id, "segment_bind_poses",
                ))
        return ResolvedHand(immutable_array(root), tuple(poses), tuple(issues))

    def restore_mounts(self, device, model, q0):
        import numpy as np

        from ..geometry import matrix_pose, pose_matrix

        try:
            root = self.root_pose(device, model, q0)
        except (ValueError, TypeError, KeyError):
            return replace(
                device,
                segments=tuple(
                    replace(item, mount_pose=None) for item in device.segments
                ),
            )
        segments = []
        for index, segment in enumerate(device.segments):
            local = None
            if segment.frame_ref.startswith("link:"):
                try:
                    local = matrix_pose(
                        np.linalg.inv(model.link_pose(segment.frame_ref[5:], q0))
                        @ root
                        @ pose_matrix(device.segment_bind_poses[index])
                    )
                except (ValueError, TypeError, KeyError, IndexError):
                    local = None
            segments.append(replace(segment, mount_pose=local))
        return replace(device, segments=tuple(segments))

    def associate(self, device, segment_id, link, model, q0):
        model.link_pose(link, q0)
        segments = tuple(
            replace(item, frame_ref=f"link:{link}", mount_pose=None)
            if item.id == segment_id else item
            for item in device.segments
        )
        if not any(item.id == segment_id for item in device.segments):
            raise KeyError(segment_id)
        return self.restore_mounts(replace(device, segments=segments), model, q0)
