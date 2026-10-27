# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""2D drawing helpers (hand skeleton, bbox) for overlay videos."""

import cv2
import numpy as np

# SOMA 25-joint skeleton (wrist=0, thumb=1-4, index=5-9, middle=10-14, ring=15-19, pinky=20-24)
_SOMA_SKEL_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),  # index
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),  # middle
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),  # ring
    (0, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),  # pinky
]


def draw_hand_skeleton(img_rgb, keypoints_2d, bone_color, bbx_xys):
    """Draw hand skeleton on a full-image RGB array (in place).

    keypoints_2d: (25, 2) SOMA joints, in normalized crop space [-0.5, 0.5].
    bbx_xys: [cx, cy, size] — converts crop-space coords to full-image pixels.
    bone_color: (R, G, B) for bones; joints are drawn white.
    """
    cx, cy, size = float(bbx_xys[0]), float(bbx_xys[1]), float(bbx_xys[2])
    kps = np.asarray(keypoints_2d)
    bones = _SOMA_SKEL_BONES
    pts = []
    for j in range(len(kps)):
        if not np.isfinite(kps[j, :2]).all():
            pts.append(None)
            continue
        pts.append(
            (
                round(kps[j, 0] * size + cx),
                round(kps[j, 1] * size + cy),
            )
        )
    h, w = img_rgb.shape[:2]
    for i, j in bones:
        p1, p2 = pts[i], pts[j]
        if (
            p1 is not None
            and p2 is not None
            and 0 <= p1[0] < w
            and 0 <= p1[1] < h
            and 0 <= p2[0] < w
            and 0 <= p2[1] < h
        ):
            cv2.line(img_rgb, p1, p2, bone_color, 2, cv2.LINE_AA)
    for point in pts:
        if point is None:
            continue
        px, py = point
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img_rgb, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_bbox_on_image(vis_bgr, bbx_xys, color=(0, 255, 0), label=None):
    """Draw a cxcys bbox + red center dot on a BGR image (in place).

    Optionally draws a label above the top-left corner in the same color.
    """
    cx, cy, size = float(bbx_xys[0]), float(bbx_xys[1]), float(bbx_xys[2])
    x1, y1 = round(cx - size / 2), round(cy - size / 2)
    x2, y2 = round(cx + size / 2), round(cy + size / 2)
    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.circle(vis_bgr, (round(cx), round(cy)), 4, (0, 0, 255), -1)
    if label is not None:
        cv2.putText(
            vis_bgr,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
