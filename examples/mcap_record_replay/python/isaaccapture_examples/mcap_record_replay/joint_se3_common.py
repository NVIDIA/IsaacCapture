# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the raw-hand JointSe3Pose scripts (live / record / replay).

These helpers depend on viser and the schema, not the retargeting engine.
"""

import viser

from isaaccapture.schema import JointName

# Display-only gap between the two hands along X, in metres. Each side is drawn half of
# this off the origin; poses themselves are untouched, so intra-hand pinch distances hold.
DEFAULT_SEPARATION_M = 0.3

# Tip label -> JointName, thumb first so the pinch partner is easy to spot.
TIPS = [
    ("thumb", JointName.HAND_RAW_THUMB_TIP),
    ("index", JointName.HAND_RAW_INDEX_TIP),
    ("middle", JointName.HAND_RAW_MIDDLE_TIP),
    ("ring", JointName.HAND_RAW_RING_TIP),
    ("little", JointName.HAND_RAW_LITTLE_TIP),
]

# The tips a pinch is measured to, i.e. every tip but the thumb itself.
FINGERS = TIPS[1:]


def hand_side(collection_id: str) -> str | None:
    """Side named by a collection id ("left" / "right"), or None when it names neither."""
    name = collection_id.lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    return None


def make_server(host: str, port: int, log_prefix: str) -> viser.ViserServer:
    """A viser server with the scene conventions these scripts share."""
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid(name="/grid", width=0.5, height=0.5, cell_size=0.05)
    print(f"{log_prefix} viser listening on {host}:{port} (http://localhost:{port})")
    return server


class TipViz:
    """One labelled coordinate frame per fingertip, for a single collection."""

    def __init__(
        self, server: viser.ViserServer, collection_id: str, separation: float
    ):
        side = hand_side(collection_id)
        x_offset = {"left": -0.5, "right": 0.5}.get(side, 0.0) * separation
        # Tips hang off a per-hand root, so the X shift is one node, not five.
        server.scene.add_frame(
            f"/{collection_id}",
            position=(x_offset, 0.0, 0.0),
            show_axes=False,
        )
        self._frames = {}
        for label, joint in TIPS:
            name = f"{side} {label}" if side else f"{collection_id} {label}"
            self._frames[joint] = server.scene.add_frame(
                f"/{collection_id}/{label}",
                axes_length=0.02,
                axes_radius=0.002,
                visible=False,
            )
            server.scene.add_label(f"/{collection_id}/{label}/label", text=name)

    def update(self, data) -> None:
        for _, joint in TIPS:
            frame = self._frames[joint]
            found = data.lookup(joint) if data is not None else None
            if found is None:
                # Absent from the frame means untracked; there is no validity flag.
                frame.visible = False
                continue
            p, q = found.pose.position, found.pose.orientation
            frame.position = (p.x, p.y, p.z)
            # schema quaternion is (x, y, z, w); viser expects (w, x, y, z).
            frame.wxyz = (q.w, q.x, q.y, q.z)
            frame.visible = True
