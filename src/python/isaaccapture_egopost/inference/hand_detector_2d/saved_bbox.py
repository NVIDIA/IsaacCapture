# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Load pre-saved per-hand bbox JSON files as a drop-in HandBBoxDetector.

The JSON format matches what the pipeline writes via ``output_paths_per_hand``:
    { "000000": [cx, cy, size], "000001": [...], ... }
One file per hand, named ``<stem>_left_bbox.json`` / ``<stem>_right_bbox.json``.
"""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAVED_BBOX_FILE_PATTERN = "{stem}_{hand}_bbox.json"


class SavedBBoxDetector:
    """Read bboxes from pre-saved JSON files instead of running a detector.

    Reads ``<bbox_dir>/<video_stem>/<video_stem>_{hand}_bbox.json``.
    Input validation requires a file for at least one requested hand per video.
    Missing hands and frame IDs are treated as no-detection.
    """

    # Boxes come from JSON written against the source video, not from the frames
    # passed to run(), so a run that rectified the pixel aspect ratio has to map
    # them forward.
    reports_in_source_frame = True

    # Read-only: run() never writes to output_paths_per_hand. Lets callers
    # (e.g. run_egohand_inference_pipeline.py's stale-JSON cleanup) skip touching
    # files this detector doesn't own, without special-casing the class by name.
    writes_bbox_json = False

    def __init__(self, bbox_dir: str | Path | None = None):
        """Bind and validate the saved_bbox_path root shared by all videos."""
        bbox_path = os.path.expandvars(str(bbox_dir)) if bbox_dir is not None else ""
        if not bbox_path.strip():
            raise ValueError(
                "SavedBBoxDetector: saved_bbox_path resolves to an empty path. "
                "Set saved_bbox_path in the detector YAML or export BBOX_DIR "
                "to the bbox output directory."
            )
        # expandvars leaves references to unset variables unchanged.
        if re.search(r"\$\{|\$[A-Za-z_][A-Za-z0-9_]*", bbox_path):
            raise ValueError(
                "SavedBBoxDetector: saved_bbox_path contains an unresolved environment "
                f"variable: {bbox_path!r}. Set the referenced variable "
                "(e.g. export BBOX_DIR=/path/to/bbox_output) or use an explicit path."
            )
        self._bbox_root = Path(bbox_path).expanduser()
        if not self._bbox_root.is_dir():
            raise ValueError(
                "SavedBBoxDetector: saved_bbox_path is not an existing directory: "
                f"{self._bbox_root}. Point it at the bbox detection output directory."
            )
        self._bbox_dir: Path | None = None

    def _validate_clip_files(self, video_path, hands):
        stem = Path(video_path).stem
        paths = [
            self._bbox_root
            / stem
            / SAVED_BBOX_FILE_PATTERN.format(stem=stem, hand=hand)
            for hand in hands
        ]
        if not any(path.is_file() for path in paths):
            raise ValueError(
                f"SavedBBoxDetector: no matching bbox files for {video_path}. "
                f"Expected at least one of: {', '.join(map(str, paths))}. "
                "Check saved_bbox_path / BBOX_DIR and run bbox detection first."
            )

    def validate_inputs(self, video_paths, hands):
        """Require a bbox file for at least one requested hand in every video."""
        for video_path in video_paths:
            self._validate_clip_files(video_path, hands)

    def prepare_clip(self, *, video_path=None, K=None):
        """Select the clip directory, requiring at least one hand's bbox file.

        Ignores ``K`` -- this backend needs only the video path.
        """
        self._validate_clip_files(video_path, ("left", "right"))
        self._bbox_dir = self._bbox_root / Path(video_path).stem

    def run(self, image_ids, images_bgr, hands, output_paths_per_hand=None):
        bboxes = {image_id: {} for image_id in image_ids}
        stem = self._bbox_dir.name

        for hand in hands:
            json_path = self._bbox_dir / SAVED_BBOX_FILE_PATTERN.format(
                stem=stem, hand=hand
            )
            if not json_path.exists():
                log.warning("[SavedBBoxDetector] not found: %s", json_path)
                continue
            data = json.loads(json_path.read_text())
            loaded = 0
            for image_id in image_ids:
                if image_id in data:
                    bboxes[image_id][hand] = np.array(data[image_id], dtype=np.float32)
                    loaded += 1
            log.info(
                "[SavedBBoxDetector] %s: loaded %d/%d frames from %s",
                hand,
                loaded,
                len(image_ids),
                json_path.name,
            )

        return {iid: v for iid, v in bboxes.items() if v}
