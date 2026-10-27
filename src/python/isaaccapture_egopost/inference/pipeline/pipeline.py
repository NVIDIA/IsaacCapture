# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Run EgoHand/SOMA inference on image batches.

The caller supplies per-image ``ImageData`` entries. This module runs
EgoHand/SOMA inference on dataset-provided ground-truth or detected 2D hand
bounding boxes and returns SOMA predictions indexed by hand and image ID.
Each prediction includes SOMA parameters, root-relative mesh vertices and
3D keypoints, camera translation, projected 2D keypoints, and full-image
camera intrinsics.
An optional source camera restores 2D outputs after full-image undistortion.

SOMA NPZ writes are controlled by the ``write_soma_npzs`` kwarg on
``run()``. Default ``True`` -- standalone inference callers keep the on-disk
artifacts so they can be opened and inspected. The dataset orchestrator
can set it to ``False`` to skip the writes while keeping predictions in memory.
"""

import logging
import time
from pathlib import Path

from inference.camera.types import ImageData
from inference.pipeline.predictor import EgoHandPredictor
from inference.hand_detector_2d.hand_bbox_detector import (
    HandBBoxDetector,
    pad_hand_bboxes,
)
from inference.perf import PerfReporter, PerfStage

log = logging.getLogger(__name__)


class EgoHandInferencePipeline:
    """Coordinate bbox detection and EgoHand inference over image batches.

    Accepts an already-constructed ``EgoHandPredictor`` and an optional
    ``HandBBoxDetector``. This separation makes the class testable without GPU
    checkpoints — inject stub objects via the constructor.

    Typical production construction:

    .. code-block:: python

        predictor = EgoHandPredictor.from_checkpoint(
            egohand_ckpt_path=..., soma_hand_data_folder=..., batch_size=..., device=...,
        )
        bbox_detector = RTDetrConfig(weights=...).build(device=..., tracking=True)
        pipeline = EgoHandInferencePipeline(predictor=predictor, bbox_detector=bbox_detector)

    Pass ``bbox_detector=None`` when bboxes are always supplied via ``ImageData.gt_2d_bboxes``.

    Args:
        predictor: Loaded and device-placed ``EgoHandPredictor``.
        bbox_detector: Optional detector satisfying the ``HandBBoxDetector``
            interface. When set, ``run()`` uses it for bbox detection; when
            ``None``, ``run()`` uses GT bboxes from ``ImageData``.
        hands: Hand sides to run inference for. Defaults to both.
    """

    def __init__(
        self,
        predictor: EgoHandPredictor,
        bbox_detector: HandBBoxDetector | None = None,
        hands: list[str] | None = None,
        perf_reporter: PerfReporter | None = None,
    ):
        self.predictor = predictor
        self.bbox_detector = bbox_detector
        self.hands = hands if hands is not None else ["left", "right"]
        self.device = predictor.device
        self.batch_size = predictor.batch_size
        self.perf_reporter = perf_reporter

    def run(
        self,
        image_data_per_image: dict[str, ImageData],
        return_debug_data: bool = False,
        write_soma_npzs: bool = True,
        *,
        output_folder: str | Path,
        frame_times: dict[str, float] | None = None,
        defer_smoothing: bool = False,
        source_camera=None,
    ) -> dict:
        """Run bbox detection and EgoHand inference from clip ``ImageData`` entries.

        Args:
            image_data_per_image: ``ImageData`` records indexed by image ID.
                EgoHand requires ``bgr`` and ``intrinsics``; records where either
                field is ``None`` are skipped immediately before inference.
            return_debug_data: If true, return images, bboxes, predictions,
                and mesh faces for debug overlay rendering.
            write_soma_npzs: If true, write per-frame ``frame*_<hand>_soma.npz``
                files to ``output_folder``. Default ``True`` (standalone
                inference / debug runs need the files on disk for inspection).
                When false, predictions are returned without writing NPZ files.
            output_folder: Flat destination folder for bbox JSON and SOMA NPZ
                outputs.
            frame_times: Image ID to timestamp in seconds, required when the
                predictor or bbox detector smooths a complete clip.
            defer_smoothing: Return raw frame-batch results with their bboxes
                and cameras for whole-clip postprocessing. Requires disabled
                debug output and NPZ writes.
            source_camera: Original camera for undistorted inputs. Return source
                bboxes for the caller to save, instead of writing detector boxes.

        Returns:
            Dict carrying ``inference_results_per_hand`` (per-hand, per-image
            SOMA arrays). When ``return_debug_data`` is true, also includes
            ``images_bgr`` indexed by image ID, plus ``bboxes_per_image`` and
            ``faces_per_hand`` for overlay rendering.
        """
        smooth_bboxes = getattr(self.bbox_detector, "requires_full_clip", False)
        if smooth_bboxes and defer_smoothing:
            raise ValueError(
                "Offline bbox smoothing requires a complete clip before EgoHand cropping"
            )
        if defer_smoothing and (return_debug_data or write_soma_npzs):
            raise ValueError(
                "Deferred smoothing requires debug output and NPZ writes disabled"
            )
        if not image_data_per_image:
            raise ValueError("EgoHand inference requires at least one image")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        inference_inputs = []
        for image_id, image_data in image_data_per_image.items():
            if image_data.bgr is None or image_data.intrinsics is None:
                continue
            inference_inputs.append((image_id, image_data.bgr, image_data.intrinsics))
        if not inference_inputs:
            raise RuntimeError("No image has all inputs required for EgoHand inference")
        image_ids = [image_id for image_id, _, _ in inference_inputs]
        inference_images_bgr = [image_bgr for _, image_bgr, _ in inference_inputs]
        intrinsics_per_image = {
            image_id: intrinsics for image_id, _, intrinsics in inference_inputs
        }

        if self.bbox_detector is not None:
            bbox_paths = {
                hand: output_folder / f"{output_folder.name}_{hand}_bbox.json"
                for hand in self.hands
            }

            start_time = time.perf_counter()
            padded_bboxes_per_image = self.bbox_detector.run(
                image_ids=image_ids,
                images_bgr=inference_images_bgr,
                hands=self.hands,
                output_paths_per_hand=bbox_paths if source_camera is None else None,
                **({"frame_times": frame_times} if smooth_bboxes else {}),
            )

            per_frame_bbox_detection_time = (
                (time.perf_counter() - start_time) * 1000 / len(image_ids)
            )
            log.warning(
                "Per-frame bbox detection time: %s ms", per_frame_bbox_detection_time
            )

            if self.perf_reporter is not None:
                self.perf_reporter.record_latency(
                    PerfStage.BBOX_DETECTION, per_frame_bbox_detection_time
                )
        else:
            padded_bboxes_per_image = pad_hand_bboxes(
                {
                    image_id: dict(image_data.gt_2d_bboxes or {})
                    for image_id, image_data in image_data_per_image.items()
                }
            )

        output_paths_per_hand: dict | None
        if write_soma_npzs:
            output_paths_per_hand = {
                hand: {
                    image_id: output_folder / f"frame{image_id}_{hand}_soma.npz"
                    for image_id in image_ids
                }
                for hand in self.hands
            }
        else:
            output_paths_per_hand = None

        start_time = time.perf_counter()
        predictor_options = {}
        if frame_times is not None:
            predictor_options["frame_times"] = frame_times
        if defer_smoothing:
            predictor_options["defer_smoothing"] = True
        if source_camera is not None:
            predictor_options["source_cameras_per_image"] = dict.fromkeys(
                image_ids, source_camera
            )
        inference_results_per_hand = self.predictor.run(
            images_bgr=inference_images_bgr,
            image_ids=image_ids,
            bboxes_per_image=padded_bboxes_per_image,
            hands=self.hands,
            intrinsics_per_image=intrinsics_per_image,
            output_paths_per_hand=output_paths_per_hand,
            **predictor_options,
        )

        per_frame_inference_time = (
            (time.perf_counter() - start_time) * 1000 / len(image_ids)
        )
        log.warning("Per-frame inference time: %s ms", per_frame_inference_time)

        if self.perf_reporter is not None:
            self.perf_reporter.record_latency(
                PerfStage.INFERENCE, per_frame_inference_time
            )

        result = {"inference_results_per_hand": inference_results_per_hand}
        if defer_smoothing:
            result.update(
                bboxes_per_image=padded_bboxes_per_image,
                intrinsics_per_image=intrinsics_per_image,
            )
        if source_camera is not None:
            result["bboxes_per_image"] = padded_bboxes_per_image
        if return_debug_data:
            result.update(
                {
                    "images_bgr": dict(zip(image_ids, inference_images_bgr)),
                    "bboxes_per_image": padded_bboxes_per_image,
                    "faces_per_hand": {
                        hand: self.predictor.model.soma_hand.faces(hand)
                        for hand in self.hands
                    },
                }
            )
        return result
