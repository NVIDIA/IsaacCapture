# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Unit tests for EgoHandInferencePipeline, EgoHandPredictor helpers, and the inference helper modules.

All tests run without a GPU or real checkpoints by injecting stub predictors
and detectors via the constructors introduced in the refactor.
"""

import numpy as np
import pytest
import torch

from inference.camera.types import ImageData
from inference.pipeline.predictor import EgoHandPredictor
from inference.pipeline.pipeline import EgoHandInferencePipeline
from inference.camera.intrinsics import estimate_K
from inference.preprocessing.crop import (
    crop_and_preprocess,
    project_camera_points_to_crop_with_validity,
)
from inference.preprocessing.mirroring import (
    mirror_full_image_camera_horizontal,
    mirror_legacy_crop_camera_horizontal,
    prepare_right_hand_model_input,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubPredictor:
    """Minimal predictor that returns one result dict per (hand, image) pair."""

    device = "cpu"
    batch_size = 1

    def run(
        self,
        images_bgr,
        image_ids,
        bboxes_per_image,
        hands,
        intrinsics_per_image=None,
        output_paths_per_hand=None,
    ):
        return {
            h: {
                image_id: {"stub": True}
                for image_id in image_ids
                if h in (bboxes_per_image.get(image_id) or {})
            }
            for h in hands
        }


class StubBBoxDetector:
    """Bbox detector that returns caller-supplied padded bboxes."""

    def __init__(self, bboxes_per_image):
        # bboxes_per_image: {image_id: {hand: bbox_array}}
        self._bboxes = bboxes_per_image

    def run(self, image_ids, images_bgr, hands, output_paths_per_hand):
        return self._bboxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_K = np.eye(3, dtype=np.float32)
_K[0, 0] = _K[1, 1] = 100.0

_TINY_BGR = np.zeros((4, 4, 3), dtype=np.uint8)
_BBOX = np.array([32.0, 32.0, 10.0], dtype=np.float32)


def _image_data(bgr=_TINY_BGR, intrinsics=_K, bboxes=None):
    """Build an ImageData with GT bboxes for both hands by default."""
    if bboxes is None:
        bboxes = {"left": _BBOX.copy(), "right": _BBOX.copy()}
    return ImageData(
        bgr=bgr,
        intrinsics=intrinsics,
        camera_to_world_transform=None,
        gt_2d_bboxes=bboxes,
    )


def _make_image_data_per_image(image_ids, bgr=_TINY_BGR, bboxes=None):
    return {fid: _image_data(bgr=bgr, bboxes=bboxes) for fid in image_ids}


def _padded_bboxes(image_ids, hands=("left", "right")):
    """Simulate the output of pad_hand_bboxes / a bbox detector."""
    return {fid: {h: _BBOX.copy() for h in hands} for fid in image_ids}


def test_horizontal_camera_mirror_uses_pixel_centers_and_full_intrinsics():
    width = 640
    bbox = np.array([123.25, 202.5, 180.0], dtype=np.float32)
    intrinsics = np.array(
        [
            [810.0, 17.0, 301.5],
            [0.0, 825.0, 239.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_point = np.array([0.15, -0.08, 1.7], dtype=np.float32)

    mirrored_bbox, mirrored_K = mirror_full_image_camera_horizontal(
        bbox,
        intrinsics,
        width,
    )

    flip_limit = float(width - 1)
    np.testing.assert_allclose(
        mirrored_bbox,
        np.array(
            [flip_limit - bbox[0], bbox[1], bbox[2]],
            dtype=np.float32,
        ),
    )
    assert mirrored_K[0, 1] == -intrinsics[0, 1]
    assert mirrored_K[0, 2] == flip_limit - intrinsics[0, 2]

    original_h = intrinsics @ camera_point
    original_pixel = original_h[:2] / original_h[2]
    mirrored_point = camera_point * np.array(
        [-1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    mirrored_h = mirrored_K @ mirrored_point
    mirrored_pixel = mirrored_h[:2] / mirrored_h[2]
    np.testing.assert_allclose(
        mirrored_pixel,
        np.array(
            [flip_limit - original_pixel[0], original_pixel[1]],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-5,
    )
    # Callers are allowed to retain the source calibration unchanged.
    np.testing.assert_array_equal(
        bbox,
        np.array([123.25, 202.5, 180.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        intrinsics,
        np.array(
            [
                [810.0, 17.0, 301.5],
                [0.0, 825.0, 239.25],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_legacy_horizontal_camera_mirror_preserves_crop_checkpoint_contract():
    width = 640
    bbox = np.array([123.25, 202.5, 180.0], dtype=np.float32)
    intrinsics = np.array(
        [
            [810.0, 17.0, 301.5],
            [0.0, 825.0, 239.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    mirrored_bbox, mirrored_K = mirror_legacy_crop_camera_horizontal(
        bbox,
        intrinsics,
        width,
    )

    assert mirrored_bbox[0] == width - bbox[0]
    assert mirrored_K[0, 2] == width - intrinsics[0, 2]
    # The historical crop-after-flip metadata did not transform skew or the
    # remaining intrinsic entries. Old checkpoints depend on this exact input.
    expected_K = intrinsics.copy()
    expected_K[0, 2] = width - expected_K[0, 2]
    np.testing.assert_array_equal(mirrored_K, expected_K)


def test_calibrated_right_hand_raster_flips_full_image_before_crop():
    height, width = 300, 480
    horizontal_ramp = np.arange(width, dtype=np.uint16)[None, :]
    image = np.repeat(horizontal_ramp, height, axis=0)
    image = np.stack(
        [image % 256, (image * 3) % 256, (image * 7) % 256],
        axis=-1,
    ).astype(np.uint8)
    bbox = np.array([133.4, 141.2, 173.0], dtype=np.float32)
    intrinsics = np.array(
        [
            [810.0, 17.0, 221.5],
            [0.0, 825.0, 137.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    actual, mirrored_bbox, mirrored_K = prepare_right_hand_model_input(
        image,
        bbox,
        intrinsics,
        calibrated_full_image=True,
    )
    expected_bbox, expected_K = mirror_full_image_camera_horizontal(
        bbox,
        intrinsics,
        width,
    )
    expected, _ = crop_and_preprocess(
        np.ascontiguousarray(image[:, ::-1]),
        expected_bbox,
    )
    crop_then_flip, _ = crop_and_preprocess(image, bbox)
    crop_then_flip = crop_then_flip.flip(-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(mirrored_bbox, expected_bbox)
    np.testing.assert_array_equal(mirrored_K, expected_K)
    assert torch.max(torch.abs(actual - crop_then_flip)) > 1e-3


def test_legacy_right_hand_raster_retains_crop_then_flip_contract():
    image = np.arange(80 * 120 * 3, dtype=np.uint32).reshape(80, 120, 3)
    image = (image % 256).astype(np.uint8)
    bbox = np.array([31.25, 42.5, 57.0], dtype=np.float32)
    intrinsics = np.array(
        [
            [200.0, 5.0, 55.0],
            [0.0, 210.0, 39.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    actual, mirrored_bbox, mirrored_K = prepare_right_hand_model_input(
        image,
        bbox,
        intrinsics,
        calibrated_full_image=False,
    )
    expected, _ = crop_and_preprocess(image, bbox)
    expected_bbox, expected_K = mirror_legacy_crop_camera_horizontal(
        bbox,
        intrinsics,
        image.shape[1],
    )

    torch.testing.assert_close(actual, expected.flip(-1), rtol=0, atol=0)
    np.testing.assert_array_equal(mirrored_bbox, expected_bbox)
    np.testing.assert_array_equal(mirrored_K, expected_K)


# ---------------------------------------------------------------------------
# EgoHandInferencePipeline — constructor
# ---------------------------------------------------------------------------


def test_constructor_accepts_stub_predictor():
    pipeline = EgoHandInferencePipeline(predictor=StubPredictor())
    assert pipeline.hands == ["left", "right"]
    assert pipeline.bbox_detector is None


def test_constructor_stores_injected_detector():
    detector = StubBBoxDetector({})
    pipeline = EgoHandInferencePipeline(
        predictor=StubPredictor(), bbox_detector=detector
    )
    assert pipeline.bbox_detector is detector


def test_constructor_custom_hands():
    pipeline = EgoHandInferencePipeline(predictor=StubPredictor(), hands=["left"])
    assert pipeline.hands == ["left"]


# ---------------------------------------------------------------------------
# EgoHandInferencePipeline.run() — happy path (GT bboxes from ImageData)
# ---------------------------------------------------------------------------


def test_run_with_gt_bboxes_returns_expected_structure(tmp_path):
    image_ids = ["00000", "00001", "00002"]
    image_data = _make_image_data_per_image(image_ids)

    pipeline = EgoHandInferencePipeline(predictor=StubPredictor())
    result = pipeline.run(
        image_data_per_image=image_data,
        write_soma_npzs=False,
        output_folder=tmp_path / "out",
    )

    assert "inference_results_per_hand" in result
    assert set(result["inference_results_per_hand"]) == {"left", "right"}


def test_run_with_stub_detector_uses_detector_bboxes(tmp_path):
    image_ids = ["00000", "00001"]
    image_data = _make_image_data_per_image(image_ids)

    detector = StubBBoxDetector(_padded_bboxes(image_ids))
    pipeline = EgoHandInferencePipeline(
        predictor=StubPredictor(), bbox_detector=detector
    )
    result = pipeline.run(
        image_data_per_image=image_data,
        write_soma_npzs=False,
        output_folder=tmp_path / "out",
    )

    assert set(result["inference_results_per_hand"]["left"]) == set(image_ids)


# ---------------------------------------------------------------------------
# EgoHandInferencePipeline.run() — detector invocation tracking
# ---------------------------------------------------------------------------


def test_bbox_detector_is_invoked_when_injected(tmp_path):
    image_ids = ["00000", "00001"]
    image_data = _make_image_data_per_image(image_ids)
    call_log = []

    class TrackingDetector(StubBBoxDetector):
        def run(self, image_ids, images_bgr, hands, output_paths_per_hand):
            call_log.append({"image_ids": list(image_ids), "hands": list(hands)})
            return super().run(image_ids, images_bgr, hands, output_paths_per_hand)

    detector = TrackingDetector(_padded_bboxes(image_ids))
    pipeline = EgoHandInferencePipeline(
        predictor=StubPredictor(), bbox_detector=detector
    )
    pipeline.run(
        image_data_per_image=image_data,
        write_soma_npzs=False,
        output_folder=tmp_path / "out",
    )

    assert len(call_log) == 1, "detector.run() should be called exactly once"
    assert call_log[0]["image_ids"] == image_ids
    assert call_log[0]["hands"] == ["left", "right"]


def test_bbox_detector_not_invoked_when_none(tmp_path):
    image_ids = ["00000"]
    image_data = _make_image_data_per_image(image_ids)

    # No detector injected — GT bboxes from ImageData should be used
    pipeline = EgoHandInferencePipeline(predictor=StubPredictor(), bbox_detector=None)
    result = pipeline.run(
        image_data_per_image=image_data,
        write_soma_npzs=False,
        output_folder=tmp_path / "out",
    )

    assert "inference_results_per_hand" in result


# ---------------------------------------------------------------------------
# EgoHandInferencePipeline.run() — missing / None images
# ---------------------------------------------------------------------------


def test_image_with_none_bgr_is_skipped(tmp_path):
    image_data = {
        "00000": _image_data(),
        "00001": ImageData(
            bgr=None,
            intrinsics=_K,
            camera_to_world_transform=None,
            gt_2d_bboxes={"left": _BBOX, "right": _BBOX},
        ),
        "00002": _image_data(),
    }

    pipeline = EgoHandInferencePipeline(predictor=StubPredictor())
    result = pipeline.run(
        image_data_per_image=image_data,
        write_soma_npzs=False,
        output_folder=tmp_path / "out",
    )

    # "00001" had bgr=None so predictor never saw it
    assert "00001" not in result["inference_results_per_hand"]["left"]
    assert "00000" in result["inference_results_per_hand"]["left"]


def test_all_images_none_bgr_raises(tmp_path):
    image_data = {
        "00000": ImageData(
            bgr=None,
            intrinsics=_K,
            camera_to_world_transform=None,
            gt_2d_bboxes=None,
        ),
    }

    pipeline = EgoHandInferencePipeline(predictor=StubPredictor())
    with pytest.raises(RuntimeError, match="No image"):
        pipeline.run(
            image_data_per_image=image_data,
            write_soma_npzs=False,
            output_folder=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# EgoHandInferencePipeline.run() — error cases
# ---------------------------------------------------------------------------


def test_empty_image_data_raises(tmp_path):
    pipeline = EgoHandInferencePipeline(predictor=StubPredictor())
    with pytest.raises(ValueError, match="at least one"):
        pipeline.run(
            image_data_per_image={},
            write_soma_npzs=False,
            output_folder=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# estimate_K
# ---------------------------------------------------------------------------


def test_estimate_K_default_focal_is_max_dimension():
    K = estimate_K(width=640, height=480)
    assert float(K[0, 0]) == 640.0
    assert float(K[1, 1]) == 640.0
    assert float(K[0, 2]) == 320.0
    assert float(K[1, 2]) == 240.0


def test_estimate_K_explicit_focal():
    K = estimate_K(width=100, height=200, focal=500.0)
    assert float(K[0, 0]) == 500.0
    assert float(K[1, 1]) == 500.0


def test_estimate_K_returns_3x3():
    K = estimate_K(width=256, height=256)
    assert K.shape == (3, 3)


def test_validated_inference_projection_marks_depth_and_overflow_missing():
    camera_points = torch.tensor(
        [
            [
                [0.1, 0.2, 2.0],
                [0.0, 0.0, 0.0],
                [1e38, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    K = torch.tensor(
        [
            [
                [100.0, 0.0, 32.0],
                [0.0, 100.0, 32.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    bbx_xys = torch.tensor([[32.0, 32.0, 10.0]])

    keypoints, valid = project_camera_points_to_crop_with_validity(
        camera_points,
        K,
        bbx_xys,
    )

    assert valid.tolist() == [[True, False, False]]
    assert torch.isfinite(keypoints[0, 0]).all()
    assert torch.isnan(keypoints[0, 1:]).all()


class _FakeFullPerspectiveModel:
    camera_mode = "sam3db_perspective"
    kp2d_min_depth = 1e-4

    def forward_step(self, batch, train=False):
        batch_size = batch["img"].shape[0]
        identity = torch.eye(3).reshape(1, 1, 3, 3)
        pred_keypoints_2d = torch.zeros(
            batch_size,
            25,
            2,
        )
        pred_keypoints_2d[:, 0] = float("nan")
        projection_valid = torch.ones(
            batch_size,
            25,
            dtype=torch.bool,
        )
        projection_valid[:, 0] = False
        scale_params = torch.zeros(batch_size, 27)
        scale_params[:, 0] = 1.0
        return {
            "pred_vertices": torch.zeros(batch_size, 3, 3),
            "pred_cam_t": torch.tensor([0.0, 0.0, 2.0]).expand(batch_size, -1),
            "pred_cam": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, -1),
            "pred_keypoints_3d": torch.zeros(
                batch_size,
                25,
                3,
            ),
            "pred_keypoints_2d": pred_keypoints_2d,
            "pred_projection_valid": projection_valid,
            "pred_hand_params": {
                "wrist_orient": identity.expand(
                    batch_size,
                    1,
                    3,
                    3,
                ),
                "hand_pose": identity.expand(
                    batch_size,
                    24,
                    3,
                    3,
                ),
                "identity_coeffs": torch.zeros(
                    batch_size,
                    5,
                ),
                "scale_params": scale_params,
            },
        }


def _fake_prepared_input(hand):
    return {
        "image_id": "frame-0",
        "real_hand_side": hand,
        "crop_tensor": torch.zeros(3, 8, 8),
        "model_bbx_xys": np.array(
            [32.0, 32.0, 10.0],
            dtype=np.float32,
        ),
        "model_K": _K.copy(),
        "model_img_size": np.array(
            [64.0, 64.0],
            dtype=np.float32,
        ),
        "output_bbx_xys": np.array(
            [32.0, 32.0, 10.0],
            dtype=np.float32,
        ),
        "output_K": _K.copy(),
    }


def _fake_predictor():
    predictor = EgoHandPredictor.__new__(EgoHandPredictor)
    predictor.batch_size = 1
    predictor.device = "cpu"
    predictor.dtype = torch.float32
    predictor.model = _FakeFullPerspectiveModel()
    predictor._pose_mirror = None
    return predictor


def test_production_image_predictor_preserves_left_projection_validity():
    result = _fake_predictor()._run_imagebased_predictions(
        [_fake_prepared_input("left")]
    )[0]

    assert not result["pred_projection_valid"][0]
    assert result["pred_projection_valid"][1:].all()
    assert np.isnan(result["pred_keypoints_2d"][0]).all()


def test_production_image_predictor_preserves_right_projection_validity(
    monkeypatch,
):
    def _fake_mirror(
        model,
        poses,
        transl,
        identity_coeffs,
        global_scale,
        K_fullimg,
        bbx_xys,
        scale_params=None,
        pose_mirror=None,
    ):
        batch_size = poses.shape[0]
        keypoints = torch.zeros(batch_size, 25, 2)
        keypoints[:, 0] = float("nan")
        valid = torch.ones(batch_size, 25, dtype=torch.bool)
        valid[:, 0] = False
        return {
            "pred_vertices": torch.zeros(batch_size, 3, 3),
            "transl": transl,
            "pred_keypoints_3d": torch.zeros(
                batch_size,
                25,
                3,
            ),
            "pred_keypoints_2d": keypoints,
            "pred_projection_valid": valid,
            "poses": poses,
        }

    monkeypatch.setattr(
        "inference.pipeline.predictor.mirror_left_prediction_to_right",
        _fake_mirror,
    )
    result = _fake_predictor()._run_imagebased_predictions(
        [_fake_prepared_input("right")]
    )[0]

    assert not result["pred_projection_valid"][0]
    assert result["pred_projection_valid"][1:].all()
    assert np.isnan(result["pred_keypoints_2d"][0]).all()
