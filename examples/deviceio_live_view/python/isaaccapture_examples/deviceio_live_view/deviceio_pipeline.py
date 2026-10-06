# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the human DeviceIO sources used by the live viewer."""

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HeadSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner

from .body_pipeline import BodyViewPipeline, create_body_view_pipeline
from .hand_pipeline import HandViewPipeline, create_hand_view_pipeline


def build_all_human_pipeline(
    soma_data_root=None,
    *,
    body_schema=None,
    soma_body_collection_id="soma_body_demo",
    soma_body_representation="joint-rotations",
    body: BodyViewPipeline | None = None,
    hands: HandViewPipeline | None = None,
):
    """Wire every human-related DeviceIO source into one pipeline."""
    head = HeadSource(name="head")
    controllers = ControllersSource(name="controllers")
    selected_body = body or create_body_view_pipeline(
        soma_data_root,
        body_schema=body_schema,
        soma_body_collection_id=soma_body_collection_id,
        soma_body_representation=soma_body_representation,
    )
    selected_hands = hands or create_hand_view_pipeline()
    return OutputCombiner(
        {
            "hand_left": selected_hands.left,
            "hand_right": selected_hands.right,
            "head": head.output("head"),
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "body": selected_body.output,
        }
    )
