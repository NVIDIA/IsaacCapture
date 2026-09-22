# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the human DeviceIO sources used by the live viewer."""

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    HeadSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner

from .body_pipeline import BodyViewPipeline, create_body_view_pipeline

HANDS_CHANNEL = "hands"


def build_all_human_pipeline(
    soma_data_root=None,
    *,
    body_schema=None,
    soma_collection_id="soma_demo",
    soma_body_representation="joint-rotations",
    body: BodyViewPipeline | None = None,
):
    """Wire every human-related DeviceIO source into one pipeline."""
    hands = HandsSource(name=HANDS_CHANNEL)
    head = HeadSource(name="head")
    controllers = ControllersSource(name="controllers")
    selected_body = body or create_body_view_pipeline(
        soma_data_root,
        body_schema=body_schema,
        soma_collection_id=soma_collection_id,
        soma_body_representation=soma_body_representation,
    )
    return OutputCombiner(
        {
            "hand_left": hands.output(HandsSource.LEFT),
            "hand_right": hands.output(HandsSource.RIGHT),
            "head": head.output("head"),
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "body": selected_body.output,
        }
    )
