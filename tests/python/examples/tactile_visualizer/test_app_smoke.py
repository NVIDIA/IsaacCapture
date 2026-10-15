# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import viser

from isaaccapture_examples.tactile_visualizer.app import TactileVisualizer
from isaaccapture_examples.tactile_visualizer.model import (
    SessionState,
    default_openxr_shape,
)
from isaaccapture_examples.tactile_visualizer.urdf_config import UrdfConfig


def test_offline_app_constructs_with_viser_1_1_api():
    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        app = TactileVisualizer(
            server=server,
            urdf_path=Path(__file__).with_name("minimal_hand.urdf"),
            urdf_config_path=None,
            urdf_config=UrdfConfig(),
            document=default_openxr_shape(2),
            document_name="fixture",
            source=None,
            side="right",
            liveness_timeout_s=1.0,
            taxel_size_m=0.004,
            write_root=None,
        )

        assert app.controller.state is SessionState.OFFLINE
        assert len(app._taxel_handles) == 2
        assert app.urdf_joint_names == ("index_MCP_FE",)
        assert len(app._tree_joint_inputs) == 26
        assert len(app._tree_taxel_inputs) == 2
    finally:
        server.stop()


def test_invalid_urdf_is_reported_without_stopping_server(tmp_path):
    urdf = tmp_path / "bad.urdf"
    urdf.write_text("not XML", encoding="utf-8")
    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        app = TactileVisualizer(
            server=server,
            urdf_path=urdf,
            urdf_config_path=None,
            urdf_config=UrdfConfig(),
            document=default_openxr_shape(),
            document_name="untitled",
            source=None,
            side="right",
            liveness_timeout_s=1.0,
            taxel_size_m=0.004,
            write_root=None,
        )

        assert app.urdf is None
        assert "URDF unavailable" in app._last_error
        assert app.controller.state is SessionState.OFFLINE
    finally:
        server.stop()
