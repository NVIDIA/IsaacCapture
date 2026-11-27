# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save All reports real persistence outcomes without opening a GUI window."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from isaaccapture.retargeting_engine.interface import BoolParameter, ParameterState


@pytest.fixture
def ui_module(monkeypatch):
    imgui = MagicMock()
    imgui.button.side_effect = lambda label, **_kwargs: label == "Save All"
    glfw = MagicMock()
    glfw.get_window_size.return_value = (800, 600)
    integrations = ModuleType("imgui.integrations")
    renderer = ModuleType("imgui.integrations.glfw")
    renderer.GlfwRenderer = MagicMock()
    for name, module in {
        "imgui": imgui,
        "glfw": glfw,
        "imgui.integrations": integrations,
        "imgui.integrations.glfw": renderer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Load the actual UI without caching its mocked optional GUI dependencies.
    package = importlib.util.find_spec("isaaccapture.retargeting_engine_ui")
    source = Path(package.origin).with_name("multi_retargeter_tuning_ui.py")
    spec = importlib.util.spec_from_file_location(
        "isaaccapture.retargeting_engine_ui._save_all_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.MultiRetargeterTuningUIImGui, "_start", lambda self: None
    )
    return module


@pytest.mark.parametrize("layout", ["TABS", "FLOATING"])
@pytest.mark.parametrize(
    "scenario, failed",
    [
        ("all", []),
        ("none", ["left", "right"]),
        ("left_missing", ["left"]),
        ("right_missing", ["right"]),
        ("left_write_failure", ["left"]),
    ],
)
def test_save_all_reports_failures_and_still_saves_other_states(
    ui_module, tmp_path, layout, scenario, failed
):
    paths = {name: tmp_path / f"{name}.json" for name in ("left", "right")}
    if scenario == "left_write_failure":
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        paths["left"] = blocker / "left.json"
    states = {}
    for name in paths:
        missing = scenario == "none" or scenario == f"{name}_missing"
        states[name] = ParameterState(
            name,
            [BoolParameter("enabled", "Enabled", default_value=True)],
            config_file=None if missing else str(paths[name]),
        )
    retargeters = [
        SimpleNamespace(_name=name, get_parameter_state=lambda state=state: state)
        for name, state in states.items()
    ]
    ui = ui_module.MultiRetargeterTuningUIImGui(
        retargeters, layout_mode=getattr(ui_module.LayoutModeImGui, layout)
    )
    ui._render_tabbed_layout = MagicMock()
    ui._render_floating_layout = MagicMock()

    ui._render_ui()

    assert ui._status_success is (not failed)
    if failed:
        assert ui._status_message == f"✗ Failed to save: {', '.join(failed)}"
    else:
        assert ui._status_message == "✓ All configurations saved successfully!"
    for name, path in paths.items():
        if name not in failed:
            assert json.loads(path.read_text()) == {"enabled": True}
        else:
            assert not path.is_file()
