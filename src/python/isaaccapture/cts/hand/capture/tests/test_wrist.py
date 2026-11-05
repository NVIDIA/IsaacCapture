# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import wrist


@pytest.mark.parametrize(
    ("plugin", "expected"),
    [
        ("wuji_glove_plugin", "WUJI_GLOVE_WRIST_SOURCE"),
        ("manus_hand_plugin", "MANUS_HAND_WRIST_SOURCE"),
        ("glove_x", "GLOVE_X_WRIST_SOURCE"),
    ],
)
def test_the_variable_is_the_plugin_name_without_plugin(plugin, expected):
    assert wrist.variable(plugin) == expected


def test_wuji_is_labelled_from_its_own_variable():
    env = wrist.read(["wuji_glove_plugin"], {"WUJI_GLOVE_WRIST_SOURCE": "controller"})
    assert env == {"WUJI_GLOVE_WRIST_SOURCE": "controller"}
    assert wrist.label(env) == "controller"


def test_any_glove_is_labelled_from_its_own_variable():
    env = wrist.read(["glove_x"], {"GLOVE_X_WRIST_SOURCE": "hand_tracking"})
    assert wrist.label(env) == "hand_tracking"


def test_another_plugins_variable_does_not_label_the_take():
    environ = {"WUJI_GLOVE_WRIST_SOURCE": "controller"}
    assert wrist.read([], environ) == {}
    env = wrist.read(["glove_x"], environ)
    assert env == {"GLOVE_X_WRIST_SOURCE": None}
    assert wrist.label(env) is None


def test_an_empty_variable_is_unset():
    assert wrist.label({"GLOVE_X_WRIST_SOURCE": ""}) is None


def test_plugins_that_agree_give_one_label():
    env = {"GLOVE_X_WRIST_SOURCE": "controller", "GLOVE_Y_WRIST_SOURCE": "controller"}
    assert wrist.label(env) == "controller"


def test_plugins_that_disagree_are_refused():
    env = {
        "GLOVE_X_WRIST_SOURCE": "controller",
        "GLOVE_Y_WRIST_SOURCE": "hand_tracking",
    }
    with pytest.raises(ValueError, match=r"GLOVE_X_WRIST_SOURCE=controller"):
        wrist.label(env)
