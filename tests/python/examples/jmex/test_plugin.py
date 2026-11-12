# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the example launches ``jmex_plugin``, where there is no Isaac Lab."""

from __future__ import annotations

from pathlib import Path

import pytest
from repo_paths import repo_root

from isaaccapture_examples.jmex import plugin


def test_the_plugin_is_looked_for_in_this_checkouts_install_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", lambda self: True)
        search_paths = plugin.plugin_search_paths()

    assert search_paths == [repo_root() / "install" / "plugins"]


def test_the_plugin_is_launched_on_the_channel_and_collection_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plugin, "plugin_search_paths", lambda: [tmp_path])

    config = plugin.plugin_config(255, "suit_b", otherwise="pass --no-plugin.")

    assert config.search_paths == [tmp_path]
    assert config.plugin_args[-2:] == ["--channel=255", "--collection-id=suit_b"]
    assert config.required


def test_no_installed_plugin_names_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin, "plugin_search_paths", list)

    with pytest.raises(RuntimeError, match=r"BUILD_PLUGIN_JMEX.*pass --no-plugin\."):
        plugin.plugin_config(1, "jmex_full_body", otherwise="pass --no-plugin.")
