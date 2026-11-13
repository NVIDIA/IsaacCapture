# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import shlex
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "deploy_sender.py"
spec = importlib.util.spec_from_file_location("deploy_sender", SCRIPT)
deploy_sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_sender)

RUN_CONFIG = deploy_sender.RUN_CONFIG
RELEASES_DIR = deploy_sender.RELEASES
RELEASE = "~/camera-streamer/releases/rel"
LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 12\n"
)


def dry_run_lines(capsys, *argv):
    assert deploy_sender.main(["user@host", "--dry-run", *argv]) == 0
    # A quoted multi-line script spans lines, so split on the marker rather than on newlines.
    return [part.strip() for part in capsys.readouterr().out.split("DRY-RUN ")[1:]]


def remote_scripts(lines):
    """The bash scripts passed to `bash -lc` in the dry-run ssh commands."""
    scripts = []
    for line in lines:
        for word in shlex.split(line):
            if word.startswith("bash -lc "):
                scripts.append(shlex.split(word)[2])
    return scripts


def test_run_config_lives_inside_the_release_source_tree():
    assert RUN_CONFIG == "src/configs/run-config.yaml"


def test_run_uploads_to_and_starts_with_the_run_config(capsys, tmp_path):
    config = tmp_path / "mine.yaml"
    config.write_text("cameras: []\n")
    lines = dry_run_lines(capsys, "run", "--name", "rel", "--config", str(config))
    upload = shlex.split(lines[0])
    assert f"cat > {RELEASES_DIR}/rel/{RUN_CONFIG}" in upload
    (script,) = remote_scripts(lines[1:])
    assert f"./build/argus_sender --config {RUN_CONFIG}" in script


def test_stop_and_reset_patterns_match_the_started_command(capsys):
    started = f"^\\./build/argus_sender --config {RUN_CONFIG}"
    (stop,) = remote_scripts(dry_run_lines(capsys, "stop"))
    assert stop.count(f'pgrep -f "{started}"') == 2
    (reset,) = remote_scripts(dry_run_lines(capsys, "reset"))
    assert f'pgrep -f "{started}"' in reset
    (reboot,) = remote_scripts(dry_run_lines(capsys, "reset", "--reboot"))
    assert f'pgrep -f "{started}"' in reboot


def test_remove_producer_stops_before_removing(capsys):
    (script,) = remote_scripts(dry_run_lines(capsys, "remove-producer"))
    removal = script.index("rm -rf /opt/isaaccapture/releases")
    guards = [
        "list-unit-files",
        "could not stop",
        "still running after the stop",
        "could not disable",
    ]
    positions = [script.index(guard) for guard in guards]
    assert positions == sorted(positions)
    assert positions[-1] < removal
    assert script.index("rm -f /etc/systemd/system/") < removal


def deploy_script(monkeypatch, capsys):
    monkeypatch.setattr(deploy_sender, "repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(
        deploy_sender,
        "git_output",
        lambda root, *args: "" if args[0] == "status" else "abc123def",
    )
    lines = dry_run_lines(capsys, "deploy", "--name", "rel")
    return remote_scripts(lines)


def test_deploy_checks_for_lfs_remap_pointers_after_the_copy(monkeypatch, capsys):
    scripts = deploy_script(monkeypatch, capsys)
    checks = [s for s in scripts if "git-lfs.github.com/spec/v1" in s]
    assert len(checks) == 1
    assert "*.remap" in checks[0]
    assert scripts.index(checks[0]) == 1  # after prepare, before the build


@pytest.fixture
def lfs_check(monkeypatch, capsys, tmp_path):
    script = next(
        s
        for s in deploy_script(monkeypatch, capsys)
        if "git-lfs.github.com/spec/v1" in s
    )
    release = tmp_path / "release"
    (release / "src/configs").mkdir(parents=True)
    script = script.replace(RELEASE, str(release))
    assert str(release) in script

    def run():
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=False
        )

    return release, run


def test_lfs_check_rejects_only_pointer_remaps(lfs_check):
    release, run = lfs_check
    (release / "src/configs/good.remap").write_bytes(b"\x00\x01binary")
    (release / "src/configs/bad.remap").write_text(LFS_POINTER)
    (release / "src/configs/notes.txt").write_text(LFS_POINTER)
    result = run()
    assert result.returncode != 0
    assert str(release / "src/configs/bad.remap") in result.stderr
    assert "good.remap" not in result.stderr
    assert "notes.txt" not in result.stderr


def test_lfs_check_passes_without_pointers(lfs_check):
    release, run = lfs_check
    (release / "src/configs/good.remap").write_bytes(b"\x00\x01binary")
    result = run()
    assert result.returncode == 0, result.stderr
