# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
OWNED_UNITS = (
    "camera-argus-sender",
    "camera-time-sync",
    "camera-vi-error-trace",
)
OWNED_FILES = (
    "etc/systemd/system/camera-argus-sender.service",
    "etc/systemd/system/camera-time-sync.service",
    "etc/systemd/system/camera-vi-error-trace.service",
    "etc/chrony/conf.d/camera-agx.conf",
    "etc/systemd/journald.conf.d/10-camera-persistent-journal.conf",
    "usr/local/lib/camera/lock_jetson_camera_clocks.sh",
)


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


@pytest.fixture
def board(tmp_path):
    install = tmp_path / "board/camera_streamer"
    system = tmp_path / "system"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_path = tmp_path / "services.json"
    log_path = tmp_path / "systemctl.log"
    state = {
        unit: {"load": "not-found", "enabled": "not-found", "active": "inactive"}
        for unit in OWNED_UNITS
    }
    for unit in ("nvargus-daemon", "chrony"):
        state[unit] = {"load": "loaded", "enabled": "enabled", "active": "active"}
    state_path.write_text(json.dumps(state))
    executable(
        bin_dir / "systemctl",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
path = pathlib.Path(os.environ['FAKE_SERVICES'])
state = json.loads(path.read_text())
args = sys.argv[1:]
with open(os.environ['FAKE_SYSTEMCTL_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
units = [arg.removesuffix('.service') for arg in args if arg.endswith('.service')]
if args[0] == 'show':
    unit = units[0]
    value = state.get(unit, {'load': 'not-found', 'enabled': '', 'active': 'inactive'})
    prop = args[args.index('-p') + 1]
    print(value[{'LoadState': 'load', 'UnitFileState': 'enabled', 'ActiveState': 'active'}[prop]])
else:
    for unit in units:
        value = state.setdefault(unit, {'load': 'loaded', 'enabled': 'disabled', 'active': 'inactive'})
        if args[0] in ('start', 'restart', 'try-restart'): value['active'] = 'active'
        if args[0] == 'stop': value['active'] = 'inactive'
        if args[0] == 'enable': value['enabled'] = 'enabled'
        if args[0] == 'disable': value['enabled'] = 'disabled'
    path.write_text(json.dumps(state))
""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAMERA_DEPLOY_SYSTEM_ROOT": str(system),
        "FAKE_SERVICES": str(state_path),
        "FAKE_SYSTEMCTL_LOG": str(log_path),
    }

    def run(action, *args, check=True):
        return subprocess.run(
            ["bash", str(DEPLOY / "deployment_state.sh"), action, str(install), *args],
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )

    return install, system, state_path, log_path, run, bin_dir


def write_owned_files(system, prefix):
    for name in OWNED_FILES:
        path = system / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{prefix}: {name}\n")


def test_restore_first_install_removes_only_deployment_files(board):
    install, system, state_path, log_path, run, _ = board
    driver = system / "etc/camera-driver.json"
    driver.parent.mkdir(parents=True)
    driver.write_text("installed driver configuration")
    run("snapshot")
    write_owned_files(system, "new")
    (install / "argus_sender").mkdir()
    (install / "argus_sender/binary").write_text("new")
    state = json.loads(state_path.read_text())
    state["nvargus-daemon"].update(enabled="disabled", active="inactive")
    for unit in OWNED_UNITS:
        state[unit].update(load="loaded", enabled="enabled", active="active")
    state_path.write_text(json.dumps(state))

    run("restore")

    assert not (install / "argus_sender").exists()
    assert all(not (system / name).exists() for name in OWNED_FILES)
    assert driver.read_text() == "installed driver configuration"
    state = json.loads(state_path.read_text())
    assert state["nvargus-daemon"]["enabled"] == "enabled"
    assert state["nvargus-daemon"]["active"] == "active"
    assert all(state[unit]["active"] == "inactive" for unit in OWNED_UNITS)
    assert "camstream" not in log_path.read_text()
    assert "cam-init" not in log_path.read_text()
    previous_log = log_path.read_text()
    run("restore")
    assert log_path.read_text() == previous_log


@pytest.mark.parametrize("previous_active", ["active", "inactive"])
def test_restore_previous_sender_files_and_service_state(board, previous_active):
    install, system, state_path, _, run, _ = board
    for name in (
        "argus_sender/build/argus_sender",
        "argus_sender/remaps/cam4.remap",
        "argus_sender/tests/test.py",
    ):
        path = install / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("previous release")
    independent_test = install / "tests/integration.py"
    independent_test.parent.mkdir()
    independent_test.write_text("independent integration test")
    write_owned_files(system, "previous")
    state = json.loads(state_path.read_text())
    state["nvargus-daemon"].update(enabled="disabled", active="inactive")
    for unit in OWNED_UNITS:
        state[unit].update(load="loaded", enabled="disabled", active=previous_active)
    state_path.write_text(json.dumps(state))
    run("snapshot")
    (install / "argus_sender/build/argus_sender").write_text("new binary")
    (install / "argus_sender/remaps/cam4.remap").write_text("new calibration")
    independent_test.write_text("independent work after deployment")
    write_owned_files(system, "new")
    state = json.loads(state_path.read_text())
    for unit in OWNED_UNITS:
        state[unit].update(enabled="enabled", active="active")
    state_path.write_text(json.dumps(state))

    run("restore")

    assert (
        install / "argus_sender/build/argus_sender"
    ).read_text() == "previous release"
    assert (
        install / "argus_sender/remaps/cam4.remap"
    ).read_text() == "previous release"
    assert (install / "argus_sender/tests/test.py").read_text() == "previous release"
    assert independent_test.read_text() == "independent work after deployment"
    assert all(
        (system / name).read_text() == f"previous: {name}\n" for name in OWNED_FILES
    )
    state = json.loads(state_path.read_text())
    assert state["nvargus-daemon"]["active"] == "inactive"
    assert all(state[unit]["active"] == previous_active for unit in OWNED_UNITS)
    assert all(state[unit]["enabled"] == "disabled" for unit in OWNED_UNITS)


def test_activate_swaps_the_staged_release_and_can_restore_it(board):
    install, _, _, _, run, _ = board
    old_binary = install / "argus_sender/build/argus_sender"
    old_binary.parent.mkdir(parents=True)
    old_binary.write_text("previous binary")
    stage = install.parent / ".camera-streamer-stage.TEST"
    new_binary = stage / "argus_sender/build/argus_sender"
    new_binary.parent.mkdir(parents=True)
    executable(new_binary, "#!/usr/bin/env bash\nexit 0\n")
    (stage / "argus_sender/tests").mkdir()
    result = run("activate", str(stage), check=False)
    assert result.returncode != 0
    assert old_binary.read_text() == "previous binary"
    run("snapshot")
    run("activate", str(stage))
    assert old_binary.read_text() == "#!/usr/bin/env bash\nexit 0\n"
    assert not new_binary.exists()
    run("restore")
    assert old_binary.read_text() == "previous binary"


def test_restore_without_snapshot_does_not_touch_services(board):
    _, _, _, log_path, run, _ = board
    result = run("restore", check=False)
    assert result.returncode != 0
    assert not log_path.exists()


def test_failed_snapshot_preserves_existing_restore_point(board):
    install, _, _, _, run, bin_dir = board
    (install / "argus_sender").mkdir(parents=True)
    (install / "argus_sender/binary").write_text("original")
    run("snapshot")
    executable(bin_dir / "cp", "#!/usr/bin/env bash\nexit 42\n")
    result = run("snapshot", check=False)
    assert result.returncode == 42
    assert (
        install / ".deployment-backup/argus_sender/binary"
    ).read_text() == "original"
    assert (install / ".deployment-backup/complete").exists()


@pytest.fixture
def remote_deploy(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "ssh.log"
    executable(
        bin_dir / "ssh",
        """#!/usr/bin/env python3
import json, os, sys
cmd = sys.argv[-1]
payload = sys.stdin.read()
with open(os.environ['FAKE_SSH_LOG'], 'a') as log:
    log.write(json.dumps({'command': cmd, 'input': payload}) + '\\n')
if cmd == 'bash -s' and "printf 'USER=" in payload:
    print('USER=nvidia\\nHOME=/home/nvidia')
elif cmd.startswith('mktemp '):
    print('/home/nvidia/.camera-streamer-stage.TEST')
elif cmd.startswith('sudo -n cat '):
    sys.exit(1)
elif cmd.startswith('cmake ') and os.environ.get('FAIL_AT') == 'build':
    sys.exit(7)
elif 'systemctl reset-failed camera-argus-sender.service &&' in cmd and os.environ.get('FAIL_AT') == 'activation':
    sys.exit(17)
""",
    )
    executable(bin_dir / "rsync", "#!/usr/bin/env bash\nexit 0\n")

    def run(*args, fail_at=""):
        result = subprocess.run(
            [
                "bash",
                str(DEPLOY / "deploy_argus_sender.sh"),
                "camera-board",
                "--host-lan-ip",
                "192.0.2.10",
                *args,
            ],
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "FAKE_SSH_LOG": str(log_path),
                "FAIL_AT": fail_at,
            },
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        calls = [json.loads(line) for line in log_path.read_text().splitlines()]
        return result, calls

    return run


@pytest.mark.parametrize(
    "fail_at,expected", [("", 0), ("build", 7), ("activation", 17)]
)
def test_deployment_stages_build_and_restores_after_activation_failure(
    remote_deploy, fail_at, expected
):
    result, calls = remote_deploy(fail_at=fail_at)
    assert result.returncode == expected, result.stderr
    commands = [call["command"] for call in calls]
    actions = [
        command.split(" -- ", 1)[1].split()[0].strip("'")
        for command in commands
        if command.startswith("sudo -n bash -s -- ")
    ]
    if fail_at == "build":
        assert actions == []
    elif fail_at == "activation":
        assert actions == ["snapshot", "activate", "restore"]
    else:
        assert actions == ["snapshot", "activate"]
    assert all(
        "camstream" not in command and "cam-init" not in command for command in commands
    )
    assert not any("rb_camera.sh" in call["input"] for call in calls)


@pytest.mark.parametrize("flag", ["--rollback", "--restore"])
def test_explicit_restore_does_not_require_build_or_driver_layout(remote_deploy, flag):
    result, calls = remote_deploy(flag)
    assert result.returncode == 0, result.stderr
    commands = [call["command"] for call in calls]
    assert any(" -- 'restore' " in command for command in commands)
    assert not any(command.startswith(("cmake ", "mktemp ")) for command in commands)


def test_deployment_check_is_read_only(remote_deploy):
    result, calls = remote_deploy("--check")
    assert result.returncode == 0, result.stderr
    forbidden = (
        "systemctl restart ",
        "systemctl stop ",
        "systemctl enable ",
        "systemctl disable ",
        "sudo -n install ",
        "mktemp ",
    )
    assert not any(any(word in call["command"] for word in forbidden) for call in calls)
    assert not any("bash -s --" in call["command"] for call in calls)


def test_sender_unit_requires_only_platform_camera_ownership():
    unit = (DEPLOY / "camera-argus-sender.service.in").read_text()
    assert "cam-init" not in unit
    assert "camstream" not in unit
    assert "Conflicts=nvargus-daemon.service" in unit
