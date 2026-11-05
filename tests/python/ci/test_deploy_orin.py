# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Orin deploy script; every subprocess call is mocked or runs against a temp git repo."""

import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import repo_root

spec = importlib.util.spec_from_file_location(
    "deploy_orin", repo_root() / "src/plugins/sensing/scripts/deploy_orin.py"
)
deploy_orin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_orin)

DeployError = deploy_orin.DeployError
STREAM = "--add-stream=sensor=0,livekit=left"


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class LoadConfigTests(TempDirTestCase):
    def load(self, config):
        path = self.tmp / "orin.json"
        path.write_text(json.dumps(config))
        return deploy_orin.load_config(path)

    def valid(self, **override):
        config = {
            "livekit_url": "wss://lk.example.org",
            "token_file": "/etc/isaaccapture/join.token",
            "args": [STREAM],
        }
        config.update(override)
        return config

    def test_accepts_valid_config(self):
        config = self.valid()
        self.assertEqual(self.load(config), config)

    def test_rejects_unknown_key(self):
        with self.assertRaises(DeployError):
            self.load(self.valid(extra=1))

    def test_rejects_missing_required_keys(self):
        for key in ("livekit_url", "token_file"):
            config = self.valid()
            del config[key]
            with self.subTest(key=key), self.assertRaises(DeployError):
                self.load(config)

    def test_rejects_relative_token_file(self):
        with self.assertRaisesRegex(DeployError, "absolute"):
            self.load(self.valid(token_file="join.token"))

    def test_rejects_missing_add_stream(self):
        for args in ([], ["--other=1"]):
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(DeployError, "add-stream"),
            ):
                self.load(self.valid(args=args))

    def test_rejects_url_and_token_flags_in_args(self):
        for flag in ("--livekit-url=wss://x", "--livekit-token-file=/t"):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(DeployError, "not in args"),
            ):
                self.load(self.valid(args=[STREAM, flag]))

    def test_rejects_non_string_url_and_token_file(self):
        for key in ("livekit_url", "token_file"):
            for value in (5, None, ["/t"], ""):
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaisesRegex(DeployError, "non-empty strings"),
                ):
                    self.load(self.valid(**{key: value}))

    def test_unreadable_or_invalid_json_is_a_deploy_error(self):
        with self.assertRaisesRegex(DeployError, "cannot read the config"):
            deploy_orin.load_config(self.tmp / "missing.json")
        bad = self.tmp / "bad.json"
        bad.write_text("{not json")
        with self.assertRaisesRegex(DeployError, "cannot read the config"):
            deploy_orin.load_config(bad)

    def test_rejects_non_object_and_bad_args(self):
        path = self.tmp / "list.json"
        path.write_text("[]")
        with self.assertRaises(DeployError):
            deploy_orin.load_config(path)
        with self.assertRaisesRegex(DeployError, "list of strings"):
            self.load(self.valid(args=[STREAM, 3]))


class UnitRenderTests(unittest.TestCase):
    def test_quote_escapes_systemd_specials(self):
        self.assertEqual(deploy_orin.systemd_quote("a%b"), "a%%b")
        self.assertEqual(deploy_orin.systemd_quote("$HOME"), "$$HOME")
        self.assertEqual(deploy_orin.systemd_quote("a\\b"), "a\\\\b")

    def test_quote_wraps_whitespace_and_quotes(self):
        self.assertEqual(deploy_orin.systemd_quote("a b"), '"a b"')
        self.assertEqual(deploy_orin.systemd_quote('a"b'), '"a\\"b"')
        self.assertEqual(deploy_orin.systemd_quote("plain"), "plain")

    def test_quote_wraps_empty_and_single_quote_args(self):
        self.assertEqual(deploy_orin.systemd_quote(""), '""')
        self.assertEqual(deploy_orin.systemd_quote("a'b"), '"a\'b"')

    def test_render_unit(self):
        config = {
            "livekit_url": "wss://lk.example.org",
            "token_file": "/etc/t oken",
            "args": [STREAM, "--name=100%", "--x=$Y"],
        }
        text = deploy_orin.render_unit(config, "/opt/ic", "bob", "video")
        exec_start = next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))
        self.assertEqual(
            exec_start,
            "ExecStart=/opt/ic/current/plugins/sensing_camera/camera_plugin_sensing "
            "--livekit-url=wss://lk.example.org "
            '"--livekit-token-file=/etc/t oken" '
            f"{STREAM} --name=100%% --x=$$Y",
        )
        self.assertIn("User=bob", text)
        self.assertIn("SupplementaryGroups=video", text)
        self.assertIn("WorkingDirectory=/opt/ic/current/plugins/sensing_camera", text)
        self.assertIsNone(re.search(r"@[A-Z_]+@", text))


def make_target(test, password=None):
    """A Target with its temp dir removed at cleanup; ORIN_PASSWORD is set or cleared explicitly."""
    env = {k: v for k, v in os.environ.items() if k != deploy_orin.PASSWORD_ENV}
    if password:
        env[deploy_orin.PASSWORD_ENV] = password
    with mock.patch.dict(os.environ, env, clear=True):
        target = deploy_orin.Target("u@orin")
    test.addCleanup(shutil.rmtree, target._dir, True)
    return target


class ValidateConnectionTests(unittest.TestCase):
    def validate(self, result):
        target = make_target(self)
        with mock.patch.object(deploy_orin.subprocess, "run", return_value=result):
            target.validate_connection()

    def test_ssh_failure_surfaces_stderr(self):
        with self.assertRaisesRegex(DeployError, "Permission denied"):
            self.validate(completed(255, stderr="Permission denied (publickey)\n"))

    def test_missing_python3(self):
        with self.assertRaisesRegex(DeployError, "python3 is not installed"):
            self.validate(completed(1))

    def test_non_aarch64(self):
        with self.assertRaisesRegex(DeployError, "x86_64"):
            self.validate(completed(0, stdout="x86_64\n"))

    def test_success(self):
        self.validate(completed(0, stdout="aarch64\n"))


class PasswordTests(unittest.TestCase):
    PASSWORD = "s3cr3t-pw"

    def test_password_never_in_argv(self):
        target = make_target(self, self.PASSWORD)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))
            return completed(0, stdout="aarch64\n")

        with mock.patch.object(deploy_orin.subprocess, "run", fake_run):
            target.validate_connection()
            target.capture("id -un")
            target.readable("/t")
            target.ssh("true")
            target.sudo("python3 x remote activate --owner bob")
            target.put(Path("/tmp/a"), "/r/a")
            target.get("/r/a", Path("/tmp/a"))
            askpass_text = Path(target._env["SSH_ASKPASS"]).read_text()
            target.close()
        self.assertGreater(len(calls), 7)
        for cmd, _ in calls:
            self.assertFalse([w for w in cmd if self.PASSWORD in w], cmd)
        _, sudo_kwargs = next(
            (c, k) for c, k in calls if "sudo -S -p '' python3" in c[-1]
        )
        self.assertEqual(sudo_kwargs["input"], self.PASSWORD + "\n")
        self.assertNotIn(self.PASSWORD, askpass_text)

    def test_without_password_sudo_uses_tty_and_no_askpass(self):
        target = make_target(self)
        with mock.patch.object(
            deploy_orin.subprocess, "run", return_value=completed(0)
        ) as run:
            target.sudo("true")
        self.assertIn("-t", run.call_args.args[0])
        self.assertNotIn("SSH_ASKPASS", target._env)
        self.assertNotIn("input", run.call_args.kwargs)

    def test_sudo_failure_raises(self):
        target = make_target(self, self.PASSWORD)
        with mock.patch.object(
            deploy_orin.subprocess, "run", return_value=completed(1)
        ):
            with self.assertRaisesRegex(DeployError, "sudo"):
                target.sudo("true")


class DeployOrderTests(TempDirTestCase):
    def run_deploy(self, validate_effect=None):
        config = {"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]}
        args = deploy_orin.parser().parse_args(
            ["u@orin", "--backend", "sipl", "--config", "orin.json", "--no-activate"]
        )
        manager = mock.Mock()
        manager.make_snapshot.return_value = {
            "base_revision": "a" * 40,
            "dirty": False,
            "sha256": "b" * 64,
        }
        manager.target.validate_connection.side_effect = validate_effect
        manager.target.capture.return_value = "x"
        manager.target.readable.return_value = True
        with (
            mock.patch.object(deploy_orin, "repo_root", return_value=self.tmp),
            mock.patch.object(deploy_orin, "make_snapshot", manager.make_snapshot),
        ):
            try:
                deploy_orin.run_deploy(args, config, manager.target)
            except DeployError:
                pass
        return [c[0] for c in manager.mock_calls]

    def test_validates_connection_first(self):
        names = self.run_deploy()
        self.assertEqual(names[0], "target.validate_connection")
        self.assertIn("make_snapshot", names)
        self.assertIn("target.put", names)

    def test_failed_validation_blocks_everything_else(self):
        names = self.run_deploy(validate_effect=DeployError("no route"))
        self.assertEqual(names, ["target.validate_connection"])

    def test_deploy_closes_target_on_failure(self):
        config = self.tmp / "orin.json"
        config.write_text(
            json.dumps({"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]})
        )
        args = deploy_orin.parser().parse_args(
            ["u@orin", "--backend", "sipl", "--config", str(config)]
        )
        target = mock.Mock()
        with (
            mock.patch.object(deploy_orin, "Target", return_value=target),
            mock.patch.object(deploy_orin, "run_deploy", side_effect=DeployError("x")),
        ):
            with self.assertRaises(DeployError):
                deploy_orin.deploy(args)
        target.close.assert_called_once()


class SnapshotTests(TempDirTestCase):
    def git(self, *args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.org",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        (self.repo / "sub").mkdir(parents=True)
        self.git("init", "-q")
        (self.repo / "a.txt").write_text("one\n")
        (self.repo / "sub/b.txt").write_text("two\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")
        self.dest = self.tmp / "snap.tar.gz"

    def members(self):
        with tarfile.open(self.dest) as tar:
            return {
                m.name: tar.extractfile(m).read().decode()
                for m in tar.getmembers()
                if m.isfile()
            }

    def test_clean_tree_ignores_untracked(self):
        (self.repo / "untracked.txt").write_text("x")
        meta = deploy_orin.make_snapshot(self.repo, self.dest)
        self.assertEqual(self.members(), {"a.txt": "one\n", "sub/b.txt": "two\n"})
        self.assertFalse(meta["dirty"])
        self.assertEqual(meta["base_revision"], self.git("rev-parse", "HEAD"))
        self.assertEqual(meta["sha256"], deploy_orin.sha256_file(self.dest))

    def test_tracked_edit_is_included_and_marks_dirty(self):
        (self.repo / "a.txt").write_text("edited\n")
        (self.repo / "untracked.txt").write_text("x")
        meta = deploy_orin.make_snapshot(self.repo, self.dest)
        self.assertEqual(self.members()["a.txt"], "edited\n")
        self.assertNotIn("untracked.txt", self.members())
        self.assertTrue(meta["dirty"])


class ActivateTests(TempDirTestCase):
    """remote_activate with subprocess faked; the unit path is redirected into the temp dir."""

    def setUp(self):
        super().setUp()
        self.prefix = self.tmp / "opt"
        self.prefix.mkdir()
        self.unit = self.tmp / "in.service"
        self.unit.write_text("[Unit]\n")
        # Path("/etc/systemd/system") / <absolute path> resolves to the absolute path.
        self.service = str(self.tmp / "isaaccapture-producer.service")
        self.calls = []
        self.events = []
        self.journal = ""
        self.unmanaged = []
        for p in (
            mock.patch.object(
                deploy_orin, "unmanaged_producers", lambda: list(self.unmanaged)
            ),
            mock.patch.object(deploy_orin, "SERVICE", self.service),
            mock.patch.object(deploy_orin.subprocess, "run", self.fake_run),
            mock.patch.object(deploy_orin.time, "sleep"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def link_target(self):
        link = self.prefix / "current"
        return os.readlink(link) if link.is_symlink() else None

    def fake_run(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd[:2] == ["systemctl", "stop"]:
            self.events.append(("stop", self.link_target()))
        elif cmd[:2] == ["systemctl", "start"]:
            self.events.append(("start", self.link_target()))
        elif cmd[0] == "journalctl":
            return completed(0, stdout=self.journal)
        return completed(0)

    def make_archive(self, good=True):
        stage = self.tmp / "stage"
        (stage / "bin").mkdir(parents=True)
        (stage / "bin/producer").write_text("binary")
        digest = deploy_orin.sha256_file(stage / "bin/producer") if good else "0" * 64
        (stage / "SHA256SUMS").write_text(f"{digest}  ./bin/producer\n")
        archive = self.tmp / "producer.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=".")
        return archive

    def activate(self, archive, timeout=60, replace_unmanaged=False):
        deploy_orin.remote_activate(
            argparse.Namespace(
                archive=str(archive),
                prefix=str(self.prefix),
                release="r1",
                unit=str(self.unit),
                owner="bob",
                ready_timeout=timeout,
                replace_unmanaged=replace_unmanaged,
            )
        )

    def test_checksum_mismatch_aborts_before_stop(self):
        with self.assertRaisesRegex(DeployError, "checksum mismatch"):
            self.activate(self.make_archive(good=False))
        self.assertFalse([c for c in self.calls if c[0] == "systemctl"])
        self.assertIsNone(self.link_target())

    def test_checksum_mismatch_removes_the_new_release_dir(self):
        with self.assertRaisesRegex(DeployError, "checksum mismatch"):
            self.activate(self.make_archive(good=False))
        self.assertFalse((self.prefix / "releases" / "r1").exists())
        self.assertFalse([c for c in self.calls if c[0] == "chown"])

    def test_unmanaged_producer_refuses_before_installing_anything(self):
        self.unmanaged = [(4242, "alice"), (4243, "bob")]
        with mock.patch.object(deploy_orin.os, "kill") as kill:
            with self.assertRaises(DeployError) as ctx:
                self.activate(self.make_archive())
        message = str(ctx.exception)
        self.assertIn("pid 4242 (user alice)", message)
        self.assertIn("pid 4243 (user bob)", message)
        self.assertIn("--replace-unmanaged", message)
        kill.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.prefix / "releases").exists())
        self.assertIsNone(self.link_target())
        self.assertFalse(Path(self.service).exists())

    def test_replace_unmanaged_interrupts_only_listed_pids_between_stop_and_start(
        self,
    ):
        self.journal = "Joined LiveKit room x\nstats livekit 12 published\n"
        self.unmanaged = [(4242, "alice")]
        order = []
        self.events = order

        def fake_kill(pid, sig):
            order.append(("kill", pid, sig))

        def fake_stop(procs, timeout_s=20):
            real_kill = deploy_orin.os.kill
            for pid, _ in procs:
                real_kill(pid, deploy_orin.signal.SIGINT)

        with mock.patch.object(deploy_orin.os, "kill", fake_kill):
            with mock.patch.object(
                deploy_orin, "stop_unmanaged", side_effect=fake_stop
            ):
                self.activate(self.make_archive(), replace_unmanaged=True)
        self.assertEqual([e[0] for e in order], ["stop", "kill", "start"], order)
        self.assertEqual(order[1], ("kill", 4242, signal.SIGINT))
        self.assertEqual(order[2], ("start", "releases/r1"))

    def test_without_unmanaged_nothing_is_killed(self):
        self.journal = "Joined LiveKit room x\nstats livekit 12 published\n"
        with mock.patch.object(deploy_orin.os, "kill") as kill:
            self.activate(self.make_archive(), replace_unmanaged=True)
        kill.assert_not_called()

    def test_success_switches_link_between_stop_and_start(self):
        self.journal = "Joined LiveKit room x\nstats livekit 12 published\n"
        replaced = []
        real_replace = os.replace

        def spy(src, dst):
            replaced.append((Path(src).name, Path(dst).name))
            real_replace(src, dst)

        with mock.patch.object(deploy_orin.os, "replace", spy):
            self.activate(self.make_archive())
        self.assertEqual(replaced, [("current.new", "current")])
        self.assertEqual(self.events, [("stop", None), ("start", "releases/r1")])
        self.assertFalse((self.prefix / "current.new").exists())

    def test_readiness_times_out_without_markers(self):
        self.journal = "Joined LiveKit room x\nno stats yet\n"
        ticks = iter(range(0, 1000, 10))
        with mock.patch.object(deploy_orin.time, "monotonic", lambda: next(ticks)):
            with mock.patch.object(deploy_orin, "print", create=True):
                with self.assertRaisesRegex(DeployError, "did not become ready"):
                    self.activate(self.make_archive(), timeout=25)
        self.assertEqual(self.events[-1], ("start", "releases/r1"))


class FakeStdout(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class ProgressTests(unittest.TestCase):
    def drawn(self, tty, total, updates):
        out = FakeStdout(tty)
        with mock.patch.object(deploy_orin.sys, "stdout", out):
            bar = deploy_orin.Progress("pack", total)
            for done in updates:
                bar.update(done)
            bar.finish()
        return out.getvalue()

    def test_non_tty_prints_at_most_one_line_per_ten_percent(self):
        text = self.drawn(False, 1000, range(0, 1001, 7))
        lines = text.splitlines()
        self.assertLessEqual(len(lines), 11)
        self.assertNotIn("\r", text)
        self.assertIn("100%", lines[-1])
        self.assertEqual(len(lines), len(set(lines)))

    def test_non_tty_finish_reports_100_once(self):
        lines = self.drawn(False, 4, [4]).splitlines()
        self.assertEqual(lines, ["    pack 100%"])

    def test_tty_draws_a_30_char_bar_redrawn_in_place(self):
        text = self.drawn(True, 10, [5])
        self.assertIn("\r    pack [" + "#" * 15 + "-" * 15 + "]  50% (5/10)", text)
        self.assertIn("[" + "#" * 30 + "] 100% (10/10)", text)
        self.assertTrue(text.endswith("\n"))

    def test_scp_is_quiet_only_when_stdout_is_not_a_tty(self):
        for tty, expected in ((False, ["-q"]), (True, [])):
            with (
                self.subTest(tty=tty),
                mock.patch.object(deploy_orin.sys, "stdout", FakeStdout(tty)),
            ):
                self.assertEqual(deploy_orin.Target._scp_quiet(), expected)

    def test_put_and_get_pass_quiet_flag_to_scp(self):
        target = make_target(self)
        with (
            mock.patch.object(deploy_orin.sys, "stdout", FakeStdout(False)),
            mock.patch.object(
                deploy_orin.subprocess, "run", return_value=completed(0)
            ) as run,
        ):
            target.put(Path("/tmp/a"), "/r/a")
            target.get("/r/a", Path("/tmp/a"))
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], "scp")
            self.assertIn("-q", call.args[0])

    def test_step_heading_format(self):
        out = FakeStdout(False)
        with mock.patch.object(deploy_orin.sys, "stdout", out):
            deploy_orin.step(2, "Preflight")
        self.assertEqual(out.getvalue(), f"==> [2/{deploy_orin.STEPS}] Preflight\n")


class StopUnmanagedTests(unittest.TestCase):
    def test_sigints_exactly_the_listed_pids_and_returns_when_gone(self):
        with (
            mock.patch.object(deploy_orin.os, "kill") as kill,
            mock.patch.object(Path, "exists", return_value=False),
        ):
            deploy_orin.stop_unmanaged([(11, "a"), (12, "b")])
        self.assertEqual(
            kill.call_args_list,
            [mock.call(11, signal.SIGINT), mock.call(12, signal.SIGINT)],
        )

    def test_raises_when_a_pid_survives(self):
        ticks = iter(range(0, 1000, 10))
        with (
            mock.patch.object(deploy_orin.os, "kill"),
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(deploy_orin.time, "sleep"),
            mock.patch.object(deploy_orin.time, "monotonic", lambda: next(ticks)),
        ):
            with self.assertRaisesRegex(DeployError, r"\[11\]"):
                deploy_orin.stop_unmanaged([(11, "a")], timeout_s=25)


class SdkDirsTests(unittest.TestCase):
    def test_default_uses_system_locations(self):
        self.assertEqual(
            deploy_orin.sdk_dirs(""),
            (
                Path("/usr/src/jetson_multimedia_api"),
                Path("/usr/src/jetson_sipl_api"),
            ),
        )

    def test_sdk_root_prefixes_usr_src(self):
        self.assertEqual(
            deploy_orin.sdk_dirs("/mnt/sdk"),
            (
                Path("/mnt/sdk/usr/src/jetson_multimedia_api"),
                Path("/mnt/sdk/usr/src/jetson_sipl_api"),
            ),
        )

    def test_cli_parsers_accept_sdk_root_and_replace_unmanaged(self):
        args = deploy_orin.parser().parse_args(
            ["u@orin", "--backend", "sipl", "--config", "c", "--sdk-root", "/s"]
        )
        self.assertEqual(args.sdk_root, "/s")
        self.assertFalse(args.replace_unmanaged)
        self.assertEqual(
            deploy_orin.parser()
            .parse_args(["h", "--backend", "argus", "--config", "c"])
            .sdk_root,
            "",
        )
        remote = deploy_orin.remote_parser()
        for step, extra in (
            ("preflight", []),
            ("build", ["--src", "s", "--stage", "t"]),
        ):
            parsed = remote.parse_args(
                [step, "--backend", "sipl", "--sdk-root", "/s", *extra]
            )
            self.assertEqual(parsed.sdk_root, "/s")

    def test_preflight_names_the_sdk_root_path_when_headers_are_missing(self):
        args = argparse.Namespace(backend="sipl", sdk_root="/nonexistent-sdk")
        with (
            mock.patch.object(deploy_orin.os, "uname") as uname,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value="# R36\n"),
            mock.patch.object(deploy_orin.os, "statvfs") as statvfs,
            mock.patch.object(deploy_orin.subprocess, "run", return_value=completed(0)),
        ):
            uname.return_value.machine = "aarch64"
            statvfs.return_value.f_bavail = 1 << 30
            statvfs.return_value.f_frsize = 4096
            with self.assertRaisesRegex(DeployError, "/nonexistent-sdk/usr/src"):
                deploy_orin.remote_preflight(args)


class RemoteBuildTests(TempDirTestCase):
    LIBCUDA = "/opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1"

    def build(self, backend="sipl", sdk_root="", libcuda=False, path="/usr/bin"):
        stage = self.tmp / "stage"
        (stage / deploy_orin.BIN_REL).parent.mkdir(parents=True, exist_ok=True)
        (stage / deploy_orin.BIN_REL).write_text("bin")
        args = argparse.Namespace(
            src="/src",
            backend=backend,
            stage=str(stage),
            jobs=3,
            sdk_root=sdk_root,
        )
        real_exists, real_read_text = Path.exists, Path.read_text

        def fake_exists(p, *a, **k):
            return libcuda if str(p) == self.LIBCUDA else real_exists(p, *a, **k)

        def fake_read_text(p, *a, **k):
            if str(p) == "/etc/nv_tegra_release":
                return "# R36 (release)\n"
            return real_read_text(p, *a, **k)

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))
            return completed(0)

        with (
            mock.patch.object(deploy_orin, "CACHE", str(self.tmp / "cache")),
            mock.patch.object(deploy_orin, "run", fake_run),
            mock.patch.object(deploy_orin, "tail", return_value="x"),
            mock.patch.object(Path, "exists", fake_exists),
            mock.patch.object(Path, "read_text", fake_read_text),
            mock.patch.dict(os.environ, {"PATH": path}),
        ):
            deploy_orin.remote_build(args)
        return calls

    def configure(self, calls):
        return next(c for c in calls if c[0][:2] == ["cmake", "-S"])

    def test_sipl_configure_flags_and_cuda_path(self):
        cmd, kwargs = self.configure(self.build(sdk_root="/mnt/sdk"))
        for flag in (
            "-DSENSING_WITH_SIPL=ON",
            "-DSENSING_WITH_ARGUS=OFF",
            "-DJETSON_MMAPI_DIR=/mnt/sdk/usr/src/jetson_multimedia_api",
            "-DSIPL_API_DIR=/mnt/sdk/usr/src/jetson_sipl_api",
            "-DENABLE_CLANG_FORMAT_CHECK=OFF",
            "-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include",
        ):
            self.assertIn(flag, cmd)
        self.assertEqual(kwargs["env"]["PATH"], "/usr/local/cuda/bin:/usr/bin")

    def test_build_step_also_gets_the_cuda_path(self):
        calls = self.build()
        cmd, kwargs = next(c for c in calls if c[0][:2] == ["cmake", "--build"])
        self.assertEqual(kwargs["env"]["PATH"], "/usr/local/cuda/bin:/usr/bin")
        self.assertIn("camera_plugin_sensing", cmd)

    def test_argus_has_no_sipl_dir(self):
        cmd, _ = self.configure(self.build(backend="argus"))
        self.assertIn("-DSENSING_WITH_ARGUS=ON", cmd)
        self.assertIn("-DJETSON_MMAPI_DIR=/usr/src/jetson_multimedia_api", cmd)
        self.assertFalse([a for a in cmd if a.startswith("-DSIPL_API_DIR")])

    def test_linker_flag_only_when_gpu_libcuda_exists(self):
        absent, _ = self.configure(self.build(libcuda=False))
        self.assertFalse([a for a in absent if "LINKER_FLAGS" in a])
        present, _ = self.configure(self.build(libcuda=True))
        self.assertIn(
            f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed {self.LIBCUDA}", present
        )


if __name__ == "__main__":
    unittest.main()
