# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Orin deploy script; every subprocess call is mocked or runs against a temp git repo."""

import argparse
import contextlib
import hashlib
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

    def test_rejects_only_one_of_url_and_token_file(self):
        for key in ("livekit_url", "token_file"):
            config = self.valid()
            del config[key]
            with (
                self.subTest(only_missing=key),
                self.assertRaisesRegex(DeployError, "go together"),
            ):
                self.load(config)

    def test_accepts_config_with_only_args(self):
        config = {"args": [STREAM]}
        self.assertEqual(self.load(config), config)

    def test_args_are_still_required_without_livekit(self):
        with self.assertRaisesRegex(DeployError, "add-stream"):
            self.load({})

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

    def test_render_unit_without_livekit_args(self):
        text = deploy_orin.render_unit({"args": [STREAM]}, "/opt/ic", "bob", "video")
        exec_start = next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))
        self.assertEqual(
            exec_start,
            "ExecStart=/opt/ic/current/plugins/sensing_camera/camera_plugin_sensing "
            + STREAM,
        )
        self.assertNotIn("--livekit", text)


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
            askpass_text = Path(target._env["SSH_ASKPASS"]).read_text()
            target.close()
        self.assertGreater(len(calls), 6)
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


CACHE_DIR = "/home/u/.cache/isaaccapture-deploy"


def git_repo(test, files):
    """A temp git repo holding the given {path: text} tracked files; returns (root, git runner)."""
    root = Path(tempfile.mkdtemp(dir=test.tmp))

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.org",
             "-c", "commit.gpgsign=false", *args],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip

    git("init", "-q")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return root, git


class FakeOrin:
    """A mock Target that records uploads and keeps the source manifest an `apply` would write."""

    def __init__(self, manifest=None):
        self.manifest = manifest
        self.events = []
        self.deltas = []
        self.pending = None
        self.target = mock.Mock()
        self.target.host = "u@orin"
        self.target.readable.return_value = True
        self.target.capture.side_effect = self.capture
        self.target.put.side_effect = self.put
        self.target.ssh.side_effect = self.ssh

    def capture(self, cmd):
        if "pwd" in cmd:
            return CACHE_DIR
        if "src.manifest.json" in cmd:
            return json.dumps(self.manifest) if self.manifest is not None else ""
        return "u"

    def put(self, local, remote):
        self.events.append(("put", remote))
        if remote.endswith("delta.tar.gz"):
            with tarfile.open(local) as tar:
                names = tar.getnames()
                self.pending = json.loads(
                    tar.extractfile(f"{deploy_orin.DEPLOY_DIR}/manifest.json").read()
                )
            self.deltas.append(sorted(n for n in names if not n.startswith(".deploy/")))

    def ssh(self, cmd):
        self.events.append(("ssh", cmd))
        if " apply " in cmd:
            self.manifest = self.pending

    def run(self, root, *extra):
        args = deploy_orin.parser().parse_args(
            [
                "u@orin",
                "--config",
                "orin.json",
                "--no-activate",
                *extra,
            ]
        )
        config = {"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]}
        with mock.patch.object(deploy_orin, "repo_root", return_value=root):
            deploy_orin.run_deploy(args, config, self.target)


class DeployOrderTests(TempDirTestCase):
    def test_failed_validation_blocks_everything_else(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        orin.target.validate_connection.side_effect = DeployError("no route")
        with self.assertRaises(DeployError):
            orin.run(root)
        self.assertEqual(orin.target.mock_calls[0][0], "validate_connection")
        self.assertEqual(len(orin.target.mock_calls), 1)

    def test_helper_uploaded_and_preflight_run_before_any_delta(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n", "README.md": "r\n"})
        orin = FakeOrin()
        orin.run(root)
        kinds = []
        for kind, value in orin.events:
            if kind == "put" and value.endswith("/deploy_orin.py"):
                kinds.append("helper")
            elif kind == "ssh" and " preflight " in value:
                kinds.append("preflight")
            elif kind == "put" and value.endswith("delta.tar.gz"):
                kinds.append("delta")
        self.assertEqual(kinds, ["helper", "preflight", "delta"])
        apply_cmd = next(v for k, v in orin.events if k == "ssh" and " apply " in v)
        self.assertIn("--fresh", apply_cmd)

    def test_unchanged_second_run_sends_nothing_and_one_change_sends_only_it(self):
        root, _ = git_repo(
            self, {"src/a.cpp": "a\n", "src/b.cpp": "b\n", "README.md": "r\n"}
        )
        orin = FakeOrin()
        orin.run(root)
        self.assertEqual(orin.deltas, [["README.md", "src/a.cpp", "src/b.cpp"]])

        orin.run(root)
        self.assertEqual(len(orin.deltas), 1)

        (root / "src/b.cpp").write_text("changed\n")
        orin.run(root)
        self.assertEqual(orin.deltas[1:], [["src/b.cpp"]])
        applies = [v for k, v in orin.events if k == "ssh" and " apply " in v]
        self.assertIn("--fresh", applies[0])
        self.assertNotIn("--fresh", applies[-1])

    def test_same_src_path_for_different_edits(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        orin.run(root)
        (root / "src/a.cpp").write_text("z\n")
        orin.run(root)
        srcs = {
            m.group(1)
            for k, v in orin.events
            if k == "ssh" and (m := re.search(r"--src (\S+)", v))
        }
        self.assertEqual(srcs, {f"{CACHE_DIR}/src"})

    def test_cmake_args_are_forwarded_to_the_remote_build_in_order(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        orin.run(root, "--cmake-arg=-DA=1", "--cmake-arg=-DB=a b")
        build = next(v for k, v in orin.events if k == "ssh" and " build " in v)
        self.assertTrue(build.endswith(" --cmake-arg=-DA=1 --cmake-arg='-DB=a b'"))

    def test_activate_command_line_without_livekit_appends_no_livekit(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        args = deploy_orin.parser().parse_args(["u@orin", "--config", "orin.json"])
        config = {"args": [STREAM]}
        with mock.patch.object(deploy_orin, "repo_root", return_value=root):
            deploy_orin.run_deploy(args, config, orin.target)
        (activate,), _ = orin.target.sudo.call_args
        self.assertTrue((activate + " ").count(" --no-livekit ") == 1)

    def test_activate_command_line_uses_the_module_constants(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        args = deploy_orin.parser().parse_args(["u@orin", "--config", "orin.json"])
        config = {"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]}
        with mock.patch.object(deploy_orin, "repo_root", return_value=root):
            deploy_orin.run_deploy(args, config, orin.target)
        (activate,), _ = orin.target.sudo.call_args
        self.assertIn(" activate ", activate)
        self.assertIn("--prefix /opt/isaaccapture", activate)
        self.assertIn("--ready-timeout 60", activate)
        self.assertIn("--sensors 0 ", activate + " ")
        self.assertNotIn("--no-livekit", activate)
        self.assertEqual(
            (
                deploy_orin.PREFIX,
                deploy_orin.SERVICE_GROUPS,
                deploy_orin.READY_TIMEOUT_S,
            ),
            ("/opt/isaaccapture", "video", 60),
        )

    def test_unit_is_rendered_with_prefix_and_service_groups(self):
        root, _ = git_repo(self, {"src/a.cpp": "a\n"})
        orin = FakeOrin()
        with mock.patch.object(
            deploy_orin, "render_unit", return_value="unit\n"
        ) as render:
            orin.run(root)
        _, prefix, _, groups = render.call_args.args
        self.assertEqual(
            (prefix, groups), (deploy_orin.PREFIX, deploy_orin.SERVICE_GROUPS)
        )

    def test_deploy_closes_target_on_failure(self):
        config = self.tmp / "orin.json"
        config.write_text(
            json.dumps({"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]})
        )
        args = deploy_orin.parser().parse_args(["u@orin", "--config", str(config)])
        target = mock.Mock()
        with (
            mock.patch.object(deploy_orin, "Target", return_value=target),
            mock.patch.object(deploy_orin, "run_deploy", side_effect=DeployError("x")),
        ):
            with self.assertRaises(DeployError):
                deploy_orin.deploy(args)
        target.close.assert_called_once()


class OpenSourceTests(TempDirTestCase):
    def args(self, *extra):
        return deploy_orin.parser().parse_args(["u@orin", "--config", "c", *extra])

    def repo_with_branch(self):
        root, git = git_repo(
            self, {"src/a.cpp": "committed\n", "run.sh": "#!/bin/sh\n"}
        )
        (root / "run.sh").chmod(0o755)
        git("add", "run.sh")
        git("commit", "-q", "-m", "exec")
        git("branch", "feature")
        (root / "src/a.cpp").write_text("edited later\n")
        (root / "untracked.txt").write_text("x")
        return root, git

    def test_cli_parses_source_and_branch(self):
        args = self.args("--source", "/w", "--branch", "feat")
        self.assertEqual((args.source, args.branch), ("/w", "feat"))
        args = self.args()
        self.assertFalse(args.source)
        self.assertFalse(args.branch)

    def test_default_yields_script_checkout_with_local_edits(self):
        root, _ = git_repo(self, {"a.txt": "1\n", "b.txt": "2\n"})
        (root / "a.txt").write_text("edited\n")
        (root / "untracked.txt").write_text("x")
        with mock.patch.object(deploy_orin, "repo_root", return_value=root):
            with deploy_orin.open_source(self.args()) as (src, files, state):
                self.assertEqual(src, root)
                self.assertEqual(files, ["a.txt", "b.txt"])
                self.assertTrue(state["dirty"])

    def test_source_option_uses_another_checkout(self):
        root, git = git_repo(self, {"a.txt": "1\n"})
        sub = root / "sub"
        sub.mkdir()
        with mock.patch.object(
            deploy_orin, "repo_root", side_effect=AssertionError("own checkout used")
        ):
            with deploy_orin.open_source(self.args("--source", str(sub))) as (
                src,
                files,
                state,
            ):
                self.assertEqual(src.resolve(), root.resolve())
                self.assertEqual(files, ["a.txt"])
                self.assertEqual(state["base_revision"], git("rev-parse", "HEAD"))
                self.assertFalse(state["dirty"])

    def test_branch_yields_committed_content_only(self):
        root, git = self.repo_with_branch()
        with deploy_orin.open_source(
            self.args("--source", str(root), "--branch", "feature")
        ) as (export, files, state):
            self.assertEqual(files, ["run.sh", "src/a.cpp"])
            self.assertEqual((export / "src/a.cpp").read_text(), "committed\n")
            self.assertFalse((export / "untracked.txt").exists())
            self.assertTrue(os.access(export / "run.sh", os.X_OK))
            self.assertFalse(os.access(export / "src/a.cpp", os.X_OK))
            self.assertEqual(
                state, {"base_revision": git("rev-parse", "feature"), "dirty": False}
            )
            kept = export
        self.assertFalse(kept.exists())

    def test_unknown_branch_and_non_git_source_raise(self):
        root, _ = git_repo(self, {"a.txt": "1\n"})
        with self.assertRaisesRegex(DeployError, "not found"):
            with deploy_orin.open_source(
                self.args("--source", str(root), "--branch", "nope")
            ):
                pass
        plain = self.tmp / "plain"
        plain.mkdir()
        with self.assertRaisesRegex(DeployError, "not inside a git"):
            with deploy_orin.open_source(self.args("--source", str(plain))):
                pass

    def test_run_deploy_with_branch_uploads_the_branch_files(self):
        root, _ = self.repo_with_branch()
        orin = FakeOrin()
        orin.run(root, "--source", str(root), "--branch", "feature")
        self.assertEqual(orin.deltas, [["run.sh", "src/a.cpp"]])
        self.assertEqual(
            orin.pending["src/a.cpp"], hashlib.sha256(b"committed\n").hexdigest() + "-"
        )


class SyncTests(TempDirTestCase):
    def test_file_group(self):
        for name in ("CMakeLists.txt", "src/a.cpp", "cmake/x.cmake", "tests/t.py", "examples/egocap/x"):  # fmt: skip
            self.assertEqual(deploy_orin.file_group(name), "source", name)
        for name in ("deps/lib/x.c", "README.md", "examples/other/x", "docs/CMakeLists.txt"):  # fmt: skip
            self.assertEqual(deploy_orin.file_group(name), "bulk", name)

    def test_tracked_files_include_edits_not_untracked(self):
        root, _ = git_repo(self, {"a.txt": "1\n", "sub/b.txt": "2\n"})
        (root / "a.txt").write_text("edited\n")
        (root / "untracked.txt").write_text("x")
        self.assertEqual(deploy_orin.tracked_files(root), ["a.txt", "sub/b.txt"])

    def test_tracked_files_skip_deleted(self):
        root, _ = git_repo(self, {"a.txt": "1\n", "b.txt": "2\n"})
        (root / "b.txt").unlink()
        self.assertEqual(deploy_orin.tracked_files(root), ["a.txt"])

    def test_hash_files_records_exec_bit(self):
        root, _ = git_repo(self, {"a.txt": "1\n", "run.sh": "#!/bin/sh\n"})
        (root / "run.sh").chmod(0o755)
        entries = deploy_orin.hash_files(root, ["a.txt", "run.sh"])
        self.assertEqual(
            entries["a.txt"], deploy_orin.sha256_file(root / "a.txt") + "-"
        )
        self.assertEqual(
            entries["run.sh"], deploy_orin.sha256_file(root / "run.sh") + "x"
        )

    def test_group_digest_is_stable_and_tracks_changes(self):
        entries = {"src/a": "h1-", "src/b": "h2-", "README.md": "h3-"}
        reordered = {"README.md": "h3-", "src/b": "h2-", "src/a": "h1-"}
        digest = deploy_orin.group_digest
        self.assertEqual(digest(entries, "source"), digest(reordered, "source"))
        for changed in ({"src/a": "h9-"}, {"src/a": "h1x"}, {"src/c": "h4-"}):
            self.assertNotEqual(
                digest(entries, "source"), digest({**entries, **changed}, "source")
            )
        self.assertEqual(
            digest(entries, "bulk"), digest({**entries, "src/a": "h9-"}, "bulk")
        )

    def test_diff_identical_sends_nothing(self):
        entries = {"src/a": "1-", "README.md": "2-"}
        self.assertEqual(deploy_orin.diff_manifests(entries, dict(entries)), ([], []))

    def test_diff_changed_new_and_removed_per_group(self):
        local = {"src/a": "1-", "src/new": "n-", "src/same": "s-", "README.md": "r2-", "doc": "d-"}  # fmt: skip
        remote = {"src/a": "0-", "src/same": "s-", "src/gone": "g-", "README.md": "r1-", "doc": "d-", "old.txt": "o-"}  # fmt: skip
        send, removed = deploy_orin.diff_manifests(local, remote)
        self.assertEqual(send, ["README.md", "src/a", "src/new"])
        self.assertEqual(removed, ["old.txt", "src/gone"])

    def test_diff_skips_a_group_with_equal_digest_without_comparing_files(self):
        local = {"src/a": "1-", "README.md": "r2-"}
        remote = {"src/a": "0-", "README.md": "r1-"}

        def fake_digest(entries, group):
            return "same" if group == "source" else repr(sorted(entries.items()))

        with mock.patch.object(deploy_orin, "group_digest", fake_digest):
            send, removed = deploy_orin.diff_manifests(local, remote)
        self.assertEqual((send, removed), (["README.md"], []))

    def test_diff_against_empty_remote_sends_everything(self):
        local = {"src/a": "1-", "README.md": "2-"}
        self.assertEqual(
            deploy_orin.diff_manifests(local, {}), (["README.md", "src/a"], [])
        )

    def test_build_delta_has_only_sent_files_and_deploy_members(self):
        root, _ = git_repo(self, {"src/a": "a\n", "src/b": "b\n", "README.md": "r\n"})
        manifest = deploy_orin.hash_files(root, deploy_orin.tracked_files(root))
        dest = self.tmp / "delta.tar.gz"
        deploy_orin.build_delta(root, ["src/b"], ["old/x"], manifest, dest)
        with tarfile.open(dest) as tar:
            self.assertEqual(
                sorted(tar.getnames()),
                [".deploy/manifest.json", ".deploy/removed.json", "src/b"],
            )
            self.assertEqual(
                json.loads(tar.extractfile(".deploy/manifest.json").read()), manifest
            )
            self.assertEqual(
                json.loads(tar.extractfile(".deploy/removed.json").read()), ["old/x"]
            )

    def test_read_remote_manifest_tolerates_missing_or_garbage(self):
        def read(text):
            target = mock.Mock()
            target.capture.return_value = text
            return deploy_orin.read_remote_manifest(target, "/c")

        self.assertEqual(read('{"a": "h-"}'), {"a": "h-"})
        for text in ("", "{oops", "[1]"):
            self.assertEqual(read(text), {})


class ApplyTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.src = self.tmp / "orin" / "src"
        self.manifest_path = self.tmp / "orin" / "src.manifest.json"
        self.manifest_path.parent.mkdir()

    def delta(self, files, removed=(), manifest=None):
        root = self.tmp / "local"
        root.mkdir(exist_ok=True)
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        if manifest is None:
            manifest = deploy_orin.hash_files(root, sorted(files))
        dest = self.tmp / "delta.tar.gz"
        deploy_orin.build_delta(root, sorted(files), list(removed), manifest, dest)
        return dest

    def apply(self, archive, fresh=False):
        deploy_orin.remote_apply(
            argparse.Namespace(
                archive=str(archive),
                src=str(self.src),
                manifest=str(self.manifest_path),
                fresh=fresh,
            )
        )

    def test_extracts_and_writes_manifest(self):
        self.apply(self.delta({"src/a.cpp": "a\n", "README.md": "r\n"}))
        self.assertEqual((self.src / "src/a.cpp").read_text(), "a\n")
        self.assertFalse((self.src / ".deploy").exists())
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(sorted(manifest), ["README.md", "src/a.cpp"])

    def test_removes_files_and_prunes_empty_dirs(self):
        (self.src / "gone/deep").mkdir(parents=True)
        (self.src / "gone/deep/x").write_text("x")
        (self.src / "keep").mkdir()
        (self.src / "keep/a").write_text("a")
        (self.src / "keep/b").write_text("b")
        self.apply(self.delta({"new": "n\n"}, removed=["gone/deep/x", "keep/b"]))
        self.assertFalse((self.src / "gone").exists())
        self.assertTrue((self.src / "keep/a").exists())
        self.assertFalse((self.src / "keep/b").exists())
        self.assertTrue(self.src.is_dir())

    def test_hash_mismatch_raises_and_leaves_manifest_unwritten(self):
        bad = {"a": "0" * 64 + "-"}
        with self.assertRaisesRegex(DeployError, "does not match"):
            self.apply(self.delta({"a": "a\n"}, manifest=bad))
        self.assertFalse(self.manifest_path.exists())

    def test_fresh_wipes_stale_files(self):
        self.src.mkdir()
        (self.src / "stale").write_text("old")
        self.apply(self.delta({"a": "a\n"}), fresh=True)
        self.assertFalse((self.src / "stale").exists())
        self.assertTrue((self.src / "a").exists())

    def test_without_fresh_keeps_untouched_files(self):
        self.src.mkdir()
        (self.src / "stale").write_text("old")
        self.apply(self.delta({"a": "a\n"}))
        self.assertTrue((self.src / "stale").exists())

    def test_refuses_removal_outside_src(self):
        victim = self.tmp / "orin" / "victim"
        victim.write_text("keep")
        for name in ("../victim", "/etc/hostname"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(DeployError, "outside"),
            ):
                self.apply(self.delta({"a": "a\n"}, removed=[name]))
        self.assertTrue(victim.exists())

    def test_remote_parser_apply_arguments(self):
        parsed = deploy_orin.remote_parser().parse_args(
            ["apply", "--archive", "a", "--src", "s", "--manifest", "m", "--fresh"]
        )
        self.assertTrue(parsed.fresh)

    def test_remote_parser_activate_accepts_sensors(self):
        base = ["activate"]
        for name in ("archive", "prefix", "release", "unit", "owner"):
            base += [f"--{name}", "x"]
        self.assertEqual(deploy_orin.remote_parser().parse_args(base).sensors, "")
        parsed = deploy_orin.remote_parser().parse_args(base + ["--sensors", "0,1"])
        self.assertEqual(parsed.sensors, "0,1")

    def test_remote_parser_activate_no_livekit_flag(self):
        base = ["activate"]
        for name in ("archive", "prefix", "release", "unit", "owner"):
            base += [f"--{name}", "x"]
        self.assertFalse(deploy_orin.remote_parser().parse_args(base).no_livekit)
        parsed = deploy_orin.remote_parser().parse_args(base + ["--no-livekit"])
        self.assertTrue(parsed.no_livekit)


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

    def activate(
        self,
        archive,
        timeout=60,
        replace_unmanaged=False,
        sensors="",
        no_livekit=False,
    ):
        deploy_orin.remote_activate(
            argparse.Namespace(
                archive=str(archive),
                prefix=str(self.prefix),
                release="r1",
                unit=str(self.unit),
                owner="bob",
                ready_timeout=timeout,
                sensors=sensors,
                no_livekit=no_livekit,
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

    def test_no_livekit_activation_is_ready_on_frames_without_a_join(self):
        self.journal = "sensor 0: 150 frames | encoded 150, dropped 0\n"
        self.activate(self.make_archive(), sensors="0", no_livekit=True)

    def test_livekit_activation_still_needs_the_join(self):
        self.journal = "sensor 0: 150 frames | livekit 5 published\n"
        with self.assertRaises(DeployError):
            self.activate(self.make_archive(), sensors="0", timeout=1)

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

    def test_put_passes_quiet_flag_to_scp(self):
        target = make_target(self)
        with (
            mock.patch.object(deploy_orin.sys, "stdout", FakeStdout(False)),
            mock.patch.object(
                deploy_orin.subprocess, "run", return_value=completed(0)
            ) as run,
        ):
            target.put(Path("/tmp/a"), "/r/a")
        self.assertEqual(len(run.call_args_list), 1)
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
            ["u@orin", "--config", "c", "--sdk-root", "/s"]
        )
        self.assertEqual(args.sdk_root, "/s")
        self.assertFalse(args.replace_unmanaged)
        self.assertEqual(
            deploy_orin.parser().parse_args(["h", "--config", "c"]).sdk_root,
            "",
        )
        remote = deploy_orin.remote_parser()
        for step, extra in (
            ("preflight", []),
            ("build", ["--src", "s", "--stage", "t"]),
        ):
            parsed = remote.parse_args([step, "--sdk-root", "/s", *extra])
            self.assertEqual(parsed.sdk_root, "/s")

    def test_backend_option_is_rejected_everywhere(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                deploy_orin.parser().parse_args(
                    ["u@orin", "--backend", "sipl", "--config", "c"]
                )
            with self.assertRaises(SystemExit):
                deploy_orin.remote_parser().parse_args(
                    ["preflight", "--backend", "sipl"]
                )

    def test_removed_local_options_are_rejected(self):
        for flag in ("--prefix", "--service-groups", "--ready-timeout"):
            with self.subTest(flag=flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        deploy_orin.parser().parse_args(
                            ["u@orin", "--config", "c", flag, "1"]
                        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                deploy_orin.parser().parse_args(
                    ["u@orin", "--config", "c", "--fetch-archive"]
                )

    def test_cmake_arg_is_repeatable_and_defaults_empty(self):
        parser = deploy_orin.parser()
        self.assertEqual(parser.parse_args(["h", "--config", "c"]).cmake_arg, [])
        args = parser.parse_args(
            ["h", "--config", "c", "--cmake-arg=-DA=1", "--cmake-arg=-DB=2"]
        )
        self.assertEqual(args.cmake_arg, ["-DA=1", "-DB=2"])
        remote = deploy_orin.remote_parser().parse_args(
            ["build", "--src", "s", "--stage", "t", "--cmake-arg=-DA=1"]
        )
        self.assertEqual(remote.cmake_arg, ["-DA=1"])

    def test_preflight_names_the_sdk_root_path_when_headers_are_missing(self):
        args = argparse.Namespace(sdk_root="/nonexistent-sdk")
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


class BuildPathTests(unittest.TestCase):
    def test_build_path_prefixes_uv_then_cuda_before_the_inherited_path(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            self.assertEqual(
                deploy_orin.build_path(),
                f"{Path.home()}/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin",
            )

    def preflight(self, missing=()):
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append((cmd, kwargs))
            return completed(1 if cmd[0] == "which" and cmd[1] in missing else 0)

        args = argparse.Namespace(sdk_root="/nonexistent-sdk")
        with (
            mock.patch.object(deploy_orin.os, "uname") as uname,
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value="# R36\n"),
            mock.patch.object(Path, "is_dir", return_value=True),
            mock.patch.object(deploy_orin.os, "statvfs") as statvfs,
            mock.patch.object(deploy_orin.subprocess, "run", fake_run),
            mock.patch.dict(os.environ, {"PATH": "/usr/bin"}),
        ):
            uname.return_value.machine = "aarch64"
            statvfs.return_value.f_bavail = 1 << 30
            statvfs.return_value.f_frsize = 4096
            try:
                deploy_orin.remote_preflight(args)
            finally:
                self.seen = seen

    def test_preflight_requires_uv_using_the_build_path(self):
        with self.assertRaisesRegex(DeployError, "uv is not installed.*astral.sh"):
            self.preflight(missing=("uv",))
        uv_call = next(k for c, k in self.seen if c == ["which", "uv"])
        self.assertEqual(
            uv_call["env"]["PATH"],
            f"{Path.home()}/.local/bin:/usr/local/cuda/bin:/usr/bin",
        )

    def test_preflight_succeeds_when_uv_present(self):
        self.preflight()
        self.assertIn(["which", "uv"], [c for c, _ in self.seen])


class RemoteBuildTests(TempDirTestCase):
    LIBCUDA = "/opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1"
    EXPECTED_PATH = f"{Path.home()}/.local/bin:/usr/local/cuda/bin:/usr/bin"

    def build(
        self,
        sdk_root="",
        libcuda=False,
        path="/usr/bin",
        src="/src",
        cmake_arg=(),
        sipl=True,
    ):
        stage = self.tmp / "stage"
        args = argparse.Namespace(
            src=src,
            cmake_arg=list(cmake_arg),
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
            if cmd[:2] == ["rm", "-rf"] and cmd[2].endswith("/build"):
                shutil.rmtree(cmd[2], ignore_errors=True)
            return completed(0)

        with (
            mock.patch.object(deploy_orin, "CACHE", str(self.tmp / "cache")),
            mock.patch.object(deploy_orin, "run", fake_run),
            mock.patch.object(deploy_orin, "tail", return_value="x"),
            mock.patch.object(deploy_orin, "stage_producer") as self.stage_producer,
            mock.patch.object(Path, "exists", fake_exists),
            mock.patch.object(Path, "is_dir", return_value=sipl),
            mock.patch.object(Path, "read_text", fake_read_text),
            mock.patch.dict(os.environ, {"PATH": path}),
        ):
            deploy_orin.remote_build(args)
        return calls

    def configure(self, calls):
        return next(c for c in calls if c[0][:2] == ["cmake", "-S"])

    def test_configure_flags_and_cuda_path(self):
        cmd, kwargs = self.configure(self.build(sdk_root="/mnt/sdk"))
        for flag in (
            "-DJETSON_MMAPI_DIR=/mnt/sdk/usr/src/jetson_multimedia_api",
            "-DSIPL_API_DIR=/mnt/sdk/usr/src/jetson_sipl_api",
            "-DENABLE_CLANG_FORMAT_CHECK=OFF",
            "-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include",
        ):
            self.assertIn(flag, cmd)
        self.assertEqual(kwargs["env"]["PATH"], self.EXPECTED_PATH)

    def cleaned(self, calls):
        return [
            c for c, _ in calls if c[:2] == ["rm", "-rf"] and c[2].endswith("/build")
        ]

    def test_changed_source_path_cleans_build_dir_but_same_path_keeps_it(self):
        self.assertTrue(self.cleaned(self.build(src="/cache/src")))
        self.assertFalse(self.cleaned(self.build(src="/cache/src")))
        self.assertTrue(self.cleaned(self.build(src="/cache/src-abc123")))
        self.assertFalse(self.cleaned(self.build(src="/cache/src-abc123")))

    def test_release_is_staged_from_the_build_tree_not_cmake_install(self):
        calls = self.build()
        self.assertEqual(
            self.stage_producer.call_args.args,
            (self.tmp / "cache/build", self.tmp / "stage"),
        )
        self.assertFalse([c for c, _ in calls if "--install" in c])
        self.assertFalse([a for c, _ in calls for a in c if "component" in a])

    def test_build_step_also_gets_the_cuda_path(self):
        calls = self.build()
        cmd, kwargs = next(c for c in calls if c[0][:2] == ["cmake", "--build"])
        self.assertEqual(kwargs["env"]["PATH"], self.EXPECTED_PATH)
        self.assertIn("camera_plugin_sensing", cmd)

    def test_sipl_dir_omitted_when_absent(self):
        cmd, _ = self.configure(self.build(sipl=False))
        self.assertIn("-DJETSON_MMAPI_DIR=/usr/src/jetson_multimedia_api", cmd)
        self.assertFalse([a for a in cmd if a.startswith("-DSIPL_API_DIR")])

    def test_cmake_args_are_appended_last_in_order(self):
        extra = ["-DENABLE_CLANG_FORMAT_CHECK=ON", "-DX=1"]
        cmd, _ = self.configure(self.build(cmake_arg=extra, libcuda=True))
        self.assertEqual(cmd[-2:], extra)
        self.assertGreater(
            cmd.index(extra[0]),
            max(i for i, a in enumerate(cmd) if a.startswith("-DCMAKE_EXE_LINKER")),
        )

    def test_changed_cmake_args_clean_build_dir(self):
        self.assertTrue(self.cleaned(self.build(cmake_arg=["-DX=1"])))
        self.assertFalse(self.cleaned(self.build(cmake_arg=["-DX=1"])))
        self.assertTrue(self.cleaned(self.build(cmake_arg=["-DX=2"])))

    def test_linker_flag_only_when_gpu_libcuda_exists(self):
        absent, _ = self.configure(self.build(libcuda=False))
        self.assertFalse([a for a in absent if "LINKER_FLAGS" in a])
        present, _ = self.configure(self.build(libcuda=True))
        self.assertIn(
            f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed {self.LIBCUDA}", present
        )


LIB_LINE = "\tliblivekit_ffi.so => {} (0x0000ffff)"


class CacheValueTests(TempDirTestCase):
    def test_reads_typed_entries_and_ignores_prefix_lookalikes(self):
        (self.tmp / "CMakeCache.txt").write_text(
            "// comment\nFOO_BAR:STRING=wrong\nFOO:PATH=/a=b\nEMPTY:STRING=\n"
        )
        self.assertEqual(deploy_orin.cache_value(self.tmp, "FOO"), "/a=b")
        self.assertEqual(deploy_orin.cache_value(self.tmp, "EMPTY"), "")
        self.assertEqual(deploy_orin.cache_value(self.tmp, "ABSENT"), "")


class StageProducerTests(TempDirTestCase):
    """stage_producer on a fabricated build tree, with both ldd runs faked."""

    def setUp(self):
        super().setUp()
        self.build = self.tmp / "build"
        self.stage = self.tmp / "stage"
        self.plugin = self.build / "src/plugins/sensing/app"
        (self.plugin / "configs").mkdir(parents=True)
        (self.plugin / "configs/platform.json").write_text("{}")
        (self.plugin / "camera_plugin_sensing").write_text("bin")
        self.libdir = self.build / "_deps/sdk/lib"
        self.libdir.mkdir(parents=True)
        (self.libdir / "liblivekit_ffi.so").write_text("lib")
        self.cache = ""
        self.build_ldd = LIB_LINE.format(self.libdir / "liblivekit_ffi.so")
        self.release_ldd = "\tlibc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x1)"
        self.licenses = self.tmp / "lic"
        self.licenses.mkdir()
        (self.licenses / "LICENSE").write_text("license")
        (self.licenses / "NOTICE").write_text("notice")

    def stage_it(self, cache=None):
        """Run stage_producer; cache defaults to recording both licence files."""
        if cache is None:
            cache = (
                f"SENSING_LIVEKIT_LICENSE_FILE:FILEPATH={self.licenses}/LICENSE\n"
                f"SENSING_LIVEKIT_NOTICE_FILE:FILEPATH={self.licenses}/NOTICE\n"
            )
        (self.build / "CMakeCache.txt").write_text(cache)
        runs = []

        def fake_run(cmd, **kwargs):
            runs.append((list(cmd), kwargs))
            return completed(0, stdout=self.release_ldd)

        with (
            mock.patch.object(deploy_orin, "tail", return_value=self.build_ldd),
            mock.patch.object(deploy_orin.subprocess, "run", fake_run),
        ):
            deploy_orin.stage_producer(self.build, self.stage)
        return runs

    def test_happy_path_stages_binary_configs_libs_and_licences(self):
        runs = self.stage_it()
        out = self.stage / "plugins/sensing_camera"
        self.assertEqual((self.stage / deploy_orin.BIN_REL).read_text(), "bin")
        self.assertTrue((out / "configs/platform.json").is_file())
        self.assertEqual((out / "liblivekit_ffi.so").read_text(), "lib")
        self.assertEqual((out / "licenses/livekit-sdk/LICENSE").read_text(), "license")
        self.assertEqual((out / "licenses/livekit-sdk/NOTICE").read_text(), "notice")
        cmd, kwargs = runs[-1]
        self.assertEqual(cmd, ["ldd", str(self.stage / deploy_orin.BIN_REL)])
        self.assertEqual(kwargs["env"]["LD_LIBRARY_PATH"], str(out))

    def test_ldd_not_found_for_a_livekit_library_is_not_copied(self):
        self.build_ldd = "\tliblivekit_ffi.so => not found"
        self.stage_it()
        out = self.stage / "plugins/sensing_camera"
        self.assertEqual(list(out.glob("liblivekit*")), [])
        self.assertFalse((out / "licenses").exists())

    def test_no_binary_is_an_error(self):
        (self.plugin / "camera_plugin_sensing").unlink()
        with self.assertRaisesRegex(DeployError, "found 0"):
            self.stage_it()

    def test_a_directory_with_the_binary_name_is_not_a_binary(self):
        (self.plugin / "camera_plugin_sensing").unlink()
        (self.build / "camera_plugin_sensing").mkdir()
        with self.assertRaisesRegex(DeployError, "found 0"):
            self.stage_it()

    def test_two_binaries_is_an_error(self):
        other = self.build / "other"
        other.mkdir()
        (other / "camera_plugin_sensing").write_text("bin2")
        with self.assertRaisesRegex(DeployError, "found 2"):
            self.stage_it()

    def test_missing_configs_dir_is_an_error(self):
        shutil.rmtree(self.plugin / "configs")
        with self.assertRaisesRegex(DeployError, "platform configs"):
            self.stage_it()

    def test_no_livekit_library_resolved_stages_without_libs_or_licences(self):
        self.build_ldd = "\tlibc.so.6 => /lib/libc.so.6 (0x1)"
        # No licence cache entries either: nothing should look for them.
        self.stage_it(cache="")
        out = self.stage / "plugins/sensing_camera"
        self.assertEqual((self.stage / deploy_orin.BIN_REL).read_text(), "bin")
        self.assertTrue((out / "configs/platform.json").is_file())
        self.assertEqual(list(out.glob("liblivekit*")), [])
        self.assertFalse((out / "licenses").exists())

    def test_livekit_library_resolved_stages_libs_and_licences(self):
        self.stage_it()
        out = self.stage / "plugins/sensing_camera"
        self.assertTrue((out / "liblivekit_ffi.so").is_file())
        self.assertTrue((out / "licenses/livekit-sdk/LICENSE").is_file())

    def test_missing_licence_is_an_error(self):
        for name in ("LICENSE", "NOTICE"):
            (self.licenses / name).unlink()
            with self.subTest(missing=name), self.assertRaisesRegex(DeployError, name):
                self.stage_it()
            (self.licenses / name).write_text("x")
            shutil.rmtree(self.stage)

    def test_library_still_resolved_from_the_build_tree_is_an_error(self):
        self.release_ldd = LIB_LINE.format(self.libdir / "liblivekit_ffi.so")
        with self.assertRaisesRegex(DeployError, "from the build tree"):
            self.stage_it()

    def test_unresolved_library_is_an_error(self):
        self.release_ldd = "\tlibfoo.so.1 => not found"
        with self.assertRaisesRegex(DeployError, "unresolved shared libraries"):
            self.stage_it()

    def test_licence_found_via_the_cache_entry(self):
        self.stage_it()
        self.assertEqual(
            deploy_orin.livekit_notice(self.build, "LICENSE"), self.licenses / "LICENSE"
        )

    def test_licence_found_in_the_fetchcontent_directory(self):
        for name in ("LICENSE", "NOTICE"):
            fetched = self.build / "_deps" / f"livekit_sdk_{name.lower()}-src"
            fetched.mkdir(parents=True)
            (fetched / name).write_text(f"fetched {name}")
        self.stage_it(cache="")
        notices = self.stage / "plugins/sensing_camera/licenses/livekit-sdk"
        self.assertEqual((notices / "LICENSE").read_text(), "fetched LICENSE")
        self.assertEqual((notices / "NOTICE").read_text(), "fetched NOTICE")

    def test_licence_found_in_or_beside_a_local_sdk_directory(self):
        sdk = self.tmp / "sdk-root/install"
        sdk.mkdir(parents=True)
        cache = f"LIVEKIT_SDK_DIR:PATH={sdk}\n"
        (sdk / "LICENSE").write_text("in sdk")
        (sdk.parent / "NOTICE").write_text("beside sdk")
        self.stage_it(cache=cache)
        notices = self.stage / "plugins/sensing_camera/licenses/livekit-sdk"
        self.assertEqual((notices / "LICENSE").read_text(), "in sdk")
        self.assertEqual((notices / "NOTICE").read_text(), "beside sdk")

    def test_cache_entry_wins_over_the_other_locations(self):
        fetched = self.build / "_deps/livekit_sdk_license-src"
        fetched.mkdir(parents=True)
        (fetched / "LICENSE").write_text("fetched")
        self.stage_it()
        self.assertEqual(
            (
                self.stage / "plugins/sensing_camera/licenses/livekit-sdk/LICENSE"
            ).read_text(),
            "license",
        )

    def test_unit_sets_ld_library_path_to_the_release_directory(self):
        config = {"livekit_url": "wss://x", "token_file": "/t", "args": [STREAM]}
        text = deploy_orin.render_unit(config, "/opt/ic", "bob", "video")
        self.assertIn(
            "Environment=LD_LIBRARY_PATH=/opt/ic/current/plugins/sensing_camera",
            text.splitlines(),
        )


class ConfiguredSensorsTests(unittest.TestCase):
    def sensors(self, *args):
        return deploy_orin.configured_sensors({"args": list(args)})

    def test_single_and_multiple_streams_in_order(self):
        self.assertEqual(self.sensors(STREAM), [0])
        self.assertEqual(
            self.sensors(
                "--add-stream=sensor=0,livekit=left",
                "--add-stream=sensor=1,livekit=right",
            ),
            [0, 1],
        )

    def test_sensor_not_first_field_and_multi_digit(self):
        self.assertEqual(self.sensors("--add-stream=livekit=a,sensor=12"), [12])

    def test_ignores_other_arguments(self):
        self.assertEqual(self.sensors("--verbose", "--other=sensor=3", STREAM), [0])

    def test_no_match(self):
        self.assertEqual(self.sensors("--add-stream=livekit=x"), [])
        self.assertEqual(deploy_orin.configured_sensors({}), [])


class ReadinessTests(TempDirTestCase):
    JOINED = "Joined LiveKit room r\n"

    def run_wait(self, journal, sensors, active=True, timeout=10, livekit=True):
        def fake_run(cmd, **kwargs):
            return completed(0 if active else 3)

        self.journal_calls = 0

        def fake_tail(cmd):
            if cmd[0] != "journalctl":
                return "ActiveState=active"
            self.journal_calls += 1
            return journal

        ticks = iter(range(0, 1000, 5))
        stderr = io.StringIO()
        with (
            mock.patch.object(deploy_orin.subprocess, "run", fake_run),
            mock.patch.object(deploy_orin, "tail", fake_tail),
            mock.patch.object(deploy_orin.time, "sleep"),
            mock.patch.object(deploy_orin.time, "monotonic", lambda: next(ticks)),
            mock.patch.object(deploy_orin, "CACHE", str(self.tmp)),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                deploy_orin.wait_ready(
                    "2026-01-01 00:00:00", timeout, sensors, livekit=livekit
                )
            finally:
                self.stderr = stderr.getvalue()

    def test_one_of_two_sensors_publishing_names_the_other(self):
        journal = (
            self.JOINED
            + "sensor 0: 30 fps | livekit 12 published\n"
            + "sensor 1: 0 fps | livekit 0 published\n"
        )
        with self.assertRaises(DeployError) as ctx:
            self.run_wait(journal, [0, 1])
        message = str(ctx.exception)
        self.assertIn("sensors not publishing: 1", message)
        self.assertNotIn("publishing: 0", message)

    def test_both_sensors_publishing_returns(self):
        journal = (
            self.JOINED
            + "sensor 0: 30 fps | livekit 12 published\n"
            + "sensor 1: 30 fps | livekit 3 published\n"
        )
        self.run_wait(journal, [0, 1])

    def test_sensor_number_does_not_match_a_longer_number(self):
        journal = self.JOINED + "sensor 10: 30 fps | livekit 5 published\n"
        with self.assertRaisesRegex(DeployError, "sensors not publishing: 1"):
            self.run_wait(journal, [1])

    def test_without_livekit_every_sensor_with_frames_is_ready(self):
        journal = (
            "sensor 0: 150 frames | encoded 150, dropped 0\n"
            "sensor 1: 10 frames | encoded 10, dropped 0\n"
        )
        self.run_wait(journal, [0, 1], livekit=False)

    def test_without_livekit_a_sensor_with_zero_frames_is_named(self):
        journal = "sensor 0: 150 frames | encoded 150\nsensor 1: 0 frames | encoded 0\n"
        with self.assertRaises(DeployError) as ctx:
            self.run_wait(journal, [0, 1], livekit=False)
        self.assertIn("sensors not publishing: 1", str(ctx.exception))
        self.assertNotIn("publishing: 0", str(ctx.exception))

    def test_without_livekit_sensor_number_is_not_confused_with_a_longer_one(self):
        journal = "sensor 0: 5 frames\nsensor 10: 10 frames\n"
        with self.assertRaisesRegex(DeployError, "sensors not publishing: 1;"):
            self.run_wait(journal, [0, 1], livekit=False)
        self.run_wait("sensor 1: 10 frames\n", [1], livekit=False)

    def test_without_livekit_a_zero_frames_line_with_a_later_count_digit(self):
        with self.assertRaises(DeployError):
            self.run_wait("sensor 0: 0 frames\n", [0], livekit=False)

    def test_with_livekit_frames_alone_are_not_readiness(self):
        with self.assertRaises(DeployError):
            self.run_wait(self.JOINED + "sensor 0: 150 frames\n", [0])

    def test_requires_joining_the_room(self):
        with self.assertRaises(DeployError):
            self.run_wait("sensor 0: x | livekit 5 published\n", [0])

    def test_no_sensors_any_publishing_counts(self):
        self.run_wait(self.JOINED + "stats livekit 7 published\n", [])

    def test_no_sensors_nothing_publishing_fails_without_a_sensor_list(self):
        with self.assertRaises(DeployError) as ctx:
            self.run_wait(self.JOINED + "livekit 0 published\n", [])
        self.assertNotIn("sensors not publishing", str(ctx.exception))

    def test_inactive_service_stops_waiting_and_reports(self):
        ticks_used = []
        with mock.patch.object(deploy_orin.time, "sleep", ticks_used.append):
            with self.assertRaises(DeployError):
                self.run_wait(self.JOINED + "livekit 5 published\n", [], active=False)
        self.assertEqual(ticks_used, [])

    def test_inactive_service_lists_every_sensor_missing(self):
        with self.assertRaisesRegex(DeployError, "sensors not publishing: 0, 1"):
            self.run_wait("sensor 0: x | livekit 5 published\n", [0, 1], active=False)

    def test_inactive_service_never_fetches_for_the_loop(self):
        with self.assertRaises(DeployError):
            self.run_wait(self.JOINED, [0], active=False)
        self.assertEqual(self.journal_calls, 1)  # the report's fetch only

    def test_timeout_adds_no_journal_fetch_beyond_loop_and_report(self):
        # Clock ticks 0 (deadline = 10), 5 (one loop fetch), 10 (expired);
        # the failure report then makes the only other fetch.
        with self.assertRaises(DeployError):
            self.run_wait(
                self.JOINED + "sensor 0: x | livekit 0 published\n", [0], timeout=10
            )
        self.assertEqual(self.journal_calls, 2)

    def test_failure_prints_the_report_to_stderr(self):
        with self.assertRaises(DeployError):
            self.run_wait("boom\n", [0], active=False)
        self.assertIn("service: ActiveState=active", self.stderr)
        self.assertIn("boom", self.stderr)


class FailureReportTests(TempDirTestCase):
    def report(self, log, sensors=(0, 1)):
        def fake_tail(cmd):
            if cmd[0] == "systemctl":
                return "ActiveState=failed\nSubState=failed\nNRestarts=2"
            return log

        with (
            mock.patch.object(deploy_orin, "tail", fake_tail),
            mock.patch.object(deploy_orin, "CACHE", str(self.tmp / "cache")),
        ):
            return deploy_orin.failure_report("since", list(sensors))

    def test_noise_is_filtered_out_and_counted(self):
        log = "\n".join(
            [
                "Severity 5 : driver info",
                "Severity 6 : driver debug",
                "real line one",
                "real line two",
            ]
        )
        out = self.report(log)
        self.assertNotIn("driver info", out)
        self.assertNotIn("driver debug", out)
        self.assertIn("4 lines since the start, 2 of them driver INFO/DEBUG", out)
        self.assertIn("  real line one", out)

    def test_service_state_and_sensors_listed(self):
        out = self.report("x")
        self.assertIn("service: ActiveState=failed, SubState=failed, NRestarts=2", out)
        self.assertIn("sensors expected to publish: 0, 1", out)
        self.assertNotIn("sensors expected", self.report("x", sensors=()))

    def test_fault_lines_listed_and_capped_at_eight(self):
        log = "\n".join(f"camera_{i} startup_failed" for i in range(12))
        out = self.report(log)
        self.assertIn("producer fault lines:", out)
        faults = out.split("producer fault lines:")[1].split("first lines")[0]
        self.assertEqual(len([ln for ln in faults.splitlines() if ln.strip()]), 8)

    def test_no_fault_section_without_fault_lines(self):
        self.assertNotIn("producer fault lines", self.report("plain"))

    @staticmethod
    def numbered(n):
        return "\n".join(f"line {i:02d}" for i in range(n))

    @staticmethod
    def listed(out):
        return [ln.strip() for ln in out.splitlines() if ln.startswith("  line ")]

    def test_first_and_last_lines_with_elision_count(self):
        out = self.report(self.numbered(40))
        self.assertIn("last lines (10 not shown between):", out)
        head, tail_part = out.split("last lines")
        self.assertIn("line 00", head)
        self.assertIn("line 14", head)
        self.assertNotIn("line 15", head)
        self.assertNotIn("line 24", tail_part)
        self.assertIn("line 25", tail_part)
        self.assertIn("line 39", tail_part)

    def test_16_to_29_quiet_lines_have_no_duplicates(self):
        for n in (16, 22, 29, 30):
            with self.subTest(n=n):
                out = self.report(self.numbered(n))
                shown = self.listed(out)
                self.assertEqual(shown, [f"line {i:02d}" for i in range(n)])
                self.assertIn("remaining lines:", out)
                self.assertNotIn("not shown", out)
                self.assertNotIn("last lines", out)

    def test_31_lines_hides_one(self):
        out = self.report(self.numbered(31))
        self.assertIn("last lines (1 not shown between):", out)
        self.assertEqual(len(self.listed(out)), 30)

    def test_15_or_fewer_lines_have_no_last_section(self):
        for n in (1, 14, 15):
            with self.subTest(n=n):
                out = self.report(self.numbered(n))
                self.assertNotIn("last lines", out)
                self.assertNotIn("remaining lines", out)
                self.assertNotIn("not shown", out)
                self.assertEqual(len(self.listed(out)), n)

    def test_complete_journal_saved_under_cache(self):
        log = "Severity 5 : noise\nreal\n"
        out = self.report(log)
        saved = self.tmp / "cache" / "last-start-failure.log"
        self.assertEqual(saved.read_text(), log + "\n")
        self.assertIn(str(saved), out)

    def test_empty_log(self):
        out = self.report("")
        self.assertIn("log: 0 lines since the start, 0 of them", out)
        self.assertTrue((self.tmp / "cache" / "last-start-failure.log").exists())
        self.assertNotIn("last lines", out)


if __name__ == "__main__":
    unittest.main()
