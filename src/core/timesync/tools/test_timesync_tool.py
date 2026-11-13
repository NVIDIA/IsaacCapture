# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the timesync chrony setup tool (no real chrony, ssh or network)."""

from __future__ import annotations

import contextlib
import io
import stat
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import timesync_tool as tool

REAL_LINE = (
    "5BBD5B9D,91.189.91.157,3,1794516440.890180266,-0.000037218,-0.000042871,"
    "0.001530292,-12.973,-0.003,0.278,0.055693440,0.001789562,258.4,Normal"
)
LOCAL_LINE = (
    "7F7F0101,,10,1794518542.68,0.000000000,0.000000000,0.000000000,0.000,0.000,"
    "0.000,0.000000000,0.000000000,0.0,Normal"
)
HOST_IP = "192.168.1.221"
HOST_ID = "C0A801DD"


def remote_line(ref_id: str = HOST_ID, dispersion: str = "0.000300000") -> str:
    return (
        f"{ref_id},,4,1794518542.68,-0.000001000,0.000000000,0.000000000,0.000,"
        f"0.000,0.050,0.000400000,{dispersion},4.0,Normal"
    )


def make_tracking(**overrides: object) -> tool.Tracking:
    fields: dict[str, object] = dict(
        reference_id=HOST_ID,
        reference_name="",
        stratum=4,
        ref_time=1.0,
        system_time=0.0,
        root_delay=0.0004,
        root_dispersion=0.0003,
        skew_ppm=0.05,
        update_interval=4.0,
        leap_status="Normal",
    )
    fields.update(overrides)
    return tool.Tracking(**fields)  # type: ignore[arg-type]


class ParseTrackingTest(unittest.TestCase):
    def test_real_chrony_line(self) -> None:
        t = tool.parse_tracking(REAL_LINE)
        self.assertEqual(t.reference_id, "5BBD5B9D")
        self.assertEqual(t.reference_name, "91.189.91.157")
        self.assertEqual(t.stratum, 3)
        self.assertAlmostEqual(t.ref_time, 1794516440.890180266)
        self.assertAlmostEqual(t.system_time, -0.000037218)
        self.assertAlmostEqual(t.root_delay, 0.05569344)
        self.assertAlmostEqual(t.root_dispersion, 0.001789562)
        self.assertAlmostEqual(t.skew_ppm, 0.278)
        self.assertAlmostEqual(t.update_interval, 258.4)
        self.assertEqual(t.leap_status, "Normal")
        self.assertTrue(t.synchronised)

    def test_local_reference_line(self) -> None:
        t = tool.parse_tracking(LOCAL_LINE + "\n")
        self.assertEqual(t.reference_id, "7F7F0101")
        self.assertEqual(t.reference_name, "")
        self.assertEqual(t.stratum, 10)
        self.assertTrue(t.synchronised)

    def test_not_synchronised(self) -> None:
        self.assertFalse(make_tracking(leap_status="Not synchronised").synchronised)

    def test_stratum_zero_not_synchronised(self) -> None:
        self.assertFalse(make_tracking(stratum=0).synchronised)

    def test_wrong_field_counts(self) -> None:
        fields = REAL_LINE.split(",")
        for bad in (",".join(fields[:13]), ",".join(fields + ["x"])):
            with self.assertRaises(tool.ToolError):
                tool.parse_tracking(bad)

    def test_non_numeric_fields(self) -> None:
        for index in (2, 3, 4, 9, 10, 11, 12):
            fields = REAL_LINE.split(",")
            fields[index] = "abc"
            with self.subTest(index=index), self.assertRaises(tool.ToolError):
                tool.parse_tracking(",".join(fields))


class ReferenceIdTest(unittest.TestCase):
    def test_known_addresses(self) -> None:
        self.assertEqual(tool.reference_id_for(HOST_IP), HOST_ID)
        self.assertEqual(tool.reference_id_for("91.189.91.157"), "5BBD5B9D")

    def test_bad_addresses(self) -> None:
        for bad in ("999.1.1.1", "a.b.c.d", "1.2.3"):
            with self.subTest(bad=bad), self.assertRaises(tool.ToolError):
                tool.reference_id_for(bad)


class LanAddressTest(unittest.TestCase):
    def test_loopback(self) -> None:
        self.assertEqual(tool.lan_address_toward("127.0.0.1"), "127.0.0.1")


class RemoteDropInTest(unittest.TestCase):
    def test_serves_the_pc_and_lets_it_read_state(self) -> None:
        self.assertEqual(
            tool.remote_drop_in(HOST_IP).splitlines(),
            [
                f"allow {HOST_IP}",
                "bindcmdaddress 0.0.0.0",
                f"cmdallow {HOST_IP}",
            ],
        )


class RemoteDropInOfflineTest(unittest.TestCase):
    def test_default_has_no_local_reference(self) -> None:
        self.assertEqual(
            tool.remote_drop_in(HOST_IP),
            f"allow {HOST_IP}\nbindcmdaddress 0.0.0.0\ncmdallow {HOST_IP}\n",
        )
        self.assertNotIn("local", tool.remote_drop_in(HOST_IP, False))
        self.assertNotIn("local stratum", tool.remote_script(HOST_IP))

    def test_offline_adds_local_stratum(self) -> None:
        drop_in = tool.remote_drop_in(HOST_IP, True)
        self.assertIn("local stratum 10", drop_in.splitlines())
        self.assertIn(f"allow {HOST_IP}", drop_in)
        self.assertIn("local stratum 10", tool.remote_script(HOST_IP, True))


class RemoteScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.script = tool.remote_script(HOST_IP)

    def test_install_guard_and_include_check(self) -> None:
        self.assertIn("command -v chronyd", self.script)
        self.assertIn("apt-get install -y chrony", self.script)
        self.assertIn("conf\\.d", self.script)
        self.assertIn("/etc/chrony/chrony.conf", self.script)

    def test_writes_only_the_drop_in(self) -> None:
        self.assertEqual(self.script.count("tee "), 1)
        self.assertIn(f"tee {tool.DROP_IN}", self.script)
        self.assertNotIn(">/etc", self.script.replace(">/dev/null", ""))

    def test_idempotent_compare_before_write(self) -> None:
        compare = self.script.index('"$current" != "$wanted"')
        self.assertLess(compare, self.script.index("tee "))
        self.assertIn("is already as wanted", self.script)

    def test_unchanged_drop_in_starts_a_stopped_chrony(self) -> None:
        start = self.script.index("elif ! sudo -n systemctl is-active --quiet chrony")
        branch = self.script[start : self.script.index("else", start)]
        self.assertIn("systemctl start chrony", branch)
        self.assertNotIn("restart", branch)

    def test_serves_the_pc(self) -> None:
        self.assertIn(f"allow {HOST_IP}", self.script)
        self.assertIn(f"cmdallow {HOST_IP}", self.script)
        self.assertIn("bindcmdaddress 0.0.0.0", self.script)
        self.assertNotIn("server ", self.script)

    def test_wanted_text_is_quoted(self) -> None:
        script = tool.remote_script("1.2.3.4'; rm -rf x; '")
        self.assertNotIn("\nrm -rf x", script)

    def test_bash_accepts_script(self) -> None:
        result = subprocess.run(
            ["bash", "-n"],
            input=self.script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class JudgeTest(unittest.TestCase):
    def judge(self, local: tool.Tracking) -> list[str]:
        return tool.judge(local, HOST_ID, 0.002)

    def test_usable(self) -> None:
        self.assertEqual(self.judge(make_tracking()), [])

    def test_unsynchronised(self) -> None:
        problems = self.judge(make_tracking(leap_status="Not synchronised"))
        self.assertEqual(len(problems), 1)
        self.assertIn("not synchronised", problems[0])

    def test_wrong_reference(self) -> None:
        problems = self.judge(make_tracking(reference_id="5BBD5B9D"))
        self.assertEqual(len(problems), 1)
        self.assertIn("5BBD5B9D", problems[0])

    def test_high_dispersion(self) -> None:
        problems = self.judge(make_tracking(root_dispersion=0.01))
        self.assertEqual(len(problems), 1)
        self.assertIn("dispersion", problems[0])

    def test_large_system_time_is_not_ready(self) -> None:
        problems = self.judge(make_tracking(root_dispersion=0.0003, system_time=0.050))
        self.assertEqual(len(problems), 1)
        self.assertIn("System time", problems[0])
        # The sign does not matter.
        problems = self.judge(make_tracking(root_dispersion=0.0003, system_time=-0.050))
        self.assertEqual(len(problems), 1)
        self.assertIn("System time", problems[0])

    def test_small_system_time_stays_ready(self) -> None:
        self.assertEqual(self.judge(make_tracking(system_time=0.0005)), [])
        self.assertEqual(self.judge(make_tracking(system_time=-0.0005)), [])

    def test_problems_accumulate(self) -> None:
        bad = make_tracking(
            leap_status="Not synchronised",
            reference_id="5BBD5B9D",
            root_dispersion=0.01,
        )
        self.assertEqual(len(self.judge(bad)), 3)


def run_main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tool.main(argv)
    return code, out.getvalue(), err.getvalue()


class MainConfigureRemoteTest(unittest.TestCase):
    def test_dry_run_prints_script(self) -> None:
        code, out, _ = run_main(
            ["configure-remote", "user@host", "--host-lan-ip", HOST_IP, "--dry-run"]
        )
        self.assertEqual(code, 0)
        self.assertIn("DRY-RUN", out)
        self.assertIn(f"allow {HOST_IP}", out)
        self.assertIn(f"cmdallow {HOST_IP}", out)
        self.assertIn("bindcmdaddress 0.0.0.0", out)

    def test_dry_run_offline_adds_local_reference(self) -> None:
        code, out, _ = run_main(
            [
                "configure-remote",
                "user@host",
                "--host-lan-ip",
                HOST_IP,
                "--offline",
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("local stratum 10", out)
        code, out, _ = run_main(
            ["configure-remote", "user@host", "--host-lan-ip", HOST_IP, "--dry-run"]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("local stratum", out)

    def test_bad_host_lan_ip(self) -> None:
        code, _, err = run_main(
            ["configure-remote", "user@host", "--host-lan-ip", "nope", "--dry-run"]
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_timeout_is_an_error(self) -> None:
        def expire(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=5)

        with unittest.mock.patch.object(tool.subprocess, "run", expire):
            code, _, err = run_main(
                [
                    "configure-remote",
                    "user@host",
                    "--host-lan-ip",
                    HOST_IP,
                    "--timeout",
                    "5",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("did not finish within 5 s", err)


class EnsureLocalTest(unittest.TestCase):
    def test_does_nothing_when_chronyd_is_found(self) -> None:
        with (
            unittest.mock.patch.object(
                tool.shutil, "which", return_value="/usr/sbin/chronyd"
            ),
            unittest.mock.patch.object(tool.subprocess, "run") as run,
        ):
            code, out, _ = run_main(["ensure-local"])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("already installed", out)

    def test_installs_chrony_when_missing(self) -> None:
        done = subprocess.CompletedProcess([], 0)
        with (
            unittest.mock.patch.object(tool.shutil, "which", return_value=None),
            unittest.mock.patch.object(Path, "is_file", return_value=False),
            unittest.mock.patch.object(
                tool.subprocess, "run", return_value=done
            ) as run,
        ):
            code, _, _ = run_main(["ensure-local"])
        self.assertEqual(code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(any("apt-get" in c and "install" in c for c in commands))
        self.assertTrue(all(c[0] == "sudo" for c in commands))

    def test_failed_install_is_an_error(self) -> None:
        failed = subprocess.CompletedProcess([], 100)
        with (
            unittest.mock.patch.object(tool.shutil, "which", return_value=None),
            unittest.mock.patch.object(Path, "is_file", return_value=False),
            unittest.mock.patch.object(tool.subprocess, "run", return_value=failed),
        ):
            code, _, err = run_main(["ensure-local"])
        self.assertEqual(code, 1)
        self.assertIn("exited 100", err)


class FollowerConfigTest(unittest.TestCase):
    def test_follower_config(self) -> None:
        text = tool.follower_config("192.168.1.50", 123, 12301, Path("/w"))
        lines = text.splitlines()
        self.assertIn("server 192.168.1.50 port 123 iburst minpoll 0 maxpoll 3", lines)
        self.assertIn("port 0", lines)
        self.assertIn("cmdport 12301", lines)
        self.assertIn("bindcmdaddress 127.0.0.1", lines)
        for line in lines:
            self.assertFalse(line.startswith(("allow", "local")), line)

    def test_follower_config_uses_given_server_port(self) -> None:
        text = tool.follower_config("10.0.0.2", 12300, 9, Path("/w"))
        self.assertIn("server 10.0.0.2 port 12300 iburst", text)
        self.assertIn("cmdport 9\n", text)


CAMERA_IP = "192.168.1.50"
CAMERA_ID = "C0A80132"


class StatusPcFollowsTest(unittest.TestCase):
    """The PC's own instance (127.0.0.1) must follow the camera host, which is read for its own state."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def fake_chronyc(self, pc: str, camera: str | None) -> str:
        camera_branch = f"echo '{camera}'" if camera else "echo 'no reply' >&2; exit 1"
        path = self.tmp / "chronyc"
        path.write_text(
            "#!/bin/bash\n"
            'case "$*" in\n'
            f"  *'-h 127.0.0.1'*) echo '{pc}' ;;\n"
            f"  *) {camera_branch} ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def status(self, pc: str, camera: str | None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool.main(
                [
                    "status",
                    "--remote",
                    CAMERA_IP,
                    "--wait",
                    "0",
                    "--chronyc",
                    self.fake_chronyc(pc, camera),
                ]
            )
        return code, out.getvalue(), err.getvalue()

    def test_good(self) -> None:
        code, out, err = self.status(
            remote_line(ref_id=CAMERA_ID), remote_line(ref_id="5BBD5B9D")
        )
        self.assertEqual(code, 0, err)
        self.assertIn("ok: this machine follows " + CAMERA_ID, out)

    def test_pc_follows_another_reference_id(self) -> None:
        code, _, err = self.status(
            remote_line(ref_id="5BBD5B9D"), remote_line(ref_id="5BBD5B9D")
        )
        self.assertEqual(code, 1)
        self.assertIn("not ready:", err)
        self.assertIn("this machine follows 5BBD5B9D", err)
        self.assertIn(CAMERA_ID, err)

    def test_pc_instance_unreadable(self) -> None:
        path = self.tmp / "chronyc"
        path.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool.main(["status", "--remote", CAMERA_IP, "--chronyc", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("cannot be read", err.getvalue())

    def test_camera_host_unreadable(self) -> None:
        code, _, err = self.status(remote_line(ref_id=CAMERA_ID), None)
        self.assertEqual(code, 1)
        self.assertIn("not ready:", err)
        self.assertIn("camera host's chronyd cannot be read", err)

    def test_camera_host_unsynchronised(self) -> None:
        unsynchronised = remote_line(ref_id="5BBD5B9D").replace(
            "Normal", "Not synchronised"
        )
        code, _, err = self.status(remote_line(ref_id=CAMERA_ID), unsynchronised)
        self.assertEqual(code, 1)
        self.assertIn("the camera host is not synchronised", err)

    def test_pc_high_dispersion(self) -> None:
        code, _, err = self.status(
            remote_line(ref_id=CAMERA_ID, dispersion="0.010000000"),
            remote_line(ref_id="5BBD5B9D"),
        )
        self.assertEqual(code, 1)
        self.assertIn("this machine's root dispersion", err)


if __name__ == "__main__":
    unittest.main()
