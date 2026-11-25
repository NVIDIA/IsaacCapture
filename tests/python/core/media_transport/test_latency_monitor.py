# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import json
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from isaac_media_transport import latency_monitor
from isaac_media_transport.latency import ClockMap
from isaac_media_transport.latency_monitor import (
    ClockSamples,
    JsonlTail,
    OnlineJoiner,
    RemoteReader,
    clock_identity,
    helper_script,
    interval_between,
    monitor,
    recording_guide,
    recording_schedule,
    terminal_summary,
    wait_for_traces,
)


def event(stage, at, *, clock="host", stamp=100, run_id="trial"):
    return dict(
        record_type="stage",
        run_id=run_id,
        track="camera.left",
        source_timestamp_ns=stamp,
        stage=stage,
        monotonic_ns=at,
        clock_id=clock,
        pid=10,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_monitor_derives_role_logs_and_preserves_explicit_overrides(
    monkeypatch, tmp_path, explicit
):
    captured = {}

    async def collect(args):
        captured.update(vars(args))

    argv = [
        "latency_monitor",
        "--client",
        "user@client",
        "--run-id",
        "trial-A1",
        "--log-dir",
        "~/traces",
        "--output",
        str(tmp_path / "measurement"),
    ]
    if explicit:
        argv += ["--client-log", "/old/sender.jsonl", "--host-log", "/old/viewer.jsonl"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(latency_monitor, "monitor", collect)
    latency_monitor.main()
    assert captured["client_log"] == (
        ["/old/sender.jsonl"] if explicit else ["~/traces/umb-trial-A1-sender.jsonl"]
    )
    assert captured["host_log"] == (
        ["/old/viewer.jsonl"]
        if explicit
        else [
            "~/traces/umb-trial-A1-receiver.jsonl",
            "~/traces/umb-trial-A1-viewer.jsonl",
        ]
    )
    assert not (tmp_path / "measurement").exists()


def test_asymmetric_probe_contains_true_offset_and_retains_uncertainty():
    samples = ClockSamples("host", drift_ppm=0)
    sample = samples.add(
        1_000_000_000, 5_001_000_000, 5_001_100_000, 1_004_100_000, "orin"
    )
    lo, hi, _ = samples.best(5_010_000_000, 1_020_000_000)
    assert lo <= -4_000_000_000 <= hi
    assert lo == sample.offset_lower_ns
    assert hi - lo == 4_002_000
    result = interval_between(
        event("transport_submit", 5_010_000_000, clock="orin"),
        event("transport_available", 1_020_000_000),
        samples,
        1_020_000_000,
    )
    assert result["lower_ms"] <= 10 <= result["upper_ms"]
    assert result["uncertainty_ms"] > 2


def test_selects_minimum_expanded_width_and_rejects_stale_maps():
    samples = ClockSamples("host")
    samples.add(1_000_000_000, 5_000_000_000, 5_000_000_000, 1_000_002_000, "orin")
    newest = samples.add(
        2_000_000_000, 6_000_000_000, 6_000_000_000, 2_001_000_000, "orin"
    )
    assert samples.best(6_001_000_000, 2_002_000_000)[2] is newest
    with pytest.raises(ValueError, match="fresh"):
        samples.best(9_000_000_000, 5_000_000_000)


def test_probe_restart_inconsistency_and_budget_fail_closed():
    samples = ClockSamples("host", uncertainty_ns=100)
    samples.add(100000, 1000000, 1000000, 102000, "orin")
    with pytest.raises(ValueError, match="uncertainty"):
        samples.best(1000000, 102000)
    with pytest.raises(ValueError, match="identity"):
        samples.add(103000, 1003000, 1003000, 104000, "other-boot")
    assert not samples.samples
    with pytest.raises(ValueError, match="ordering"):
        samples.add(1, 100, 200, 2, "orin")


def test_contradictory_sample_clears_previous_estimates():
    samples = ClockSamples("host", drift_ppm=0)
    samples.add(100000, 1000000, 1000000, 102000, "orin")
    with pytest.raises(ValueError, match="disagree"):
        samples.add(103000, 2000000, 2000000, 104000, "orin")
    assert not samples.samples


def test_tail_waits_for_complete_line_and_refuses_truncation(tmp_path):
    path = tmp_path / "frames.jsonl"
    tail = JsonlTail(path)
    assert tail.read() == []
    path.write_text('{"stage":')
    assert tail.read() == []
    with path.open("a") as stream:
        stream.write('"encoded_ready"}\n')
    assert tail.read() == [{"stage": "encoded_ready"}]
    assert tail.read() == []
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        tail.read()


def test_tail_bounds_records_and_detects_replacement(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_text("{}\n")
    tail = JsonlTail(path)
    assert tail.read() == [{}]
    replacement = tmp_path / "new"
    replacement.write_text("{}\n")
    replacement.replace(path)
    with pytest.raises(ValueError, match="replaced"):
        tail.read()
    replacement.write_text("x" * 17000)
    with pytest.raises(ValueError, match="size limit"):
        JsonlTail(replacement).read()


def test_follow_skips_history_without_losing_an_in_progress_record(tmp_path):
    path = tmp_path / "frames.jsonl"
    history = b'{"old":true}\n' * 50_000
    path.write_bytes(history + b'{"new":')
    tail = JsonlTail(path)
    assert tail.follow() == len(history)
    assert tail.read() == []
    with path.open("ab") as stream:
        stream.write(b"true}\n")
    assert tail.read() == [{"new": True}]
    assert path.read_bytes().startswith(history)


def test_follow_allows_late_creation_and_still_detects_replacement(tmp_path):
    path = tmp_path / "frames.jsonl"
    tail = JsonlTail(path)
    assert tail.follow() == 0
    path.write_text('{"new":true}\n')
    assert tail.read() == [{"new": True}]
    other = tmp_path / "replacement.jsonl"
    other.write_text("{}\n")
    other.replace(path)
    with pytest.raises(ValueError, match="replaced"):
        tail.read()


def test_online_partial_spans_are_separate_and_do_not_invent_missing_stages():
    joiner = OnlineJoiner("trial", ClockSamples("host"))
    for stage, at in (
        ("capture_available", 100),
        ("encoder_submit", 200),
        ("decode_ready", 500),
        ("viewer_submit_begin", 800),
    ):
        joiner.add(event(stage, at), 1000)
    assert joiner.flush(1001) == []
    row = joiner.flush(2000, force=True)[0]
    assert set(row["contributors"]) == {"capture_available->encoder_submit"}
    assert "encoder_submit->decode_ready" in row["partial_spans"]
    assert "encoded_ready" in row["missing_stages"]
    assert (
        row["totals"]["capture_available->viewer_submit_begin"]["estimate_ms"] == 0.0007
    )
    assert joiner.summary()["counters"]["partial_frames"] == 1


def test_unmapped_transport_retains_local_contributors():
    joiner = OnlineJoiner("trial", ClockSamples("host"))
    joiner.add(event("transport_submit", 1000, clock="orin"), 1000)
    joiner.add(event("transport_available", 2000), 1000)
    joiner.add(event("decode_begin", 2200), 1000)
    row = joiner.flush(2000, force=True)[0]
    assert (
        row["contributors"]["transport_submit->transport_available"]["estimate_ms"]
        is None
    )
    assert (
        row["contributors"]["transport_available->decode_begin"]["estimate_ms"]
        == 0.0002
    )


def test_duplicate_and_late_records_are_visible_not_rematched():
    joiner = OnlineJoiner("trial", ClockSamples("host"))
    joiner.add(event("capture_available", 1), 1)
    joiner.add(event("encoder_submit", 2), 1)
    joiner.add(event("encoder_submit", 3), 1)
    joiner.add(event("encoder_submit", 4), 1)
    row = joiner.flush(10, force=True)[0]
    assert not row["contributors"]
    assert "encoder_submit" in row["missing_stages"]
    joiner.add(event("encoded_ready", 5), 20)
    joiner.add(event("encoded_ready", 5, run_id="different-trial"), 20)
    assert not joiner.frames
    assert joiner.counters["late_events"] == 1
    assert joiner.counters["other_run_events"] == 1


def test_helper_protocol_clock_and_incremental_trace_reads(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(event("capture_available", 1000)) + "\n")

    async def check():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            helper_script(),
            str(path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        remote = RemoteReader(process)
        try:
            h1, h4, response = await remote.request("probe")
            assert (
                h1 <= response["client_received_ns"] <= response["client_sent_ns"] <= h4
            )
            assert response["clock_id"]
            _, _, response = await remote.request("read")
            assert response["events"] == [event("capture_available", 1000)]
            _, _, response = await remote.request("read")
            assert response["events"] == []
            with path.open("a") as stream:
                stream.write(json.dumps(event("capture_available", 2000)) + "\n")
            _, _, response = await remote.request("follow")
            assert response["offsets"] == [path.stat().st_size]
            with path.open("a") as stream:
                stream.write(json.dumps(event("capture_available", 3000)) + "\n")
            _, _, response = await remote.request("read")
            assert response["events"] == [event("capture_available", 3000)]
        finally:
            await remote.close()
        assert process.returncode == 0

    asyncio.run(check())


@pytest.mark.parametrize("slow_read", [False, True])
def test_monitor_writes_live_results_and_manifest_with_local_ssh_double(
    tmp_path, monkeypatch, slow_read, capsys
):
    source, host = tmp_path / "source.jsonl", tmp_path / "host.jsonl"
    now = time.monotonic_ns()
    source.write_text(json.dumps(event("transport_submit", now, clock="orin")) + "\n")
    host.write_text(
        "".join(
            json.dumps(event(stage, now + offset, clock=clock_identity())) + "\n"
            for stage, offset in (
                ("transport_available", 10_000_000),
                ("decode_begin", 11_000_000),
            )
        )
    )
    create_process = asyncio.create_subprocess_exec

    async def local_helper(*command, **options):
        remote_command = shlex.split(command[-1])
        script = remote_command[remote_command.index("-c") + 1]
        script = script.replace("clock_id = clock_identity()", "clock_id = 'orin'")
        return await create_process(
            sys.executable, "-u", "-c", script, str(source), **options
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_helper)
    args = SimpleNamespace(
        output=tmp_path / "result",
        client="user@host",
        client_log=[str(source)],
        host_log=[str(host)],
        client_python=sys.executable,
        run_id="trial",
        duration=0.1,
        warmup=0.05,
        clock_max_age=2,
        drift_ppm=1000,
        max_uncertainty_ms=5,
    )
    request = RemoteReader.request
    sent_frame = False

    async def delayed_request(self, kind):
        nonlocal sent_frame
        if kind == "status":
            for path in (source, host):
                with path.open("a") as stream:
                    stream.write(
                        json.dumps(event("capture_available", time.monotonic_ns()))
                        + "\n"
                    )
        if kind == "read" and not sent_frame:
            await asyncio.sleep(0.06)
            current = time.monotonic_ns()
            with source.open("a") as stream:
                for stage, offset in (
                    ("capture_available", -1_000_000),
                    ("transport_submit", 0),
                ):
                    stream.write(
                        json.dumps(
                            event(stage, current + offset, clock="orin", stamp=200)
                        )
                        + "\n"
                    )
            with host.open("a") as stream:
                for stage, offset in (
                    ("transport_available", 10_000_000),
                    ("decode_begin", 11_000_000),
                    ("viewer_submit_begin", 12_000_000),
                ):
                    stream.write(
                        json.dumps(
                            event(
                                stage,
                                current + offset,
                                clock=clock_identity(),
                                stamp=200,
                            )
                        )
                        + "\n"
                    )
            sent_frame = True
        if kind == "read" and slow_read:
            await asyncio.sleep(0.2)
            # A stalled SSH read must not defer the end-of-recording prompt.
            cues = (args.output / "results.jsonl").read_text()
            assert '"cue":"stop_recording"' in cues
        return await request(self, kind)

    monkeypatch.setattr(RemoteReader, "request", delayed_request)
    result = asyncio.run(monitor(args))
    assert result["error"] is None
    assert result["counters"]["frames"] == 1
    assert result["valid_total_count"] == 1
    assert result["counters"].get("late_events", 0) == 0
    records = [
        json.loads(line)
        for line in (args.output / "results.jsonl").read_text().splitlines()
    ]
    frame = next(
        record for record in records if record["record_type"] == "frame_result"
    )
    assert frame["source_timestamp_ns"] == 200
    transport = frame["contributors"]["transport_submit->transport_available"]
    assert transport["lower_ms"] <= 10 <= transport["upper_ms"]
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["clock_method"] == "ssh_4timestamp_monotonic"
    assert manifest["relative_drift_assumption_ppm"] == 1000
    assert (args.output / "summary.json").is_file()
    window = next(r for r in records if r["record_type"] == "measurement_window")
    cues = [r for r in records if r["record_type"] == "operator_cue"]
    for cue, boundary in (
        ("warmup", "collection_start_ns"),
        ("start_recording", "measurement_start_ns"),
        ("stop_recording", "measurement_end_ns"),
    ):
        row = next(r for r in cues if r["cue"] == cue)
        assert row["scheduled_monotonic_ns"] == window[boundary]
        assert row["emitted_monotonic_ns"] >= window[boundary]
    output = capsys.readouterr().out
    assert "PLEASE WAIT 0.05 SECONDS FOR WARMUP PERIOD" in output
    assert output.count("START RECORDING LATENCY VIDEO NOW") == 1
    assert output.count("MEASUREMENT IS DONE, STOP RECORDING LATENCY VIDEO") == 1


def test_recording_schedule_covers_both_countdowns_and_exact_phase_boundaries():
    cues = list(recording_schedule(90_000_000_000, 60_000_000_000))
    elapsed = [offset / 1e9 for offset, cue, _ in cues if cue == "elapsed"]
    assert elapsed == sorted(
        set(range(5, 151, 5)) | set(range(85, 91)) | set(range(145, 151))
    )
    assert [(offset / 1e9, cue) for offset, cue, _ in cues if cue != "elapsed"] == [
        (0, "warmup"),
        (90, "start_recording"),
        (150, "stop_recording"),
    ]


@pytest.mark.parametrize("warmup,duration", [(0, 60), (2.5, 3.25), (17, 11)])
def test_recording_guide_uses_absolute_deadlines_for_custom_durations(
    monkeypatch, warmup, duration
):
    started = now = 1_000_000_000
    results = io.StringIO()
    woke_early = False

    async def sleep(seconds):
        nonlocal now, woke_early
        # Early wakeups cannot advance cues; late wakeups cannot accumulate drift.
        jitter = 125_000_000 if woke_early else -1_000_000
        now += round(seconds * 1e9) + jitter
        woke_early = True

    monkeypatch.setattr(latency_monitor.time, "monotonic_ns", lambda: now)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(
        recording_guide(
            started,
            started + int(warmup * 1e9),
            started + int((warmup + duration) * 1e9),
            results,
        )
    )
    cues = [json.loads(line) for line in results.getvalue().splitlines()]
    for cue in cues:
        assert (
            0
            <= cue["emitted_monotonic_ns"] - cue["scheduled_monotonic_ns"]
            <= 125_000_000
        )
    assert [r["cue"] for r in cues if r["cue"] != "elapsed"] == (
        (["warmup"] if warmup else []) + ["start_recording", "stop_recording"]
    )
    assert cues[-1]["scheduled_monotonic_ns"] == started + int(
        (warmup + duration) * 1e9
    )


@pytest.mark.parametrize("warmup", [0, 90])
@pytest.mark.parametrize("error", [OSError("client disconnected"), TimeoutError()])
def test_monitor_cancels_recording_guide_on_read_failure(
    tmp_path, monkeypatch, warmup, error, capsys
):
    closed = []

    async def create_process(*args, **kwargs):
        return object()

    class FailingReader:
        def __init__(self, process):
            pass

        async def request(self, kind):
            if kind == "follow":
                return 0, 0, {"offsets": []}
            if kind == "status":
                return 0, 0, {"traces": []}
            if kind == "read":
                await asyncio.sleep(0)
                raise error
            now = time.monotonic_ns()
            return (
                now,
                now + 100,
                dict(client_received_ns=now, client_sent_ns=now, clock_id="orin"),
            )

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(latency_monitor, "RemoteReader", FailingReader)
    args = SimpleNamespace(
        output=tmp_path / "result",
        client="user@host",
        client_log=[],
        host_log=[],
        client_python=sys.executable,
        run_id="trial",
        duration=60,
        warmup=warmup,
        clock_max_age=2,
        drift_ppm=1000,
        max_uncertainty_ms=5,
    )
    with pytest.raises(RuntimeError, match=str(error) or type(error).__name__):
        asyncio.run(monitor(args))
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["error"] == (str(error) or type(error).__name__)
    cues = [
        json.loads(line)
        for line in (args.output / "results.jsonl").read_text().splitlines()
    ]
    assert any(r.get("cue") == "aborted" for r in cues)
    assert not any(r.get("cue") == "stop_recording" for r in cues)
    assert closed == [True]
    output = capsys.readouterr().out
    assert "MEASUREMENT ABORTED, STOP RECORDING LATENCY VIDEO" in output
    assert "MEASUREMENT IS DONE" not in output


def test_capture_phase_excludes_history_warmup_and_partial_crossing_frames():
    joiner = OnlineJoiner(
        "trial",
        ClockSamples("host"),
        collection_start_ns=100,
        measurement_start_ns=200,
        measurement_end_ns=400,
    )
    for stamp, capture, submit in (
        (1, 50, 350),
        (2, 150, 250),
        (3, 250, 300),
        (4, 450, 500),
    ):
        joiner.add(event("capture_available", capture, stamp=stamp), 600)
        joiner.add(event("viewer_submit_begin", submit, stamp=stamp), 600)
    joiner.add(event("viewer_submit_begin", 300, stamp=5), 600)
    rows = joiner.flush(600, force=True)
    assert [row["phase"] for row in rows] == [
        "historical",
        "warmup",
        "measurement",
        "after_measurement",
        "unqualified",
    ]
    summary = joiner.summary()
    assert summary["valid_total_count"] == 1
    assert summary["availability"] == "measured"
    metric = summary["metrics"][
        "camera.left/totals/capture_available->viewer_submit_begin"
    ]
    assert metric["count"] == 1
    assert metric["mean_ms"] == 0.00005
    assert (
        rows[1]["totals"]["capture_available->viewer_submit_begin"]["estimate_ms"]
        == 0.0001
    )


def test_uncertain_capture_boundary_cannot_enter_measurement_distribution():
    clocks = ClockSamples("host")
    clocks.source = "orin"
    clocks.samples.append(ClockMap("orin", "host", -900, -880, 1000, 120, 10000, 0))
    joiner = OnlineJoiner(
        "trial",
        clocks,
        collection_start_ns=0,
        measurement_start_ns=110,
        measurement_end_ns=400,
    )
    joiner.add(event("capture_available", 1000, clock="orin"), 200)
    joiner.add(event("viewer_submit_begin", 200), 200)
    row = joiner.flush(300, force=True)[0]
    assert row["phase"] == "boundary"
    assert row["capture_host_interval_ns"] == {"lower_ns": 100, "upper_ns": 120}
    assert (
        row["totals"]["capture_available->viewer_submit_begin"]["estimate_ms"]
        is not None
    )
    assert joiner.summary()["valid_total_count"] == 0
    assert not joiner.summary()["metrics"]


def test_terminal_summary_is_explicit_for_no_timings_and_bounded_for_metrics():
    joiner = OnlineJoiner("trial", ClockSamples("host"))
    warmup = terminal_summary(joiner.summary(), "warmup")
    assert "measured statistics start after warmup" in warmup
    assert "Frames joined=0" in warmup
    assert "NO FRAME TIMINGS" not in warmup
    assert "top 3" not in warmup
    assert "NO FRAME TIMINGS — not a latency result" in terminal_summary(
        joiner.summary(), "finished"
    )
    for stage, at in (
        ("capture_available", 0),
        ("encoder_submit", 10),
        ("encoded_ready", 30),
        ("access_unit_ready", 40),
        ("publish_begin", 70),
        ("viewer_submit_begin", 100),
    ):
        joiner.add(event(stage, at), 1000)
    joiner.flush(1000, force=True)
    text = terminal_summary(joiner.summary(), "measurement")
    assert "partial P2P mean/p95=" in text
    assert "n=1 unqualified=0 excluded=0" in text
    assert text.count(" ms") == 4
    assert "NO FRAME TIMINGS" not in text


def test_delayed_python_submit_return_is_not_in_primary_total_or_contributors():
    joiner = OnlineJoiner("trial", ClockSamples("host"))
    for stage, at in (
        ("capture_available", 100),
        ("viewer_submit_begin", 200),
        ("viewer_submit_end", 9_000_000),
    ):
        joiner.add(event(stage, at), 10_000_000)
    row = joiner.flush(10_000_000, force=True)[0]
    total = row["totals"]["capture_available->viewer_submit_begin"]
    assert total["estimate_ms"] == 0.0001
    assert (
        row["diagnostics"]["viewer_submit_begin->viewer_submit_end"]["estimate_ms"]
        == 8.9998
    )
    assert "viewer_submit_begin->viewer_submit_end" not in row["contributors"]
    assert "viewer_submit_end" not in row["missing_stages"]
    summary = joiner.summary()
    assert summary["valid_total_count"] == 1
    assert (
        summary["metrics"]["camera.left/totals/capture_available->viewer_submit_begin"][
            "mean_ms"
        ]
        == 0.0001
    )


def test_trace_status_ignores_partial_lines_and_does_not_consume_tail(tmp_path):
    path = tmp_path / "frames.jsonl"
    tail = JsonlTail(path)
    assert tail.status()["run_id"] is None
    stage = event("capture_available", 123)
    path.write_text(json.dumps(stage) + '\n{"record_type":"stage"')
    status = tail.status()
    assert status["run_id"] == "trial"
    assert status["stage_monotonic_ns"] == 123
    assert tail.read() == [stage]


@pytest.mark.parametrize("missing", [False, True])
def test_trace_readiness_rejects_stale_or_missing_logs(tmp_path, missing):
    path = tmp_path / "viewer.jsonl"
    if not missing:
        path.write_text(json.dumps(event("viewer_submit_begin", 123)) + "\n")

    class Reader:
        async def request(self, kind):
            assert kind == "status"
            return 0, 0, {"traces": []}

    with pytest.raises(ValueError, match=r"No fresh frame traces.*trial"):
        asyncio.run(wait_for_traces(Reader(), [JsonlTail(path)], "trial", timeout=0))


def test_trace_readiness_rejects_wrong_run_id(tmp_path):
    path = tmp_path / "sender.jsonl"
    path.write_text(json.dumps(event("capture_available", 123, run_id="actual")) + "\n")

    class Reader:
        async def request(self, kind):
            return 0, 0, {"traces": [JsonlTail(path).status()]}

    with pytest.raises(ValueError, match=r"Trace run ID mismatch.*actual.*expected"):
        asyncio.run(wait_for_traces(Reader(), [], "expected", timeout=0))


def test_trace_readiness_requires_new_stages_on_both_sites(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("sender.jsonl", "viewer.jsonl")]
    for path in paths:
        path.write_text(json.dumps(event("capture_available", 100)) + "\n")
    checks = 0

    class Reader:
        async def request(self, kind):
            nonlocal checks
            checks += 1
            with paths[0].open("a") as stream:
                stream.write(
                    json.dumps(event("capture_available", 100 + checks)) + "\n"
                )
            with paths[1].open("a") as stream:
                record = (
                    event("viewer_submit_begin", 200)
                    if checks == 3
                    else {"record_type": "telemetry"}
                )
                stream.write(json.dumps(record) + "\n")
            return 0, 0, {"traces": [JsonlTail(paths[0]).status()]}

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    asyncio.run(wait_for_traces(Reader(), [JsonlTail(paths[1])], "trial"))
    assert checks == 3


def test_rejected_clock_probe_is_saved_before_monitor_aborts(tmp_path, monkeypatch):
    class Reader:
        def __init__(self, process):
            self.calls = 0

        async def request(self, kind):
            self.calls += 1
            now = 1_000_000_000 + self.calls * 10_000
            client = now + (1_000_000_000 if self.calls == 2 else 0)
            return (
                now,
                now + 100,
                dict(client_received_ns=client, client_sent_ns=client, clock_id="orin"),
            )

        async def close(self):
            pass

    async def process(*args, **kwargs):
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", process)
    monkeypatch.setattr(latency_monitor, "RemoteReader", Reader)
    args = SimpleNamespace(
        output=tmp_path / "result",
        client="user@host",
        client_log=[],
        host_log=[],
        client_python=sys.executable,
        run_id="trial",
        duration=1,
        warmup=0,
        clock_max_age=2,
        drift_ppm=1000,
        max_uncertainty_ms=5,
    )
    with pytest.raises(RuntimeError, match="intervals separated"):
        asyncio.run(monitor(args))
    records = [
        json.loads(line)
        for line in (args.output / "events.jsonl").read_text().splitlines()
    ]
    assert [row["record_type"] for row in records] == ["clock_sample", "clock_rejected"]
    rejected = records[-1]
    assert rejected["client_received_ns"] == 2_000_020_000
    assert rejected["host_sent_ns"] == 1_000_020_000
    assert "1000 ppm" in rejected["error"]
