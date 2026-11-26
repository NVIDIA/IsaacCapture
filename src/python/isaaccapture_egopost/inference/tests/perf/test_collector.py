# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import json
import pytest

from inference.perf.collector import LocalPerfWriter, PerfReporter, PerfStage


def test_perf_reporter_protocol(tmp_path):
    assert isinstance(LocalPerfWriter(output_path=tmp_path / "test.json"), PerfReporter)


def test_write_creates_file(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.INFERENCE, 57.2)
    writer.write()
    assert out.exists()


def test_write_output_schema(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(
        output_path=out,
        metadata={"checkpoint": "ckpt_v2", "device": "A100"},
    )
    writer.record_latency(PerfStage.MODEL_LOAD, 20000.0)
    writer.record_latency(PerfStage.BBOX_DETECTION, 44.0)
    writer.record_latency(PerfStage.BBOX_DETECTION, 46.0)
    writer.record_latency(PerfStage.INFERENCE, 57.0)
    writer.record_latency(PerfStage.INFERENCE, 59.0)
    writer.write()

    record = json.loads(out.read_text())
    assert record["metadata"] == {"checkpoint": "ckpt_v2", "device": "A100"}
    assert record["model_load_ms"] == pytest.approx(20000.0)

    bbox = record["stages"]["bbox_detection"]
    assert bbox["count"] == 2
    assert bbox["samples"] == [44.0, 46.0]
    assert bbox["avg_ms"] == pytest.approx(45.0)

    infer = record["stages"]["inference"]
    assert infer["count"] == 2
    assert infer["avg_ms"] == pytest.approx(58.0)


def test_model_load_not_in_stages(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.MODEL_LOAD, 19000.0)
    writer.record_latency(PerfStage.INFERENCE, 55.0)
    writer.write()

    record = json.loads(out.read_text())
    assert "model_load" not in record["stages"]
    assert record["model_load_ms"] == pytest.approx(19000.0)


def test_no_model_load(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.INFERENCE, 55.0)
    writer.write()

    record = json.loads(out.read_text())
    assert record["model_load_ms"] is None


def test_write_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "dir" / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.INFERENCE, 55.0)
    writer.write()
    assert out.exists()


def test_write_returns_path(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.INFERENCE, 55.0)
    result = writer.write()
    assert result == out


def test_empty_metadata(tmp_path):
    out = tmp_path / "perf.json"
    writer = LocalPerfWriter(output_path=out)
    writer.record_latency(PerfStage.INFERENCE, 55.0)
    writer.write()

    record = json.loads(out.read_text())
    assert record["metadata"] == {}
