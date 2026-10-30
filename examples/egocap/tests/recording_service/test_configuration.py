"""Configuration validation: refused before any effect, with no fixture required."""

from pathlib import Path

import pytest

from recording_service import (
    ConfigurationError,
    FieldSpec,
    NumericSourceSpec,
    QcConfig,
    RecordingService,
    ServiceConfig,
    VideoSourceSpec,
)

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))
VIDEO = VideoSourceSpec("cam", 64, 48, 30.0)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"quarantine": True}, "quarantine"),
        ({"numeric_sources": (), "video_sources": ()}, "at least one"),
        ({"numeric_sources": (NUMERIC, NUMERIC)}, "duplicate"),
        ({"numeric_sources": (NumericSourceSpec("../x", NUMERIC.fields),)}, "safe path"),
        ({"qc": QcConfig(required_sources=("missing",))}, "required sources"),
        ({"video_sources": (VideoSourceSpec("cam", 64, 48, 30.0, "bogus"),)}, "encoder"),
        ({"queue_capacity": 0}, "positive"),
        ({"numeric_sources": (NumericSourceSpec("s", (FieldSpec("capture_time", "float64"),)),)}, "timestamp"),
        (
            {"numeric_sources": (NumericSourceSpec("s", (FieldSpec("v", "float64"), FieldSpec("v", "int8"))),)},
            "duplicate",
        ),
        ({"numeric_sources": (NumericSourceSpec("s", (FieldSpec("../v", "float64"),)),)}, "safe array name"),
        ({"numeric_sources": (NumericSourceSpec("s", (FieldSpec("v", "not-a-dtype"),)),)}, "invalid dtype"),
        (
            {"numeric_sources": (NumericSourceSpec("s", (FieldSpec("v", "U5"),)),)},
            "must be bool, integer or float",
        ),
        ({"numeric_sources": (NumericSourceSpec("s", (FieldSpec("v", "float64", (-1,)),)),)}, "invalid shape"),
        ({"video_sources": (VideoSourceSpec("cam", 64, 48, 30.0, "ffmpeg_h264", True, {"crf": 99}),)}, "crf"),
        ({"video_sources": (VideoSourceSpec("cam", 64, 48, 30.0, "ffmpeg_h264", True, {"bogus": 1}),)}, "bogus"),
        (
            {"video_sources": (VideoSourceSpec("cam", 64, 48, 30.0, "opencv_mp4v", True, {"crf": 18}),)},
            "only supported",
        ),
        ({"latency": {"nope": 0.1}}, "unknown source"),
        ({"latency": {"cam": float("nan")}}, "finite"),
        ({"latency": {"cam": -1.0}}, "finite"),
    ],
)
async def test_bad_configuration_is_refused_before_any_effect(tmp_path: Path, overrides, message):
    """Each invalid configuration raises ConfigurationError and creates nothing under the output root."""
    base = {"output_root": tmp_path / "out", "numeric_sources": (NUMERIC,), "video_sources": (VIDEO,)}
    service = RecordingService(ServiceConfig(**{**base, **overrides}))
    with pytest.raises(ConfigurationError, match=message):
        await service.open_session()
    assert not any((tmp_path / "out").glob("*")) if (tmp_path / "out").exists() else True
