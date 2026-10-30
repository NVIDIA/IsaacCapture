"""The minimal deployment-station-YAML startup slice, wired through the real app's lifespan:
a configured station seeds the label/instruction/output-root through the existing config/context
mechanisms, and a configured-but-invalid file blocks recording without crashing backend startup or
capture preview. Uses TestClient(app) so the real lifespan runs, exactly like
test_preview_routes.py; no real CaptureServer is started since none of this needs an actual
capture connection.

``meta.operator_name`` seeds the committed context's operator too, but ONLY as a wire
``default_operator`` prefill hint (see recording_command.py's ``snapshot()``): ``start_episode``
always replaces the committed operator with the browser command's own (required) operator before a
take can start, so this seeded value never itself admits or attributes a take -- see app.py's own
comment at the ``initial_context`` construction site. Whether the frontend may even use that hint
to prefill the entry field's draft is a separate wire flag, ``auto_fill_operator_name`` (off
unless ``meta.auto_fill_operator_name: true`` is set) -- never a signal to auto-commit either way.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from egocap_web import app as app_module
from egocap_web.recording_command import RecordingContext, handle_recording_command, parse_recording_command
from egocap_web.recording_session import RecordingSessionCoordinator, TaskContext
from recording_service import FieldSpec, NumericSourceSpec, ServiceConfig

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))
ALWAYS_READY = lambda: (True, None)  # noqa: E731
TEST_CONTROL_TOKEN = "<test-token>"


def _write_station_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "station.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_configured_station_seeds_label_instruction_and_output_root(monkeypatch, tmp_path):
    output_dir = tmp_path / "recordings"
    path = _write_station_yaml(
        tmp_path,
        f"""
meta:
  egocap_station_name: Test Station
  task_description: Pour water.
  operator_name: Alex
recording:
  data_dir: {output_dir}
""",
    )
    monkeypatch.setenv("EGOCAP_STATION_CONFIG", str(path))
    with TestClient(app_module.app) as client:
        context = client.app.state.recording_context.coordinator.snapshot().context
        assert context.task_text == "Pour water."
        assert context.operator == "Alex"
        # The wire prefill hint a fresh browser tab actually receives.
        assert client.app.state.recording_context.snapshot()["default_operator"] == "Alex"
        # Off by default: this station's YAML doesn't set auto_fill_operator_name.
        assert client.app.state.recording_context.snapshot()["auto_fill_operator_name"] is False
        assert client.app.state.monitor._config.station_label == "Test Station"
        # The SAME root the recordings-browsing UI would read (via app.state), not a separately
        # recomputed value -- this is the YAML's recording.data_dir.
        assert client.app.state.recording_output_root == output_dir


def test_configured_station_can_opt_in_to_auto_filling_the_operator_name(monkeypatch, tmp_path):
    path = _write_station_yaml(
        tmp_path,
        """
meta:
  egocap_station_name: Test Station
  operator_name: Alex
  auto_fill_operator_name: true
""",
    )
    monkeypatch.setenv("EGOCAP_STATION_CONFIG", str(path))
    with TestClient(app_module.app) as client:
        assert client.app.state.recording_context.snapshot()["auto_fill_operator_name"] is True


def test_invalid_station_config_blocks_recording_but_not_capture_preview(monkeypatch, tmp_path):
    path = _write_station_yaml(tmp_path, "meta: [unbalanced\n")  # malformed YAML
    monkeypatch.setenv("EGOCAP_STATION_CONFIG", str(path))
    with TestClient(app_module.app) as client:
        ready, reason = client.app.state.recording_context.coordinator._readiness_check()
        assert ready is False
        assert reason is not None
        # The real failure detail (the tmp_path file's own path) never reaches this reason.
        assert str(path) not in reason

        # An unrelated route is completely unaffected -- capture preview stays up.
        assert client.get("/api/capture-status").status_code == 200


def test_unconfigured_station_preserves_existing_development_default(monkeypatch):
    monkeypatch.delenv("EGOCAP_STATION_CONFIG", raising=False)
    with TestClient(app_module.app) as client:
        context = client.app.state.recording_context.coordinator.snapshot().context
        assert context.task_text == ""
        assert context.operator == ""
        assert client.app.state.recording_context.snapshot()["default_operator"] is None
        assert client.app.state.recording_context.snapshot()["auto_fill_operator_name"] is False
        ready, reason = client.app.state.recording_context.coordinator._readiness_check()
        assert ready is False  # no capture connected -- unrelated to station config
        assert reason != "station configuration is invalid; see backend logs"


async def test_a_committed_operator_in_context_still_requires_the_browsers_own_operator_token():
    """Whatever operator value is already committed in TaskContext (seeded or otherwise), the
    existing attribution/authorization gate in RecordingContext.apply() is untouched -- a command
    with no operator_token is still rejected, exactly as it is without a station configured."""

    def factory() -> ServiceConfig:
        return ServiceConfig(
            output_root=Path("/tmp/does-not-matter"), numeric_sources=(NUMERIC,), video_sources=()
        )

    coordinator = RecordingSessionCoordinator(
        factory,
        queue_capacity=8,
        readiness_check=ALWAYS_READY,
        initial_context=TaskContext(task_text="Pour water.", operator="Alex"),
    )
    context = RecordingContext(coordinator=coordinator, control_token=TEST_CONTROL_TOKEN)
    reply = await handle_recording_command(
        parse_recording_command(
            {
                "request_id": "r1",
                "action": "set_task",
                "payload": {"batch_id": context.batch_id, "task": "Something else"},
                "operator_token": None,
                "control_token": TEST_CONTROL_TOKEN,
            }
        ),
        context,
    )
    assert reply["outcome"] == "rejected"
    assert "operator" in reply["reason"].lower()
    # Unaffected by the rejection -- the seeded context is still what a real command would use.
    assert coordinator.snapshot().context.task_text == "Pour water."
    assert coordinator.snapshot().context.operator == "Alex"
