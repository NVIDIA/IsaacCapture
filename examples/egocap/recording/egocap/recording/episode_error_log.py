"""Persistent per-session error log for egocap recordings.

Appends one JSON object per line (JSONL) to ``<session_root>/errors.jsonl`` for
every episode that fails a guardrail / data-quality gate, so operators have a
single tail-able, greppable record of what went wrong — without scrolling the
terminal or opening the quarantine zarr.

Each line looks like::

    {"ts": "2026-06-26T14:33:01.123456-07:00", "episode_id": 7,
     "kind": "live_abort", "reasons": ["pico_controller: Pico disconnected ..."]}

``kind`` is one of:

* ``live_abort``       — the live guardrail monitor refused to save mid-capture.
* ``data_check``       — the post-recording data check quarantined the episode.
* ``data_check_error`` — the data check itself crashed.

Writing is best-effort and never raises into the recording loop.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

from loguru import logger

ERROR_LOG_FILENAME = "errors.jsonl"


def log_episode_errors(
    session_root: Union[str, Path],
    episode_id: Optional[int],
    kind: str,
    reasons: Iterable[str],
) -> None:
    """Append one episode-failure record to ``<session_root>/errors.jsonl``.

    Args:
        session_root: Recording session directory the log lives under.
        episode_id: Failing episode id (``None`` if not yet assigned).
        kind: ``live_abort`` | ``data_check`` | ``data_check_error``.
        reasons: Human-readable failure reasons.
    """
    try:
        path = Path(session_root) / ERROR_LOG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().astimezone().isoformat(),
            "episode_id": episode_id,
            "kind": kind,
            "reasons": [str(r) for r in reasons],
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # never let logging break a recording session
        logger.warning(f"Failed to append episode error log: {exc}")
