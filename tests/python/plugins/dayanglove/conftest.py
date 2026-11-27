# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import sys

import pytest
from repo_paths import repo_root

sys.path.insert(0, str(repo_root() / "src/plugins/dayanglove"))
from receiver import DEFAULT_PROTOCOL, load_frame_type  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--protocol-python",
        default=os.environ.get("DAYANGLOVE_PROTOCOL_PY", str(DEFAULT_PROTOCOL)),
    )


@pytest.fixture(scope="session")
def frame_type(request):
    protocol = request.config.getoption("--protocol-python")
    if not protocol:
        pytest.fail(
            "Supply --protocol-python or DAYANGLOVE_PROTOCOL_PY from DAYANGlove"
        )
    return load_frame_type(Path(protocol))


@pytest.fixture
def make_frame(frame_type):
    def make(side=1, sequence=1, **changes):
        values = dict(
            schema_version=1,
            glove_id="SIM-L" if side == 1 else "SIM-R",
            hand=side,
            frame_number=sequence,
            device_sequence=sequence % 256,
            received_unix_ns=1793440000123456789,
            monotonic_ns=987654321000,
            mano_pose45=[i * 0.01 for i in range(45)],
            joints21_xyz_m=[i * 0.001 for i in range(63)],
            palm_adc=list(range(210)),
            mirror_x=side == 1,
            simulation=True,
            pose_valid=True,
            identity_source="simulation",
            status=0x25 | (side << 6),
            output_sequence=sequence,
            session_id="session-a",
            response_version="R01",
            pose_source="model_simulation",
            source_timestamp=123.25,
            replay_loop=0,
            recording_wear=0,
            pose_id="",
            device_sequence_valid=False,
        )
        values.update(changes)
        return frame_type(**values)

    return make
