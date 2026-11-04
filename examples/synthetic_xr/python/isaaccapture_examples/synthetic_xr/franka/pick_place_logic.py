# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene selection is separate; execute the identical shared pick-and-place plan."""

import argparse
import json
import hashlib
from pathlib import Path
from .session import FrankaTaskSession
from .task_plan import Scene, execute, cube_top_down
from .xr_backend import XRBackend
from .pick_place_success import destination


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with FrankaTaskSession(
        observation_dir=a.observation_dir, output=a.output
    ) as session:
        initial = session.mark("initial")
        goal = destination(initial)
        q = cube_top_down(initial["cubes"]["cube_2"]["quaternion_world"][0])
        request = {
            "backend": "xr",
            "destination": goal.tolist(),
            "orientation_xyzw": list(q),
            "run_id": initial["run_id"],
            "shared_plan_sha256": hashlib.sha256(
                (Path(__file__).parent / "task_plan.py").read_bytes()
            ).hexdigest(),
        }
        (Path(a.output) / "request.json").write_text(json.dumps(request, indent=2))
        backend = XRBackend(session)
        result = execute(backend, session.mark, Scene(destination=tuple(goal)))
        result["neutral_wait_until_step"] = backend.terminal_step
        (Path(a.output) / "task-result.json").write_text(json.dumps(result, indent=2))
        session.wait_for_export(timeout=60)


if __name__ == "__main__":
    main()
