# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a trusted application predicate without exposing the simulator object."""

import hashlib
import importlib
import inspect
import json
import math
import time
from pathlib import Path


def install_reset_observer(environment, on_reset):
    """Observe both explicit resets and Lab's same-step automatic resets."""
    original = environment._reset_idx

    def reset_indices(env_ids, *args, **kwargs):
        # Mark the boundary before native reset/rendering can publish a new scene.
        # Explicit resets use slices; automatic resets supply index tensors.
        if env_ids is None:
            count = environment.num_envs
        elif isinstance(env_ids, slice):
            count = len(range(environment.num_envs)[env_ids])
        else:
            count = len(env_ids)
        if count:
            on_reset()
        return original(env_ids, *args, **kwargs)

    environment._reset_idx = reset_indices


def load_success_hook(specification):
    module, separator, name = specification.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("Success hook must be an importable module:function")
    callback = getattr(importlib.import_module(module), name)
    if not callable(callback):
        raise TypeError("Success hook is not callable")
    inspect.signature(callback).bind({})
    source = Path(inspect.getfile(callback))
    identity = {
        "entrypoint": specification,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    return callback, identity


def evaluate_success(callback, observation):
    # A detached JSON snapshot prevents accidental mutation of live observation state.
    result = callback(json.loads(json.dumps(observation, allow_nan=False)))
    if type(result) is not bool:
        raise TypeError("Success hook must return bool, not a truthy value")
    return result


def check_driver_lease(path, run_id, timeout):
    if not path.exists():
        return
    lease = json.loads(path.read_text())
    age = time.monotonic() - lease["monotonic"]
    if (
        lease.get("run_id") != run_id
        or not math.isfinite(age)
        or not 0 <= age <= timeout
    ):
        raise RuntimeError("Task driver lease expired or belongs to another observer")
