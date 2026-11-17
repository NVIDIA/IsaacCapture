# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full stack's GPU parts on the workstation, on GPU 0 and nothing else.

Run in the workstation's APAI Python environment (setup_workstation_env.sh),
with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0. Every test is
marked "gpu" and "experiment": without torch with CUDA, or without the
experiment folder, the whole file is skipped. The experiment folder and APAI's
checkout are only read: the caches the experiment's code would keep there go to
a pytest temporary folder. No test asks a model.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import APAI_SRC_ENV, apai_runtime_env, import_experiment

pytestmark = [pytest.mark.gpu, pytest.mark.experiment]
apai = pytest.mark.apai

# GPU 0 in PCI order, as nvidia-smi names it: bus C1, device 00, function 0.
GPU_0_PCI = "C1:00.0"
# The recorded stereo pair (left_raw.jpg, right_raw.jpg) in the experiment folder.
RECORDED = ".state/thor/captures/20261106T230400Z-gpu-apai"
# The cache variables the experiment's code would point into its own folder.
CACHE_VARIABLES = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "YOLO_CONFIG_DIR": "ultralytics",
}
# The experiment's stereo depth sets these two when they are not set.
PTXAS_VARIABLES = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")
# The longest one request to the obstacle grid bridge may take, in ms: the
# first one compiles Warp's kernels.
BRIDGE_REQUEST_MS = 180_000
# The longest the launcher waits for a usable obstacle grid, in s: the first
# camera frame waits for the stereo depth model to load and warm up.
GRID_WAIT_S = 240.0

KERNEL_MODULE = """\
import warp as wp


@wp.kernel
def scale(values: wp.array(dtype=wp.float32), factor: float):
    i = wp.tid()
    values[i] = values[i] * factor
"""


@pytest.fixture(scope="session")
def cache_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One folder of caches for the file, so a model is compiled once."""
    folder = tmp_path_factory.mktemp("gpu-caches")
    for leaf in CACHE_VARIABLES.values():
        (folder / leaf).mkdir()
    return folder


@pytest.fixture(autouse=True)
def gpu_env(
    cache_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """GPU 0 only, and nothing written into the experiment folder or APAI.

    The test is skipped unless this process sees exactly one device, index 0 in
    PCI order.
    """
    order = os.environ.get("CUDA_DEVICE_ORDER", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if order != "PCI_BUS_ID" or visible != "0":
        pytest.skip(
            "the gpu tests run on GPU 0 only: CUDA_DEVICE_ORDER must be "
            f"PCI_BUS_ID (it is {order!r}) and CUDA_VISIBLE_DEVICES must be 0 "
            f"(it is {visible!r})"
        )
    for name, leaf in CACHE_VARIABLES.items():
        monkeypatch.setenv(name, str(cache_folder / leaf))
    # No .pyc next to a module of the experiment or of APAI, here or in a child.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    # git status, which the stereo depth runs on its source, writes no index.
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")
    # No model client finds a login.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-model-login"))
    # torch brings its own CUDA libraries.
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    # The obstacle grid bridge is a child process that imports APAI.
    source = os.environ.get(APAI_SRC_ENV, "")
    if source:
        inherited = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv(
            "PYTHONPATH", f"{source}{os.pathsep}{inherited}" if inherited else source
        )
    before = {name: os.environ.get(name) for name in PTXAS_VARIABLES}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def recorded_folder(experiment_dir: Path) -> Path:
    folder = experiment_dir / RECORDED
    if not (folder / "left_raw.jpg").is_file():
        pytest.skip(f"no recorded stereo pair in {folder}")
    return folder


def pci_address(torch: Any) -> str:
    """Bus, device and function of torch's device 0, as nvidia-smi prints them."""
    properties = torch.cuda.get_device_properties(0)
    bus = getattr(properties, "pci_bus_id", None)
    device = getattr(properties, "pci_device_id", None)
    if bus is not None and device is not None:
        return f"{int(bus):02X}:{int(device):02X}.0"
    # An older torch: look the device's UUID up in nvidia-smi's list.
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        pytest.fail("torch names neither the PCI address nor the UUID of device 0")
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    for line in listed.splitlines():
        name, _, address = line.partition(",")
        if name.strip() == f"GPU-{uuid}":
            return address.strip().upper()[-len(GPU_0_PCI) :]
    pytest.fail(f"nvidia-smi lists no device GPU-{uuid}:\n{listed}")


def test_torch_sees_one_cuda_device_and_it_is_gpu_0() -> None:
    torch = pytest.importorskip("torch")
    assert torch.version.cuda, "this torch was built without CUDA"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    address = pci_address(torch)
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}: "
        f"{torch.cuda.get_device_name(0)} at {address}"
    )
    assert address == GPU_0_PCI
    values = torch.arange(8, dtype=torch.float32, device="cuda:0")
    assert values.device.type == "cuda" and values.device.index == 0
    assert float((values * 2).sum().item()) == 56.0


def test_warp_runs_on_cuda_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wp = pytest.importorskip("warp")
    np = pytest.importorskip("numpy")
    wp.config.quiet = True
    wp.init()
    assert wp.is_cuda_available() and wp.get_cuda_device_count() == 1
    device = wp.get_device("cuda:0")
    assert device.is_cuda
    # A kernel in a module of its own, as Warp compiles them.
    name = "agentic_remote_gpu_kernel"
    file = tmp_path / f"{name}.py"
    file.write_text(KERNEL_MODULE)
    spec = importlib.util.spec_from_file_location(name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    began = time.perf_counter()
    values = wp.array(np.arange(8, dtype=np.float32), dtype=wp.float32, device=device)
    wp.launch(module.scale, dim=8, inputs=[values, 3.0], device=device)
    wp.synchronize_device(device)
    took = time.perf_counter() - began
    print(f"warp {wp.__version__} on {device}: compiled and ran in {took:.2f} s")
    assert str(values.device) == "cuda:0"
    assert np.array_equal(values.numpy(), np.arange(8, dtype=np.float32) * 3.0)


@apai
def test_the_grid_bridge_starts_on_cuda_0_and_answers(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("warp")
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack_numpy")
    import test_agent_end_to_end as end_to_end
    from apai.perception.occupancy import OccupancyClient

    rig = import_experiment("thor_calibration_rig")
    before = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    low, high = (-0.2, -0.5, -0.1), (0.8, 0.5, 0.7)
    # As the launcher starts it (launcher.py, agent_hooks, build).
    bridge = rig.OwnedOccupancyBridge(
        tmp_path, low, high, device="cuda:0", target_retirement=True
    ).start()
    process = bridge.process
    client = OccupancyClient("127.0.0.1", bridge.port, timeout_ms=BRIDGE_REQUEST_MS)
    try:
        probe = client.probe(timeout_ms=10_000)
        bridge.verify(probe)  # the launcher's own check of the probe
        print(f"obstacle grid bridge: {probe['describe']}, pid {probe['pid']}")
        assert probe["backend"] == "warp" and probe["device"] == "cuda:0"
        assert probe["masked_depth"] is True and probe["target_retirement"] is True
        assert probe["pid"] == process.pid != os.getpid()

        # A camera 1 m above the origin of the map, looking straight down at a
        # flat floor at height 0. The left quarter of the image is masked out.
        height, width = 120, 160
        depth = np.full((height, width), 1.0, dtype=np.float32)
        active = np.ones((height, width), dtype=np.uint8)
        active[:, :40] = 0
        k = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = (0.3, 0.0, 1.0)
        began = time.perf_counter()
        done = client.request(
            {
                "action": "integrate_masked_depth",
                "depth": depth,
                "K": k,
                "T_base_cam": pose,
                "active_mask": active,
            }
        )
        integrated = time.perf_counter() - began
        assert done["masked_free_updates"] > 0 and done["ms"] > 0

        began = time.perf_counter()
        found = client.request(
            {
                "action": "query",
                "region_min": np.asarray(low, dtype=np.float32),
                "region_max": np.asarray(high, dtype=np.float32),
            }
        )
        queried = time.perf_counter() - began
        print(
            f"first masked depth image in {integrated:.2f} s, first query in "
            f"{queried:.2f} s (both compile Warp kernels)"
        )
        grid = np.asarray(found["grid"])
        points = np.asarray(found["points"], dtype=np.float32).reshape(-1, 3)
        assert grid.ndim == 3 and not np.isnan(grid).any()
        assert float(found["voxel"]) == pytest.approx(probe["voxel"])
        # The floor is what is occupied: a thin layer at height 0, and nothing
        # under the masked quarter of the image (x below -0.1 m at the floor).
        assert len(points) > 100
        assert -0.05 <= points[:, 2].min() and points[:, 2].max() <= 0.005
        assert points[:, 0].min() >= -0.13 and points[:, 0].max() > 0.5
        assert float(grid.min()) == 0.0
    finally:
        client.close()
        bridge.close()
    # Ended by a signal, as the launcher ends it; no process is left.
    assert process.returncode in (-signal.SIGTERM, -signal.SIGKILL)
    log = (tmp_path / "occupancy-bridge.log").read_text()
    assert "APAI_SERVER_READY " in log
    # The experiment's bridge wrote nothing into APAI's checkout or its own folder.
    after = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    assert after == before


@apai
def test_ffs_depth_of_a_recorded_stereo_pair(experiment_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")
    import test_agent_end_to_end as end_to_end
    import test_stereo_camera as stereo
    from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera

    folder = recorded_folder(experiment_dir)
    before = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    caches = [Path(os.environ[name]) for name in CACHE_VARIABLES]
    cached = sum(len(end_to_end.tree_state(cache)) for cache in caches)
    images = [
        cv2.imread(str(folder / name)) for name in ("left_raw.jpg", "right_raw.jpg")
    ]
    assert all(image is not None for image in images)
    # The options the full stack hands over (config.py, _merge_experiment).
    profile = yaml.safe_load((experiment_dir / "configs/sensing_thor.yaml").read_text())
    assert profile["depth_backend"] == "ffs"

    began = time.perf_counter()
    depth = stereo_camera.experiment_depth(
        experiment_dir / stereo.STEREO_CALIBRATION, "ffs", profile["ffs"]
    )
    try:
        print(f"FFS loaded and warmed up in {time.perf_counter() - began:.1f} s")
        matcher = depth.matcher
        assert depth.backend == "ffs" and matcher.metadata["backend"] == "ffs"
        assert next(matcher.model.parameters()).is_cuda
        for count in range(3):
            began = time.perf_counter()
            rectified, depth_m = depth.compute(images)
            took_ms = (time.perf_counter() - began) * 1000
            gpu_ms = {
                key: round(value, 1) for key, value in matcher.last_timings_ms.items()
            }
            print(f"stereo pair {count}: {took_ms:.0f} ms ({gpu_ms})")
    finally:
        depth.close()

    width, height = depth.size
    assert rectified.shape == (height, width, 3) and rectified.dtype == np.uint8
    assert depth_m.shape == (height, width) and depth_m.dtype == np.float32
    assert np.isfinite(depth_m).all()
    valid = depth_m[depth_m > 0]  # 0 marks a pixel without depth
    print(f"valid depth: {valid.size / depth_m.size:.1%} of the pixels")
    assert valid.size / depth_m.size > 0.2
    assert 0.2 <= valid.min() and valid.max() <= 8.0
    assert 0.3 < float(np.median(valid)) < 3.0  # a table scene in front of the arm
    # Nothing was written into APAI's checkout or the experiment folder.
    after = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    assert after == before
    now_cached = sum(len(end_to_end.tree_state(cache)) for cache in caches)
    print(f"files in the cache folders: {cached} before, {now_cached} after")


@apai
def test_the_segmenter_loads_on_cuda_0(
    experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    yaml = pytest.importorskip("yaml")
    import test_agent_end_to_end as end_to_end
    from apai.perception.segmenter import BoxSegmenter
    from apai.types import Frame

    before = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    # The model and the device the full stack's detector takes (thor_cloud.py).
    profile = yaml.safe_load((experiment_dir / "configs/thor_cloud.yaml").read_text())
    model = experiment_dir / profile["segmenter"]["model"]
    if not model.is_file():
        pytest.skip(f"the segmentation model {model} is not there")
    # As run_agent.sh leaves it, and as the experiment's launcher runs: APAI's
    # strict CUDA mode is not set. The model runs on the GPU and a mask comes
    # back as a numpy array, which APAI's grasp code takes.
    monkeypatch.delenv("APAI_REQUIRE_CUDA", raising=False)
    image = cv2.imread(str(recorded_folder(experiment_dir) / "left_rectified.png"))
    assert image is not None
    height, width = image.shape[:2]
    k = np.array([[width, 0.0, width / 2], [0.0, width, height / 2], [0.0, 0.0, 1.0]])
    frame = Frame(rgb=image, depth_m=None, K=k)
    box = np.array([[width / 4, height / 4, 3 * width / 4, 3 * height / 4]], np.float32)

    segmenter = BoxSegmenter(str(model), profile["segmenter"]["device"])
    assert segmenter.device == "cuda:0"
    began = time.perf_counter()
    (mask,) = segmenter.masks(frame, box)
    loaded = time.perf_counter() - began
    began = time.perf_counter()
    (again,) = segmenter.masks(frame, box)
    print(
        f"SAM loaded and segmented one box in {loaded:.2f} s, then in "
        f"{time.perf_counter() - began:.3f} s"
    )
    assert mask is not None and again is not None
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == (height, width)
    covered = int(mask.sum())
    print(f"the mask covers {covered} of {height * width} pixels")
    assert 0 < covered < height * width
    # Nothing was written into APAI's checkout or the experiment folder.
    after = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    assert after == before


def no_model() -> Any:
    """In place of a model client: it counts what it is asked and asks nobody.

    A request for detections is answered with none; anything else is refused.
    """
    llm = importlib.import_module("apai.agent.llm")

    class NoModel(llm.LLMClient):
        model = "no-model"
        supports_vision = False

        def __init__(self) -> None:
            self.calls = 0

        def chat(
            self,
            system: str,
            messages: list[dict],
            tools: list[dict] | None = None,
            max_tokens: int = 1024,
            **_transport: Any,
        ) -> Any:
            self.calls += 1
            if [tool["name"] for tool in tools or []] == ["report_detections"]:
                report = llm.ToolCall("report_detections", {"objects": []})
                return llm.LLMResponse(tool_calls=[report])
            raise RuntimeError("this test asks no model")

    return NoModel()


@pytest.fixture
def full_scene(make_realtime_apai: Callable[..., Any], tmp_path: Path) -> Iterator[Any]:
    """The robot app on APAI's mock arm (driver apai-mock), and one launcher run."""
    import test_agent_end_to_end as end_to_end
    import test_launcher

    harness = make_realtime_apai(
        config=test_launcher.scene_robot_config(), connected=False
    )
    end_to_end.raise_arm(harness)
    with apai_runtime_env(tmp_path) as folder:
        yield end_to_end.Scene(harness, folder)


@apai
def test_the_full_stack_builds_and_the_grid_goes_live_on_recorded_pairs(
    full_scene: Any, experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("ultralytics")
    yaml = pytest.importorskip("yaml")
    import test_agent_end_to_end as end_to_end
    import test_stereo_camera as stereo
    from isaaccapture_examples.agentic_remote.agent import launcher

    scene = full_scene
    folder = recorded_folder(experiment_dir)
    demo = importlib.import_module("apai.apps.demo")
    occupancy = importlib.import_module("apai.perception.occupancy")
    thor_cloud = import_experiment("thor_cloud")
    # No model call and no login: both model clients of the full stack (the
    # agent's and the detector's) are stand-ins.
    agent_model, detector_model = no_model(), no_model()
    monkeypatch.setattr(demo, "make_llm", lambda cfg: agent_model)
    monkeypatch.setattr(thor_cloud, "make_client", lambda config: detector_model)
    # As run_agent.sh leaves it: APAI's strict CUDA mode is not set.
    monkeypatch.delenv("APAI_REQUIRE_CUDA", raising=False)
    # The real obstacle grid client: the experiment's bridge answers it.
    scene.grid = occupancy.OccupancyClient
    scene.typed = []  # --interactive, and the task prompt ends at once

    before = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    began = time.monotonic()
    code = scene.run(
        "--interactive",
        "--camera",
        f"recorded:{folder}",
        camera={
            "calibration_dir": str(experiment_dir / stereo.STEREO_CALIBRATION),
            "hand_eye_json": str(experiment_dir / stereo.HAND_EYE),
            "depth_backend": "ffs",
        },
        experiment={"stack": "full"},
        task={"grid_wait_s": GRID_WAIT_S},
    )
    print(f"the full stack ran for {time.monotonic() - began:.1f} s")
    print("\n".join(scene.lines))
    assert code == launcher.EXIT_OK

    # The obstacle grid went live on camera frames of the recorded pair, in
    # the experiment's bridge on GPU 0.
    assert scene.said("obstacle grid: live")
    grid = scene.trace.built["grid"]
    assert grid.refreshes >= 1 and grid.last_error is None
    status = grid.status
    assert status["backend"] == "warp" and status["device"] == "cuda:0"
    assert status["target_retirement"] is True and status["pid"] != os.getpid()
    assert "APAI_SERVER_READY " in (scene.run_dir / "occupancy-bridge.log").read_text()
    with pytest.raises(ProcessLookupError):  # the bridge was ended
        os.kill(status["pid"], 0)
    profile = yaml.safe_load((scene.run_dir / "camera_profile.yaml").read_text())
    assert profile["type"] == "link_stereo" and profile["depth_backend"] == "ffs"
    assert profile["recorded_dir"] == str(folder)
    used = json.loads((scene.run_dir / "experiment-files.json").read_text())
    assert {
        "thor_calibration_rig.py",
        "thor_camera.py",
        "thor_stereo_ffs.py",
        "thor_cloud.py",
        "thor_self_mask.py",
    } <= set(used)

    # No model was asked for a task, and no arm was taken.
    print(f"detection requests answered by the stand-in: {detector_model.calls}")
    assert agent_model.calls == 0
    assert scene.order == ["close:nobody"] and scene.asked == []
    assert scene.h.heard("owner") == [] and scene.h.spy.targets == []
    # Nothing was written into APAI's checkout or the experiment folder.
    after = [end_to_end.tree_state(tree) for tree in end_to_end.read_only_trees()]
    assert after == before
