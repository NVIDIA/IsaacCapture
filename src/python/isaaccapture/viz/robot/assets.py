# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenes each preview arm and its ghost gripper are drawn from, one per arm.

The MJCF wrappers are tracked package data; the upstream mesh and MJCF they name is
fetched, not vendored. Each ``ensure_*_scene`` assembles both halves into its own cache
directory and returns the scene path, gripper the ghost draws included.

Downloads are verified against a pinned commit: a raw.githubusercontent path is not
immutable in practice, and a substituted mesh renders as a broken arm rather than an error.
Everything lands flat, because MuJoCo drops an included file's own ``meshdir``; where a
ghost and its follower share a mesh, one of the two is named apart so the flat directory and
MuJoCo's global asset names can both hold them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import urllib.request
from pathlib import Path

LOG = logging.getLogger(__name__)

#: Bump this and the checksums together, or the download is refused.
SO_ARM_REPO = "TheRobotStudio/SO-ARM100"
SO_ARM_COMMIT = "fda892cba81032c46c40976a48c9ceadbf40a9ca"

#: The SO-101 leader gripper, as ``(upstream path, destination name, sha256)``: the ghost
#: its own scene draws, and no other's -- the reBot has no leader, so it puts its own
#: follower gripper on the hand from meshes it already fetches. The destination is per entry
#: because ``sts3215_03a_v1.stl`` is fetched twice (the leader fragment names its copy
#: ``STS3215_03a.stl``) and a flat directory cannot alias the two.
#:
#: ``so101_new_calib.urdf`` is not drawn; it is on disk to check :mod:`.so101_ghost`'s
#: trigger hinge and travel against.
GHOST_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        "STL/SO101/Individual/Wrist_Roll_SO101.stl",
        "Wrist_Roll_SO101.stl",
        "de3a65044dd4ae8bcb9659d8ca2b49598e3f5571edf89f45ad975e9776a7ffee",
    ),
    (
        "STL/SO101/Individual/Trigger_SO101.stl",
        "Trigger_SO101.stl",
        "48ecec3a3710cffdc0ae96d28547e49ddf4cbc93ccd915be7549f78e00ad2850",
    ),
    (
        "STL/SO101/Individual/Handle_SO101.stl",
        "Handle_SO101.stl",
        "fb8757bdff009c04c207481dd664813ccdac2ad989acea6057df780b52327281",
    ),
    (
        "Simulation/SO101/assets/sts3215_03a_v1.stl",
        "STS3215_03a.stl",
        "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    ),
    (
        "Simulation/SO101/so101_new_calib.urdf",
        "so101_new_calib.urdf",
        "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c",
    ),
    (
        "LICENSE",
        "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
)

#: The SO-101 follower arm. ``joints_properties.xml`` is deliberately absent: upstream
#: inlines its ``<default>`` block, so the file is never read.
SO101_ARM_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        "Simulation/SO101/so101_new_calib.xml",
        "so101_new_calib.xml",
        "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580",
    ),
    (
        "Simulation/SO101/assets/base_motor_holder_so101_v1.stl",
        "base_motor_holder_so101_v1.stl",
        "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424",
    ),
    (
        "Simulation/SO101/assets/base_so101_v2.stl",
        "base_so101_v2.stl",
        "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
    ),
    (
        "Simulation/SO101/assets/motor_holder_so101_base_v1.stl",
        "motor_holder_so101_base_v1.stl",
        "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b",
    ),
    (
        "Simulation/SO101/assets/motor_holder_so101_wrist_v1.stl",
        "motor_holder_so101_wrist_v1.stl",
        "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011",
    ),
    (
        "Simulation/SO101/assets/moving_jaw_so101_v1.stl",
        "moving_jaw_so101_v1.stl",
        "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d",
    ),
    (
        "Simulation/SO101/assets/rotation_pitch_so101_v1.stl",
        "rotation_pitch_so101_v1.stl",
        "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34",
    ),
    (
        "Simulation/SO101/assets/sts3215_03a_no_horn_v1.stl",
        "sts3215_03a_no_horn_v1.stl",
        "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887",
    ),
    (
        "Simulation/SO101/assets/sts3215_03a_v1.stl",
        "sts3215_03a_v1.stl",
        "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    ),
    (
        "Simulation/SO101/assets/under_arm_so101_v1.stl",
        "under_arm_so101_v1.stl",
        "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71",
    ),
    (
        "Simulation/SO101/assets/upper_arm_so101_v1.stl",
        "upper_arm_so101_v1.stl",
        "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f",
    ),
    (
        "Simulation/SO101/assets/waveshare_mounting_plate_so101_v2.stl",
        "waveshare_mounting_plate_so101_v2.stl",
        "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9",
    ),
    (
        "Simulation/SO101/assets/wrist_roll_follower_so101_v1.stl",
        "wrist_roll_follower_so101_v1.stl",
        "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4",
    ),
    (
        "Simulation/SO101/assets/wrist_roll_pitch_so101_v2.stl",
        "wrist_roll_pitch_so101_v2.stl",
        "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae",
    ),
)

SO_ARM_ASSETS: tuple[tuple[str, str, str], ...] = GHOST_ASSETS + SO101_ARM_ASSETS

#: The tracked wrappers. Re-copied on every call rather than gated on the completeness
#: marker, so editing one takes effect on the next run with no cache to clear.
SCENE_FILE = "scene.xml"
_WRAPPERS = (SCENE_FILE, "follower_arm.xml", "leader_gripper.xml")

#: The reBot DevArm, RobStride build, from MuJoCo Menagerie. Upstream derives it from the
#: same Seeed URDF LeRobot's IK solves against and validates against it link by link, and
#: builds it around RS-06/RS-00 actuators -- so it is the RS arm, not the Damiao one, whose
#: geometry differs.
MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
REBOT_MODEL_DIR = "seeed_rebot_devarm"
REBOT_SCENE_FILE = "scene_rebot.xml"
_REBOT_WRAPPERS = (REBOT_SCENE_FILE, "rebot_arm.xml", "rebot_gripper.xml")

#: sha256 over the sorted ``"<name> <sha256>\n"`` lines of everything fetched from
#: Menagerie -- 117 files, so one constant rather than a table nobody reads. Content, not
#: archive framing: a repo tarball is 400 MB for 15 MB of arm, and its bytes are not
#: promised to be stable. Bump it and MENAGERIE_COMMIT together, or the download is
#: refused.
REBOT_MANIFEST_SHA256 = (
    "5164271c8fc098add8c1905e52198c759239dda73831231112d70359a78874ba"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_dir(name: str) -> Path:
    root = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    dest = Path(root) / "isaaccapture" / name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _copy_wrappers(dest: Path, wrappers: tuple[str, ...]) -> None:
    source = Path(__file__).parent / "assets"
    for wrapper in wrappers:
        shutil.copyfile(source / wrapper, dest / wrapper)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310
        return response.read()


def _fetch_checksummed(dest: Path, entries: tuple[tuple[str, str, str], ...]) -> None:
    """Fetch ``(remote, name, sha256)`` entries from SO-ARM100 that are not already
    cached under their own hash."""
    for remote, name, sha in entries:
        target = dest / name
        if target.is_file() and _sha256(target) == sha:
            continue
        payload = _fetch(
            f"https://raw.githubusercontent.com/{SO_ARM_REPO}/{SO_ARM_COMMIT}/{remote}"
        )
        if hashlib.sha256(payload).hexdigest() != sha:
            raise RuntimeError(
                f"robot twin: checksum mismatch for {remote}. Upstream changed, or "
                "SO_ARM_COMMIT and the hashes in SO_ARM_ASSETS disagree."
            )
        target.write_bytes(payload)


def ensure_so101_scene() -> Path:
    """Assemble the scene into the cache directory and return its path.

    The completeness marker gates the download only: the MJCF's existence is not a
    completeness signal, since an interrupted first run leaves the meshes it names missing.
    Re-running repairs a partial cache; delete the directory to force a re-download.

    Raises:
        RuntimeError: If a download's checksum does not match :data:`SO_ARM_ASSETS`.
        OSError: If the files cannot be fetched or written.
    """
    dest = _cache_dir("so101-assets")
    _copy_wrappers(dest, _WRAPPERS)

    marker = dest / ".fetch_complete"
    if not marker.exists():
        _fetch_checksummed(dest, SO_ARM_ASSETS)
        marker.touch()

    # Absolute: on mujoco 3.11 a relative model path mis-composes an <include>d file's
    # path and fails naming a file that exists.
    return (dest / SCENE_FILE).resolve()


def _menagerie_rebot_files() -> tuple[str, ...]:
    """The model file, its licence, and every mesh it names. Read out of the MJCF rather
    than listed, so a mesh added upstream cannot be silently left behind -- the manifest
    digest is what pins the set."""
    model = _fetch(
        f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_COMMIT}/"
        f"{REBOT_MODEL_DIR}/{REBOT_MODEL_DIR}.xml"
    ).decode()
    meshes = sorted(set(re.findall(r'file="([^"]+)"', model)))
    return (f"{REBOT_MODEL_DIR}.xml", "LICENSE", *meshes)


def _rebot_cached_digest(dest: Path) -> str | None:
    """The manifest digest of what is already in ``dest``, or ``None`` if it cannot be read.

    Read only local files so cached assets can be validated offline. Tracked wrappers
    are package data, not fetched assets, and are excluded from the manifest.
    """
    wrappers = set(_REBOT_WRAPPERS)
    lines = []
    try:
        for path in sorted(dest.iterdir()):
            if not path.is_file() or path.name.startswith(".") or path.name in wrappers:
                continue
            lines.append(f"{path.name} {_sha256(path)}\n")
    except OSError:
        return None
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def ensure_rebot_devarm_rs_scene() -> Path:
    """Assemble the reBot DevArm (RobStride) scene into its cache directory and return its path.

    Same completeness-marker rule as :func:`ensure_so101_scene`. Nothing from SO-ARM100 is
    fetched: this arm's ghost is its own gripper, drawn from the meshes below.

    Raises:
        RuntimeError: If the fetched set does not hash to :data:`REBOT_MANIFEST_SHA256`.
        OSError: If the files cannot be fetched or written.
    """
    dest = _cache_dir("rebot-devarm-rs-assets")
    _copy_wrappers(dest, _REBOT_WRAPPERS)

    marker = dest / ".fetch_complete"
    if marker.exists() and _rebot_cached_digest(dest) != REBOT_MANIFEST_SHA256:
        # The marker claims a verified fetch; the bytes on disk say otherwise. Re-fetch
        # rather than compile them -- a truncated or substituted mesh draws as a broken arm
        # instead of raising, which is the failure the digest exists to catch.
        LOG.warning(
            "Robot twin: the cached reBot assets in %s do not match "
            "REBOT_MANIFEST_SHA256; re-fetching them.",
            dest,
        )
        marker.unlink(missing_ok=True)
    if not marker.exists():
        base = (
            f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_COMMIT}/"
            f"{REBOT_MODEL_DIR}"
        )
        manifest = []
        for name in _menagerie_rebot_files():
            # Everything lands flat: MuJoCo drops an included file's own meshdir, so
            # upstream's meshdir="assets" is inert once rebot_arm.xml includes it.
            flat = Path(name).name
            remote = (
                f"{base}/assets/{flat}"
                if flat.lower().endswith(".stl")
                else f"{base}/{flat}"
            )
            payload = _fetch(remote)
            (dest / flat).write_bytes(payload)
            manifest.append(f"{flat} {hashlib.sha256(payload).hexdigest()}\n")
        digest = hashlib.sha256("".join(sorted(manifest)).encode()).hexdigest()
        if digest != REBOT_MANIFEST_SHA256:
            raise RuntimeError(
                f"robot twin: reBot manifest is {digest}, expected "
                f"{REBOT_MANIFEST_SHA256}. Upstream changed, or MENAGERIE_COMMIT and "
                "REBOT_MANIFEST_SHA256 disagree."
            )
        marker.touch()

    return (dest / REBOT_SCENE_FILE).resolve()
