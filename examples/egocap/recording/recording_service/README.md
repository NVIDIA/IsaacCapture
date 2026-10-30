# recording_service

The Ego Capture UI **recording service**: submitted samples in, checked and finalized episodes out.
It is independent of acquisition, hardware and network transport; it never learns where a
sample came from, and a sample read from an existing dataset takes exactly the same path as
any other.

## `recording_service.RecordingService`

A caller supplies numeric samples and decoded camera images. The service validates and copies
each into one bounded queue, one worker per episode writes them through the real Zarr/MP4
backend, and `stop_episode` drains the queue, seals the files, checks what is actually on
disk, and finalizes (or rejects) the take.

```python
service = RecordingService(ServiceConfig(output_root, numeric_sources, video_sources, QcConfig(...)))
await service.open_session()
episode = await service.start_episode("task text", "operator")
service.submit_sample(episode.episode_id, sample)  # ACCEPTED or FULL; invalid input raises
result = await service.stop_episode(episode.episode_id)  # KEPT / REJECTED / RECOVERY_REQUIRED
await service.close()
```

### What "checked" means

Two separate checks run on the sealed episode, and their results are never mixed.

**Storage integrity (always).** The staged Zarr/MP4 output must be readable and match the declared schema
(arrays, dtypes, shapes), every required source must have rows, accepted counts must equal written counts,
capture timestamps must be finite and non-decreasing, numeric values must be finite, and each MP4 must decode to
exactly as many frames as it has timestamps. A failed check rejects the take and keeps its data. A staged dataset
that cannot be opened at all is an `ERROR`, which retains the take for recovery or discard instead.

**Capture quality (optional, `QcConfig.capture`).** Run only on integrity-valid data, from the recorded
timestamps and values. Which streams each check reads is declared, and an unknown source or non-float field is a
`ConfigurationError` at `open_session`, never a silent skip:

```python
QcConfig(
    required_sources=("pico_controller", "head_human_camera"),
    capture=CaptureQcConfig(
        camera_expected_fps=(
            ("head_human_camera", 60.0),
        ),  # camera -> expected rate (Hz); None = the source's own fps
        pose_streams=(
            ("pico_controller", "head_pose"),
        ),  # (numeric source, float field) to watch; must be required
        # defaults: camera_min_fps_ratio=0.86, freeze_seconds=1.0, freeze_eps=1e-6,
        # flat_seconds=2.0, flat_eps=1e-3
    ),
)
```

| Check | Decision |
|---|---|
| `camera_min_fps:<camera>` (gate): average rate `(frames-1)/duration` below `ratio x expected` (fewer than two frames counts as 0 fps) | `REJECTED_CAMERA`: the take is rejected and only its own data is removed |
| `camera_frame_parity` (opt-in `camera_parity`): the listed cameras' frame counts differ by more than `camera_parity_max_diff` | `REJECTED_CAMERA` |
| `sustained_freeze:<source>/<field>`: no change above `freeze_eps` for `freeze_seconds` | `FLAGGED_BAD`: the episode is kept, with `bad_episode = True` and the native `freeze_check_passed = False` (with `freeze_check_issues`) in its metadata |
| `avatar_hand_liveness:<source>/<side>` (opt-in `avatar_hand_sources`): a hand with fewer than `avatar_min_states` rows, no recorded stream, or only streams frozen for the whole episode (a dead or disconnected glove) | `FLAGGED_BAD` |
| `pico_body_joints:<source>` (opt-in `pico_body`): wrong shape, no frames, non-finite values, or more invalid / all-identity frames than allowed | `FLAGGED_BAD` |
| `flat_motion`, `dead_stream` (a pose stream frozen the whole episode, noted as sensor dropout when pinned at identity), `frame_drops:<camera>` (more than `max_dropped_frames` in any `drop_window_s`) | advisory: warnings only; `frame_drops` is also saved as the native `frame_drops` metadata |

Each device check runs only for the sources you declare, and the declaration is validated against the recorded
schema before recording (an Avatar source without hand fields, a Pico body field that is not a float `(24, 4, 4)`
array, or a parity camera that is not a video source is a `ConfigurationError`). Declare Avatar and Pico checks only
for stations whose profile records those fields.

A measured camera failure outranks bad tracking (any tracking gate); known bad tracking outranks a check that could not
run. Capture QC not configured is `SKIPPED` (nothing was measured). A check that could not run is `ERROR`: the take is
retained for recovery (still SEALED, `RECOVERY_REQUIRED`) so a recovery re-runs the checks or the operator discards it;
it is never kept as unflagged data, and nothing is ever reported as passed without being measured. Every camera and
stream is measured separately, so one read error cannot erase another's
measured failure. `EpisodeResult.qc_decision` and `capture_quality` carry the decision and per-check results (with
the measured values); kept episodes store them as `capture_qc_*` metadata, with `domain_checks` set to `EXECUTED` or
`SKIPPED`. `freeze_check_passed` is written only when a freeze check ran (`True` if it passed, `False` if it failed).
`bad_episode` is a local classification only: **quarantine routing is not supported** and a bad episode is finalized
like any other.

The configuration a pending take must match depends on how far it got: a SEALED take is still to be judged, so the whole
configuration (including the capture policy and `ALGORITHM_VERSION`) must match; a CHECKED take already carries its
decision and the policy that made it, so only the layout-affecting parts (sources, required sources, latency,
calibration) must match, and a new capture policy or algorithm cannot strand it.

The decision is made durable (with the exact policy and algorithm version, frozen when the take starts) before any
cleanup, together with one *finalization intent* for the checked take: `KEEP` publishes it, `REMOVE` deletes its own data
(a camera rejection), `RETAIN` leaves its data for explicit resolution (an integrity failure). Recovery follows the
recorded intent and never re-derives it from policy. If removing a camera-rejected take's data fails or is interrupted, the take stays listed for
`recover_episode`, which finishes the removal (the ownership stamp is deleted last so it stays retryable). Changing the
capture policy or algorithm makes an in-flight take incompatible rather than re-judging it silently. Manifests written
before capture QC existed (schema 1) still recover and read as "not run".

### Local routing (optional, `ServiceConfig.routing`)

Without `routing` kept episodes stay in the session (`<output_root>/<session>/episode_N`). With it they are
published to `<output_root>/<Studio>/<timestamp>/episode_<k>`, on the same filesystem as staging
(one atomic rename; no copy engine):

```python
ServiceConfig(
    ...,
    routing=RoutingConfig(
        session_studio="",  # set to force one studio for every take
        studio_by_task=(("stack blocks", "Kitchen"), ("set the table", "Dining Room")),
        known_studios=("Kitchen", "Dining_Room"),  # canonical spellings
        unmatched_studio="_unmatched",
    ),
)
```

- The studio is resolved when the take **starts** (session override, else the task-to-studio map, else the unmatched
  bucket) and stored in its manifest, so a later task change cannot reroute a take waiting for QC. Mode and theme
  conventions (Diverse, Freeform, Mid Train, ...) are the application's job: it resolves them into these inputs.
- A *group* (`<Studio>/<timestamp>`) is created once per (studio, task, calendar day) for a session and reused; its
  `zarr.json` carries an ownership stamp and dataset-level attributes. Its routes are recorded in
  `<session>/.recording/routes.json` beside the manifests.
- The output slot `k` is assigned only when a take is **kept**, and frozen durably before publication. A rejected or
  discarded attempt takes no slot; a kept episode that capture QC flagged bad does (it is real output).
  `EpisodeResult.episode_index` is the attempt identity (it never skips); `output_index` is the slot.
- Group attributes (`num_episodes`, `total_data_time_hours`, `episode_lengths`, `discarded_episode_indices` and
  `discarded_reasons`) are **recomputed from the finalized episodes** after each finalization, never incremented, so a
  retry or crash cannot double count. "Saved" counts flagged-bad episodes; the bad record lists them with their reasons.
- An existing directory at the assigned slot, or a group not stamped as this session's, is a collision: the take is
  retained untouched and nothing is merged or overwritten. Recovery finds a take that was renamed into its group before
  its checkpoint was saved. Routed paths are validated (exactly `<Studio>/<group>/episode_<k>`, no symlinks or
  traversal) before any rename or delete, and routed finalized output is never deleted by discard.
- Whether routing is configured is part of the configuration digest: a take recorded without routing is not
  recoverable under a routed configuration (and vice versa), because its destination was never decided. The studio
  mapping itself is not part of it: each take's studio is frozen in its manifest when it starts, so changing the mapping
  later does not strand a pending take.
- The configuration digest also covers `calibration_assets` (the set of station files this deployment copies in) --
  **enabling or changing that set changes the digest for every new session going forward.** A SEALED/CHECKED take left
  pending from before such a deployment change will fail its config-digest check and `recover_episode` will refuse it
  outright (`StateError: ... recorded with a different configuration`); there is no override, only `discard_pending`.
  Resolve every pending take under the prior deployment before rolling out a calibration_assets change -- this is
  explicit deployment sequencing the operator performs; the service leaves existing manifests exactly as written
  rather than rewriting them to paper over a configuration change.
- `config.yaml` (station identity only -- `{"meta": {"egocap_station_name": ...}}`, NOT a full source
  description) and the session's preserved calibration-asset files are copied into each routed group's root the
  first time that group exists (from the session's own preserved copies, never re-read from live settings); see
  `_seed_group_assets` in `recording_service.py`. Because this snapshot carries no `video_sources`/`numeric_sources`/
  `recording` block, the data-loading pipeline's source-config/alignment reader (see `../README.md`'s data layout
  and format section) cannot run against a session's or routed group's `config.yaml`; that reader needs its source
  configuration supplied separately, never read from the recorded output.

No `data_check_config.yaml` is written in each group directory, and there is no separate
timestamp-consistency, time-range overlap, duration or resolution advisory check (the integrity
check already verifies each video's frame count). Group totals are recomputed fresh whenever a
take in the group is finalized or recovered, not reconciled against a prior count when a session
is reopened. An integrity-rejected take keeps its data and reports its path in its result; it is
not listed in the pending list. Samples the caller never submitted, such as ones dropped before
admission, are invisible to every check. The service writes only to the station's own local
storage.

Behavior worth knowing:

- `submit_sample` never waits. `FULL` means the queue is at `queue_capacity` and nothing was consumed: wait and
  retry the same sample. The queue counts samples, not bytes, so memory is bounded by `queue_capacity` times the
  largest sample (512 x a 1280x800 RGB frame is about 1.5 GB). Overflow cannot corrupt a recording or the
  service, but a caller that ignores `FULL` silently loses that sample and the integrity checks cannot see the
  gap (accepted still equals written); live use at high rates needs a byte or per-source budget first.
- A failed integrity check rejects the take and keeps its data; nothing is finalized on a failed check. A
  camera-gate rejection removes the take's own data instead (see above).
- Cancelling a waiting caller never cancels the service's work. Repeating `stop_episode`
  joins the same stop. `close()` waits against one deadline and reports release truthfully:
  an encoder whose release is unconfirmed keeps the session locked and the data in place.
- Readable pending episodes block a new take until resolved with `recover_episode` (sealed or checked
  episodes only) or `discard_pending`. Some states cannot be resolved by any call and need a manual fix
  before the session records again (the service exposes no repair API for these cases):
  - a corrupt, unsupported or non-canonical manifest (bad JSON, unknown `schema_version` or
    `storage_phase`, paths that are not the canonical `episode_N`/`episode_N.pending`): it is listed in the
    snapshot issues, not as a pending episode, and blocks a new take. Repair the manifest by hand, or move
    it out of `.recording/` once the episode's data has been dealt with;
  - a sealed or checked pending directory whose identity stamp (`recording_episode_id` in its root
    metadata) is missing or belongs to another episode: recovery and discard both refuse it, so inspect the
    directory and remove or re-stamp it by hand. The one exception is an empty directory left by an interrupted
    removal of a camera-rejected take (stamp deleted last, durably recorded in the manifest): that is finished by
    recovery or discard;
  - a missing or changed session copy of a required calibration asset: `open_session` refuses the session.
    Restore the copy from the original file (its hash is recorded in `.recording/calibration_assets.json`).
- Manifest checkpoints are published in memory only after they are durably written. If the SEALED checkpoint
  cannot be saved, the take fails with a clear error and is not recoverable: discard it with `discard_pending`
  and record again. For CHECKED, FINALIZED and REJECTED a failed write leaves the last durable
  phase and `recover_episode` retries it, in the same process or after a restart. For DISCARDED, the retry
  is `discard_pending`.
- `close()` during RECORDING discards the active take; a stop that was already accepted is allowed to finish.
- The service takes a detached snapshot of its configuration at construction; later changes to the caller's
  config objects do not affect it.
- Quarantine routing is not supported; requesting it (`ServiceConfig.quarantine`) is a configuration error.

| Module | Contents |
|---|---|
| `models.py` | Configuration, sample, result and error types (all recording-owned frozen dataclasses; arrays and dicts inside them are not deep-frozen). |
| `recording_service.py` | The service. |

Storage, quality checking, routing and dataset reading live in `egocap/recording/`
(`episode_store.py`, `episode_layout.py`, `dataset_check.py`, `dataset_reader.py`, `capture_qc.py`, `capture_measure.py`,
`route.py`). The service imports nothing from the capture side; that is enforced by
`tests/recording_service/test_dependency_boundary.py`.

## Tests and acceptance

`tests/recording_service/` runs the public API against a real recorded episode
(`EGOCAP_REAL_EPISODE=<episode dir>`) with the real backend; without it those tests are
SKIPPED. `recording/tools/replay_episode.py` is the full acceptance run: it replays an
episode through the service and independently verifies the output against a SHA-256 manifest
of the input. These tests are not wired into CI; run them locally.
