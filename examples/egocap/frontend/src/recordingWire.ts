// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Converts the frontend's typed recording values into the exact flat,
 * snake_case payload shapes backend/egocap_web/recording_command.py's
 * validators require. This is the one place that boundary is crossed --
 * RecordingActions callers never build a wire payload by hand. There is no
 * `open_session`/`close_session` payload builder here -- this component
 * never sends those actions, only `new_batch`.
 */
import type { RecordingBatchIdentity, RecordingEpisodeIdentity } from "./recordingTypes";

/** Wire payload for `set_task`: a flat batch id and task text, not a nested `target`/task-context object. */
export function setTaskPayload(target: RecordingBatchIdentity, taskText: string): Record<string, unknown> {
  return { batch_id: target.batchId, task: taskText };
}

/** Wire payload for batch-only actions (`start_episode`/`new_batch`). */
export function batchTargetPayload(target: RecordingBatchIdentity): Record<string, unknown> {
  return { batch_id: target.batchId };
}

/** Wire payload for episode-targeted actions (`stop_episode`/`discard_episode`). */
export function episodeTargetPayload(target: RecordingEpisodeIdentity): Record<string, unknown> {
  return { batch_id: target.batchId, episode_id: target.episodeId };
}
