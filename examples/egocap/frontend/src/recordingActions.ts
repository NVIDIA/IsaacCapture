import type { useCaptureSession } from "./useCaptureSession";
import type { RecordingActions, RecordingBatchIdentity, RecordingEpisodeIdentity } from "./recordingTypes";
import { batchTargetPayload, episodeTargetPayload, setTaskPayload } from "./recordingWire";

type SendRecordingCommand = ReturnType<typeof useCaptureSession>["sendRecordingCommand"];

/** Builds the recording panel's action callbacks from the one real
 * transport connection -- every action sends a real command to the
 * backend's recording-command handlers and resolves with their real
 * accept/reject outcome, never a local fake. Wire payloads are built
 * exclusively through recordingWire.ts so the flat, snake_case shape the
 * backend's validator actually requires is never hand-rolled at a call
 * site. There is no `onOpenSession`/`onCloseSession` here -- this panel
 * never opens/closes anything, it only rotates batches via `onNewBatch`. */
export function buildRecordingActions(sendRecordingCommand: SendRecordingCommand): RecordingActions {
  return {
    onSetTask: (target: RecordingBatchIdentity, taskText: string) => sendRecordingCommand("set_task", setTaskPayload(target, taskText)),
    onStartEpisode: (target: RecordingBatchIdentity) => sendRecordingCommand("start_episode", batchTargetPayload(target)),
    onStopEpisode: (target: RecordingEpisodeIdentity) => sendRecordingCommand("stop_episode", episodeTargetPayload(target)),
    onDiscardEpisode: (target: RecordingEpisodeIdentity) => sendRecordingCommand("discard_episode", episodeTargetPayload(target)),
    onNewBatch: (target: RecordingBatchIdentity) => sendRecordingCommand("new_batch", batchTargetPayload(target)),
  };
}
