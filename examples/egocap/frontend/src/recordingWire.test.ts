/**
 * Pins recordingWire.ts's output to the exact flat, snake_case shape
 * backend/egocap_web/recording_command.py's validators require -- this is
 * the one place that boundary is crossed, so a drift here is a real
 * connection-breaking bug, not a cosmetic mismatch. No open_session/
 * close_session case here -- this component never sends those actions.
 */
import { describe, expect, it } from "vitest";
import { batchTargetPayload, episodeTargetPayload, setTaskPayload } from "./recordingWire";

describe("recordingWire", () => {
  it("setTaskPayload is a flat batch_id and task string, not a nested target/task_context", () => {
    const payload = setTaskPayload({ batchId: "b1" }, "Pour water into the cup.");
    expect(payload).toEqual({ batch_id: "b1", task: "Pour water into the cup." });
  });

  it("batchTargetPayload is a flat batch_id, not a nested target", () => {
    expect(batchTargetPayload({ batchId: "b1" })).toEqual({ batch_id: "b1" });
  });

  it("episodeTargetPayload is flat batch_id/episode_id, not a nested target", () => {
    expect(episodeTargetPayload({ batchId: "b1", episodeId: "e1" })).toEqual({ batch_id: "b1", episode_id: "e1" });
  });
});
