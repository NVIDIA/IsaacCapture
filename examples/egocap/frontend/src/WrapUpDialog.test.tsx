// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The end-of-day dialog: the day's takes and their outcomes, activity and notes, saving, and correcting a task. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { DayReportReply } from "./dayReportApi";
import { DAY_REPORT_FIXTURE } from "./dayReportTestDoubles";
import { SettingsHttpError } from "./settingsApi";
import { WrapUpDialog, type WrapUpBackend, type WrapUpDialogProps } from "./WrapUpDialog";

afterEach(cleanup);

function fakeBackend(reply: DayReportReply | Error = DAY_REPORT_FIXTURE): WrapUpBackend {
  return {
    load: vi.fn(async () => {
      if (reply instanceof Error) throw reply;
      return reply;
    }),
    save: vi.fn(async () => ({ ...DAY_REPORT_FIXTURE, saved: { report: "bench_a_2026-03-09_report.json", activity_log: "bench_a_2026-03-09_activity.json" } })),
    correct: vi.fn(async () => ({ session: "s", updated: ["episode_0", "episode_1"], missing: ["episode_2"], studio: null, config_updated: true })),
    saveNotes: vi.fn(async () => DAY_REPORT_FIXTURE.activity!),
  };
}

function renderDialog(overrides: Partial<WrapUpDialogProps> = {}, backend = fakeBackend()) {
  const props: WrapUpDialogProps = { open: true, onClose: vi.fn(), controlToken: "tok", backend, ...overrides };
  render(<WrapUpDialog {...props} />);
  return { props, backend };
}

const region = (name: string) => screen.findByRole("region", { name });

describe("WrapUpDialog", () => {
  it("renders nothing while closed and does not fetch", () => {
    const { backend } = renderDialog({ open: false });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(backend.load).not.toHaveBeenCalled();
  });

  it("asks for the control token instead of the report without it", () => {
    const { backend } = renderDialog({ controlToken: null });
    expect(screen.getByText("Enter the control token in the recording panel to see the day's report.")).toBeTruthy();
    expect(backend.load).not.toHaveBeenCalled();
  });

  it("shows the day's takes by outcome, with rejected and pending takes counted", async () => {
    const { backend } = renderDialog();
    const takes = await region("Takes");
    expect(backend.load).toHaveBeenCalledWith("tok", { date: "", station: undefined }, expect.any(AbortSignal));
    expect(screen.getByRole("heading", { name: "End of day — Bench A, 2026-03-09" })).toBeTruthy();
    expect(within(takes).getByText("5 takes, 1h 02m recorded.")).toBeTruthy();
    expect(within(takes).getAllByRole("listitem").map((tile) => tile.textContent)).toEqual([
      "2Kept1h 02m",
      "1Flagged0m 08s",
      "1Rejected0m 03s",
      "0Discarded0m 00s",
      "1Pending0m 04s",
    ]);
  });

  it("lists each session's takes with their real outcome, time, task and issues", async () => {
    renderDialog();
    const sessions = await region("Sessions");
    expect(within(sessions).getByText("08:15 · session_20260309-081500_abc123")).toBeTruthy();
    expect(within(sessions).getByText("Bench A · 2 kept · 1 flagged · 1 rejected · 1 pending")).toBeTruthy();
    const rows = within(sessions).getAllByRole("row").slice(1).map((row) => Array.from(row.querySelectorAll("td")).map((cell) => cell.textContent));
    expect(rows).toEqual([
      ["0", "Kept", "60:00", "Pour water.", "alex", "—"],
      ["1", "Kept", "2:05", "Pour water.", "alex", "—"],
      ["2", "Flagged", "0:08", "Pour water.", "alex", "sustained_freeze:pico_controller/head_pose"],
      ["3", "Rejected", "0:03", "Pour water.", "alex", "camera_min_fps:head_human_camera"],
      ["4", "Pending", "0:04", "Pour water.", "alex", "—"],
    ]);
  });

  it("shows the station's activity and saves the operator's notes", async () => {
    const { backend } = renderDialog();
    const activity = await region("Station activity");
    expect(within(activity).getByText("Bench A is blocked now.")).toBeTruthy();
    expect(Array.from(activity.querySelectorAll("dd")).map((value) => value.textContent)).toEqual(["5h 12m", "40m 00s", "2"]);
    const notes = within(activity).getByLabelText("Notes and blockers") as HTMLTextAreaElement;
    expect(notes.value).toBe("left glove battery");
    const save = within(activity).getByRole("button", { name: "Save notes" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(within(activity).getByText("✓ Saved")).toBeTruthy();
    fireEvent.change(notes, { target: { value: "left glove battery; replaced at 14:10" } });
    fireEvent.click(save);
    await waitFor(() => expect(backend.saveNotes).toHaveBeenCalledWith("tok", { notes: "left glove battery; replaced at 14:10" }));
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2)); // the report is read again
  });

  it("keeps what the operator is typing when the report is read again", async () => {
    const { backend } = renderDialog();
    const activity = await region("Station activity");
    const notes = within(activity).getByLabelText("Notes and blockers") as HTMLTextAreaElement;
    fireEvent.change(notes, { target: { value: "typing…" } });
    fireEvent.click(screen.getByLabelText("Every station"));
    await waitFor(() => expect(backend.load).toHaveBeenLastCalledWith("tok", { date: "", station: "*" }, expect.any(AbortSignal)));
    expect(notes.value).toBe("typing…");
  });

  it("corrects the task of a range of takes, says what changed, and reads the report again", async () => {
    const { backend } = renderDialog();
    const sessions = await region("Sessions");
    const correct = within(sessions).getByRole("button", { name: "Correct" }) as HTMLButtonElement;
    expect(correct.disabled).toBe(true);
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: " 0-2 " } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Place the cup on the shelf." } });
    fireEvent.click(correct);
    expect(await within(sessions).findByText("Corrected episode_0, episode_1. Not found: episode_2.")).toBeTruthy();
    expect(backend.correct).toHaveBeenCalledWith("tok", { session: "session_20260309-081500_abc123", episodes: "0-2", task: "Place the cup on the shelf." });
    expect((within(sessions).getByLabelText("Takes to correct") as HTMLInputElement).value).toBe("");
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2));
  });

  it("shows the backend's reason for a refused correction and keeps what was typed", async () => {
    const backend = fakeBackend();
    backend.correct = vi.fn(async () => {
      throw new SettingsHttpError("this session is still open for recording; start a new batch first");
    });
    renderDialog({}, backend);
    const sessions = await region("Sessions");
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    expect(await within(sessions).findByText("Not corrected: this session is still open for recording; start a new batch first")).toBeTruthy();
    expect((within(sessions).getByLabelText("Correct task") as HTMLInputElement).value).toBe("Wash the cup.");
  });

  it("offers each routed group of a session as the place to correct, and none when no take was finalized", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const routed = {
      ...DAY_REPORT_FIXTURE,
      report: {
        ...DAY_REPORT_FIXTURE.report,
        sessions: [
          {
            ...session,
            session_id: "session_routed",
            episodes: [
              { ...session.episodes[0], location: "Kitchen/2026-03-09-08-15-00/episode_4" },
              { ...session.episodes[1], location: "Bathroom/2026-03-09-08-40-00/episode_0" },
            ],
          },
          { ...session, session_id: "session_nothing_kept", started_at: null, episodes: [session.episodes[3]], problems: ["episode_9.json: unsupported manifest"] },
        ],
      },
    };
    const backend = fakeBackend(routed);
    renderDialog({}, backend);
    const sessions = await region("Sessions");
    const holder = within(sessions).getByLabelText("Recorded under") as HTMLSelectElement;
    expect(Array.from(holder.options).map((option) => option.value)).toEqual(["Kitchen/2026-03-09-08-15-00", "Bathroom/2026-03-09-08-40-00"]);
    // The take number shown is the one inside that group, which is what a correction addresses.
    expect(within(sessions).getAllByRole("row")[1].querySelector("td")?.textContent).toBe("4");
    fireEvent.change(holder, { target: { value: "Bathroom/2026-03-09-08-40-00" } });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await waitFor(() => expect(backend.correct).toHaveBeenCalledWith("tok", { session: "Bathroom/2026-03-09-08-40-00", episodes: "0", task: "Wash the cup." }));
    // The session whose only take was rejected has nothing to relabel, and its unreadable record is reported.
    expect(within(sessions).getAllByRole("button", { name: "Correct" })).toHaveLength(1);
    expect(within(sessions).getByText("session_nothing_kept")).toBeTruthy();
    expect(within(sessions).getByText("Unreadable take record: episode_9.json: unsupported manifest")).toBeTruthy();
  });

  it("saves the report and names the files written", async () => {
    const { backend } = renderDialog();
    await region("Takes");
    fireEvent.click(screen.getByRole("button", { name: "Save report" }));
    expect(await screen.findByText("Saved bench_a_2026-03-09_report.json and bench_a_2026-03-09_activity.json in station_status under the recording output.")).toBeTruthy();
    expect(backend.save).toHaveBeenCalledWith("tok", { date: "", station: undefined });
  });

  it("says why the report could not be saved", async () => {
    const backend = fakeBackend();
    backend.save = vi.fn(async () => {
      throw new SettingsHttpError("could not save the report");
    });
    renderDialog({}, backend);
    await region("Takes");
    fireEvent.click(screen.getByRole("button", { name: "Save report" }));
    expect(await screen.findByText("Not saved: could not save the report")).toBeTruthy();
  });

  it("reads another day and every station on request, where there is no activity to show", async () => {
    const empty: DayReportReply = {
      ...DAY_REPORT_FIXTURE,
      report: { ...DAY_REPORT_FIXTURE.report, date: "2026-03-08", station: null, sessions: [], totals: { ...DAY_REPORT_FIXTURE.report.totals, episodes: 1 } },
      activity: null,
    };
    const backend = fakeBackend(empty);
    renderDialog({}, backend);
    await region("Takes");
    expect(screen.getByRole("heading", { name: "End of day — every station, 2026-03-09".replace("09", "08") })).toBeTruthy();
    expect(screen.getByText("1 take, 1h 02m recorded.")).toBeTruthy();
    expect(screen.getByText("Nothing was recorded on this day.")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Station activity" })).toBeNull();
    fireEvent.change(screen.getByLabelText("Day"), { target: { value: "2026-03-07" } });
    await waitFor(() => expect(backend.load).toHaveBeenLastCalledWith("tok", { date: "2026-03-07", station: undefined }, expect.any(AbortSignal)));
  });

  it("shows a take being recorded now and a station that has not started its day", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const live: DayReportReply = {
      ...DAY_REPORT_FIXTURE,
      report: {
        ...DAY_REPORT_FIXTURE.report,
        totals: { ...DAY_REPORT_FIXTURE.report.totals, by_disposition: { ...DAY_REPORT_FIXTURE.report.totals.by_disposition, recording: { count: 1, seconds: 0 } } },
        sessions: [{ ...session, episodes: [{ ...session.episodes[4], disposition: "recording" }] }],
      },
      activity: { ...DAY_REPORT_FIXTURE.activity!, since: null, notes: "" },
    };
    renderDialog({}, fakeBackend(live));
    const takes = await region("Takes");
    expect(within(takes).getAllByRole("listitem").pop()?.textContent).toBe("1Recording0m 00s");
    expect(screen.getByText("Bench A is not started now.")).toBeTruthy();
    expect(screen.queryByText("✓ Saved")).toBeNull();
    const active: DayReportReply = { ...live, activity: { ...live.activity!, state: "active", since: 1 } };
    cleanup();
    renderDialog({}, fakeBackend(active));
    expect(await screen.findByText("Bench A is active now.")).toBeTruthy();
  });

  it("says why the report is unavailable, and closes on Escape and the Close button", async () => {
    const { props } = renderDialog({}, fakeBackend(new SettingsHttpError("control token rejected")));
    expect(await screen.findByText("The report is unavailable: control token rejected")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });
});
