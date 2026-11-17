// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The end-of-day dialog: the day's takes and their outcomes, the station's time and notes, correcting a task, saving. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { DayReportReply } from "./dayReportApi";
import { DAY_REPORT_FIXTURE, UPTIME_FIXTURE } from "./dayReportTestDoubles";
import { SettingsHttpError } from "./settingsApi";
import { WrapUpDialog, type WrapUpBackend, type WrapUpDialogProps } from "./WrapUpDialog";

afterEach(cleanup);

/** A backend that answers with `reply` and, like the real one, with the notes it was last given. */
function fakeBackend(reply: DayReportReply | Error = DAY_REPORT_FIXTURE): WrapUpBackend {
  let notes = reply instanceof Error ? "" : reply.uptime.notes;
  return {
    load: vi.fn(async () => {
      if (reply instanceof Error) throw reply;
      return { ...reply, uptime: { ...reply.uptime, notes } };
    }),
    save: vi.fn(async () => ({ ...DAY_REPORT_FIXTURE, saved: { report: "bench_a_2026-03-09_report.json", activity_log: "bench_a_2026-03-09_activity.json" } })),
    correct: vi.fn(async () => ({ session: "s", updated: ["episode_0", "episode_1"], missing: ["episode_2"], config_updated: true })),
    saveNotes: vi.fn(async (_token: string, _station: string, next: string) => {
      notes = next;
      return { ...UPTIME_FIXTURE, notes };
    }),
  };
}

function renderDialog(overrides: Partial<WrapUpDialogProps> = {}, backend = fakeBackend()) {
  const props: WrapUpDialogProps = { open: true, onClose: vi.fn(), controlToken: "tok", backend, ...overrides };
  render(<WrapUpDialog {...props} />);
  return { props, backend };
}

const region = (name: string) => screen.findByRole("region", { name });
const notesField = (card: HTMLElement) => within(card).getByLabelText("Notes") as HTMLTextAreaElement;

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

  it("shows today for the station in use, with nothing to choose a day or a station by", async () => {
    const { backend } = renderDialog();
    await region("Takes");
    expect(backend.load).toHaveBeenCalledWith("tok", expect.any(AbortSignal));
    expect(screen.getByRole("heading", { name: "End of day — Bench A, 2026-03-09" })).toBeTruthy();
    const body = document.querySelector(".settings-content-body") as HTMLElement;
    expect(Array.from(body.children).map((part) => part.getAttribute("aria-label") ?? part.textContent)).toEqual(["Takes", "Station time", "Sessions", "Save report"]);
    expect(body.querySelector('input[type="date"], input[type="checkbox"]')).toBeNull();
  });

  it("shows the day's takes by outcome, with rejected and pending takes counted", async () => {
    renderDialog();
    const takes = await region("Takes");
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

  it("opens the latest session and leaves the earlier ones collapsed", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const twoSessions = { ...DAY_REPORT_FIXTURE, report: { ...DAY_REPORT_FIXTURE.report, sessions: [{ ...session, session_id: "session_morning" }, session] } };
    renderDialog({}, fakeBackend(twoSessions));
    const sessions = await region("Sessions");
    expect(Array.from(sessions.querySelectorAll("details")).map((details) => details.open)).toEqual([false, true]);
  });

  it("says when nothing was recorded today", async () => {
    const empty: DayReportReply = {
      ...DAY_REPORT_FIXTURE,
      report: { ...DAY_REPORT_FIXTURE.report, sessions: [], totals: { ...DAY_REPORT_FIXTURE.report.totals, episodes: 0, seconds: 0 } },
      uptime: { ...UPTIME_FIXTURE, state: "ready", reason: null, active_seconds: 0, down_seconds: 0, blockers: [], notes: "" },
    };
    renderDialog({}, fakeBackend(empty));
    const time = await region("Station time");
    expect(screen.getByText("0 takes, 0m 00s recorded.")).toBeTruthy();
    expect(screen.getByText("Nothing was recorded today.")).toBeTruthy();
    expect(within(time).getByText("No take yet today.")).toBeTruthy();
    expect(within(time).queryByRole("list")).toBeNull();
    expect(within(time).queryByText("✓ Saved")).toBeNull();
  });

  it("counts a single take in the singular", async () => {
    renderDialog({}, fakeBackend({ ...DAY_REPORT_FIXTURE, report: { ...DAY_REPORT_FIXTURE.report, totals: { ...DAY_REPORT_FIXTURE.report.totals, episodes: 1 } } }));
    expect(within(await region("Takes")).getByText("1 take, 1h 02m recorded.")).toBeTruthy();
  });

  it("shows the report a newer read brought, not an older one that answers late", async () => {
    const pending: Array<(reply: DayReportReply) => void> = [];
    const backend = fakeBackend();
    backend.load = vi.fn(() => new Promise<DayReportReply>((resolve) => pending.push(resolve)));
    const props: WrapUpDialogProps = { open: true, onClose: vi.fn(), controlToken: "tok", backend };
    const { rerender } = render(<WrapUpDialog {...props} />);
    rerender(<WrapUpDialog {...props} controlToken="tok2" />);
    expect(pending).toHaveLength(2);
    await act(async () => pending[1]({ ...DAY_REPORT_FIXTURE, report: { ...DAY_REPORT_FIXTURE.report, station: "Cart B" } }));
    await act(async () => pending[0](DAY_REPORT_FIXTURE)); // the first read answers late
    expect(screen.getByRole("heading", { name: "End of day — Cart B, 2026-03-09" })).toBeTruthy();
  });

  it("takes no answer once it is closed", async () => {
    const pending: Array<(reason: Error) => void> = [];
    const backend = fakeBackend();
    backend.load = vi.fn(() => new Promise<DayReportReply>((_resolve, reject) => pending.push(reject)));
    const props: WrapUpDialogProps = { open: true, onClose: vi.fn(), controlToken: "tok", backend };
    const { rerender } = render(<WrapUpDialog {...props} />);
    rerender(<WrapUpDialog {...props} open={false} />);
    rerender(<WrapUpDialog {...props} />);
    await act(async () => pending[0](new SettingsHttpError("offline")));
    // The reopened dialog is waiting for its own read: the abandoned one's failure is not shown.
    expect(screen.getByText("Loading…")).toBeTruthy();
    expect(screen.queryByText(/The report is unavailable/)).toBeNull();
  });

  it("shows a take being recorded now", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const live: DayReportReply = {
      ...DAY_REPORT_FIXTURE,
      report: {
        ...DAY_REPORT_FIXTURE.report,
        totals: { ...DAY_REPORT_FIXTURE.report.totals, by_disposition: { ...DAY_REPORT_FIXTURE.report.totals.by_disposition, recording: { count: 1, seconds: 0 } } },
        sessions: [{ ...session, episodes: [{ ...session.episodes[4], disposition: "recording" }] }],
      },
    };
    renderDialog({}, fakeBackend(live));
    const takes = await region("Takes");
    expect(within(takes).getAllByRole("listitem").pop()?.textContent).toBe("1Recording0m 00s");
  });

  it("says why the report is unavailable, and closes on Escape and the Close button", async () => {
    const { props } = renderDialog({}, fakeBackend(new SettingsHttpError("control token rejected")));
    expect(await screen.findByText("The report is unavailable: control token rejected")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });
});

describe("station time in the end-of-day dialog", () => {
  it("shows the active and down time in one line and each blocker with its clock times", async () => {
    renderDialog();
    const time = await region("Station time");
    expect(within(time).getByText("Active 2h 05m · Down 17m")).toBeTruthy();
    expect(within(within(time).getByRole("list", { name: "Blockers" })).getAllByRole("listitem").map((line) => line.textContent)).toEqual([
      "11:02–11:14 · 12m 00s · Left wrist: cam_left: no recent frame",
      "14:05–now · 5m 00s · Left wrist: cam_left: no recent frame",
    ]);
    // The station's state is the backend's own: nothing here sets it.
    expect(within(time).getAllByRole("button").map((button) => button.textContent)).toEqual(["Save notes"]);
  });

  it("shows zero active and down time, not that there is no take, once the report counts one", async () => {
    renderDialog({}, fakeBackend({ ...DAY_REPORT_FIXTURE, uptime: { ...UPTIME_FIXTURE, state: "ready", reason: null, active_seconds: 0, down_seconds: 0, blockers: [] } }));
    const time = await region("Station time");
    expect(screen.getByText("5 takes, 1h 02m recorded.")).toBeTruthy();
    expect(within(time).getByText("Active 0m · Down 0m")).toBeTruthy();
    expect(within(time).queryByText("No take yet today.")).toBeNull();
  });

  it("shows the totals without a list when nothing blocked the station", async () => {
    renderDialog({}, fakeBackend({ ...DAY_REPORT_FIXTURE, uptime: { ...UPTIME_FIXTURE, down_seconds: 0, blockers: [] } }));
    const time = await region("Station time");
    expect(within(time).getByText("Active 2h 05m · Down 0m")).toBeTruthy();
    expect(within(time).queryByRole("list")).toBeNull();
  });

  it("saves the operator's notes and reads the report again", async () => {
    const { backend } = renderDialog();
    const time = await region("Station time");
    const notes = notesField(time);
    expect(notes.value).toBe("left glove battery");
    expect(notes.maxLength).toBe(4000);
    const save = within(time).getByRole("button", { name: "Save notes" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(within(time).getByText("✓ Saved")).toBeTruthy();
    fireEvent.change(notes, { target: { value: "left glove battery; replaced at 14:10" } });
    expect(within(time).queryByText("✓ Saved")).toBeNull();
    fireEvent.click(save);
    await waitFor(() => expect(backend.saveNotes).toHaveBeenCalledWith("tok", "Bench A", "left glove battery; replaced at 14:10"));
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2)); // the report is read again
    expect(notes.value).toBe("left glove battery; replaced at 14:10");
    expect(await within(time).findByText("✓ Saved")).toBeTruthy();
    expect(save.disabled).toBe(true);
  });

  it("keeps what the operator is typing when the report is read again", async () => {
    const { backend } = renderDialog();
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "typing…" } });
    const sessions = screen.getByRole("region", { name: "Sessions" });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2));
    await within(sessions).findByText(/^Corrected/);
    expect(notesField(time).value).toBe("typing…");
    expect((within(time).getByRole("button", { name: "Save notes" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("keeps the report, the notes being typed and the correction's outcome when the report cannot be read again", async () => {
    const backend = fakeBackend();
    const load = vi.mocked(backend.load);
    renderDialog({}, backend);
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "typing…" } });
    load.mockRejectedValueOnce(new SettingsHttpError("backend restarting"));
    const sessions = screen.getByRole("region", { name: "Sessions" });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    expect(await screen.findByText(/The report could not be read again, so it is shown as last read: backend restarting/)).toBeTruthy();
    expect(screen.queryByText(/The report is unavailable/)).toBeNull();
    expect(within(sessions).getByText(/^Corrected episode_0, episode_1/)).toBeTruthy();
    expect(notesField(time).value).toBe("typing…");
    expect(screen.getByRole("region", { name: "Takes" })).toBeTruthy();
    // Reading again is offered, and a read that answers takes the notice away without touching the draft.
    fireEvent.click(screen.getByRole("button", { name: "Read again" }));
    await waitFor(() => expect(screen.queryByText(/could not be read again/)).toBeNull());
    expect(load).toHaveBeenCalledTimes(3);
    expect(notesField(time).value).toBe("typing…");
  });

  it("keeps the saved notes on screen when the report cannot be read again after saving them", async () => {
    const backend = fakeBackend();
    renderDialog({}, backend);
    const time = await region("Station time");
    vi.mocked(backend.load).mockRejectedValueOnce(new SettingsHttpError("backend restarting"));
    fireEvent.change(notesField(time), { target: { value: "replaced the glove" } });
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    expect(await screen.findByText(/The report could not be read again/)).toBeTruthy();
    expect(notesField(time).value).toBe("replaced the glove");
    expect(within(time).getByText("✓ Saved")).toBeTruthy();
  });

  it("leaves the notes and their button able to hold focus while a save is on its way, and sends it once", async () => {
    const backend = fakeBackend();
    let finish: (uptime: typeof UPTIME_FIXTURE) => void = () => undefined;
    backend.saveNotes = vi.fn(() => new Promise<typeof UPTIME_FIXTURE>((resolve) => (finish = resolve)));
    renderDialog({}, backend);
    const time = await region("Station time");
    const notes = notesField(time);
    fireEvent.change(notes, { target: { value: "replaced the glove" } });
    const save = within(time).getByRole("button", { name: "Save notes" }) as HTMLButtonElement;
    fireEvent.click(save);
    expect(save.textContent).toBe("Saving…");
    // A disabled control gives up keyboard focus, leaving the operator outside the notes they are saving.
    expect([notes.disabled, notes.readOnly, save.disabled]).toEqual([false, true, false]);
    fireEvent.click(save);
    expect(backend.saveNotes).toHaveBeenCalledTimes(1);
    await act(async () => finish({ ...UPTIME_FIXTURE, notes: "replaced the glove" }));
    expect(notes.readOnly).toBe(false);
  });

  it("shows why the notes were not saved and keeps them", async () => {
    const backend = fakeBackend();
    backend.saveNotes = vi.fn(async () => {
      throw new SettingsHttpError("notes may be at most 4000 characters");
    });
    renderDialog({}, backend);
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "too long" } });
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    expect(await within(time).findByText("Not saved: notes may be at most 4000 characters")).toBeTruthy();
    expect(notesField(time).value).toBe("too long");
  });

  it("saves notes for the station they were typed for, even after the report was read again for another", async () => {
    const backend = fakeBackend();
    renderDialog({}, backend);
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "left glove swapped" } });
    // The station in use changes, and a correction has the report read again: it is Cart B's now.
    vi.mocked(backend.load).mockResolvedValue({ ...DAY_REPORT_FIXTURE, station_in_use: "Cart B", uptime: { ...UPTIME_FIXTURE, station: "Cart B", notes: "" } });
    const sessions = screen.getByRole("region", { name: "Sessions" });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2));
    await within(sessions).findByText(/^Corrected/);
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    await waitFor(() => expect(backend.saveNotes).toHaveBeenCalledWith("tok", "Bench A", "left glove swapped"));
  });

  it("keeps a draft with the station it was begun for when it is typed back to another station's saved notes", async () => {
    const backend = fakeBackend();
    renderDialog({}, backend);
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "left glove swapped" } });
    // The report is read again and is Cart B's, whose notes are empty.
    vi.mocked(backend.load).mockResolvedValue({ ...DAY_REPORT_FIXTURE, station_in_use: "Cart B", uptime: { ...UPTIME_FIXTURE, station: "Cart B", notes: "" } });
    const sessions = screen.getByRole("region", { name: "Sessions" });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await within(sessions).findByText(/^Corrected/);
    // Emptying the field matches Cart B's notes, but the draft is Bench A's: it stays unsaved and stays Bench A's.
    fireEvent.change(notesField(time), { target: { value: "" } });
    expect((within(time).getByRole("button", { name: "Save notes" }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.change(notesField(time), { target: { value: "x" } });
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    await waitFor(() => expect(backend.saveNotes).toHaveBeenCalledWith("tok", "Bench A", "x"));
  });

  it("says the station in use changed when the notes are refused for that, keeps them, and saves them after a reload", async () => {
    const backend = fakeBackend();
    const saveNotes = vi.mocked(backend.saveNotes);
    saveNotes.mockRejectedValueOnce(new SettingsHttpError("the station in use changed; reload the report", 409));
    renderDialog({}, backend);
    const time = await region("Station time");
    fireEvent.change(notesField(time), { target: { value: "left glove swapped" } });
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    const refusal = await within(time).findByRole("alert");
    expect(refusal.textContent).toBe("The station in use changed; reload the report. Reload");
    expect(within(time).queryByText(/^Not saved/)).toBeNull();
    expect(saveNotes).toHaveBeenLastCalledWith("tok", "Bench A", "left glove swapped");
    expect(notesField(time).value).toBe("left glove swapped");
    // Reloading reads the report of the station now in use; the draft stays and is then saved for that station.
    vi.mocked(backend.load).mockResolvedValue({ ...DAY_REPORT_FIXTURE, report: { ...DAY_REPORT_FIXTURE.report, station: "Cart B" }, station_in_use: "Cart B", uptime: { ...UPTIME_FIXTURE, station: "Cart B", notes: "" } });
    fireEvent.click(within(refusal).getByRole("button", { name: "Reload" }));
    expect(await screen.findByRole("heading", { name: "End of day — Cart B, 2026-03-09" })).toBeTruthy();
    expect(within(time).queryByRole("alert")).toBeNull();
    expect(notesField(time).value).toBe("left glove swapped");
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    await waitFor(() => expect(saveNotes).toHaveBeenLastCalledWith("tok", "Cart B", "left glove swapped"));
  });

  it("closes on Escape in the notes while nothing is unsaved, also after typing back to what is saved", async () => {
    const { props } = renderDialog();
    const notes = notesField(await region("Station time"));
    fireEvent.keyDown(notes, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(1);
    fireEvent.change(notes, { target: { value: "left glove battery; replaced" } });
    fireEvent.change(notes, { target: { value: "left glove battery" } });
    fireEvent.keyDown(notes, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("Discard unsaved changes?")).toBeNull();
  });
});

describe("closing the end-of-day dialog over something unsaved", () => {
  const question = () => screen.queryByRole("group", { name: "Discard unsaved changes?" });
  const backdrop = () => document.querySelector(".settings-backdrop") as HTMLElement;
  /** The dialog with notes typed but not saved. */
  async function withUnsavedNotes() {
    const rendered = renderDialog();
    const notes = notesField(await region("Station time"));
    fireEvent.change(notes, { target: { value: "left glove battery; replaced" } });
    return { ...rendered, notes };
  }

  it("asks instead of closing on Escape from any control, on the backdrop and on Close", async () => {
    const { props, notes } = await withUnsavedNotes();
    const asks: Array<() => void> = [
      () => fireEvent.keyDown(notes, { key: "Escape" }),
      () => fireEvent.keyDown(screen.getByRole("button", { name: "Save report" }), { key: "Escape" }),
      () => fireEvent.click(backdrop()),
      () => fireEvent.click(screen.getByRole("button", { name: "Close" })),
    ];
    for (const ask of asks) {
      ask();
      expect(question()?.textContent).toBe("Discard unsaved changes?DiscardKeep editing");
      // The question stands in for Close until it is answered.
      expect(screen.queryByRole("button", { name: "Close" })).toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
      expect(question()).toBeNull();
    }
    expect(props.onClose).not.toHaveBeenCalled();
    expect(notes.value).toBe("left glove battery; replaced");
  });

  it("closes once the operator says to discard", async () => {
    const { props } = await withUnsavedNotes();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });

  it("never discards on Escape: a second one takes the question away, and the backdrop leaves it", async () => {
    const { props, notes } = await withUnsavedNotes();
    fireEvent.keyDown(notes, { key: "Escape" });
    fireEvent.click(backdrop());
    expect(question()).not.toBeNull();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(question()).toBeNull();
    expect(props.onClose).not.toHaveBeenCalled();
  });

  it("puts focus on Keep editing, then back where the operator was, or on Close when that is gone", async () => {
    const { notes } = await withUnsavedNotes();
    notes.focus();
    fireEvent.keyDown(notes, { key: "Escape" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Keep editing" }));
    fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(document.activeElement).toBe(notes);
    // Close itself gives way to the question, so it cannot take focus back as the same element.
    const close = screen.getByRole("button", { name: "Close" });
    close.focus();
    fireEvent.click(close);
    fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close" }));
  });

  it("asks over a half-typed correction too, and no longer once it is cleared or applied", async () => {
    const { props } = renderDialog();
    const sessions = await region("Sessions");
    const takes = within(sessions).getByLabelText("Takes to correct");
    fireEvent.change(takes, { target: { value: "2-" } });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(question()).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
    fireEvent.change(takes, { target: { value: "" } });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(1);
    fireEvent.change(takes, { target: { value: "2" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await within(sessions).findByText(/^Corrected/);
    // The form reports that it holds nothing typed in an effect, which may run after the text is on screen.
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });

  it("closes without asking once the notes are saved", async () => {
    const { props } = await withUnsavedNotes();
    const time = screen.getByRole("region", { name: "Station time" });
    fireEvent.click(within(time).getByRole("button", { name: "Save notes" }));
    await within(time).findByText("✓ Saved");
    fireEvent.click(backdrop());
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });
});

describe("correcting a task", () => {
  it("corrects the task of a range of takes, says what changed, and reads the report again", async () => {
    const { backend } = renderDialog();
    const sessions = await region("Sessions");
    const correct = within(sessions).getByRole("button", { name: "Correct" }) as HTMLButtonElement;
    expect(correct.disabled).toBe(true);
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: " 0-2 " } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Place the cup on the shelf." } });
    fireEvent.click(correct);
    expect(await within(sessions).findByText("Corrected episode_0, episode_1. Not found: episode_2.")).toBeTruthy();
    // Nothing but the session, the takes and the task is sent.
    expect(backend.correct).toHaveBeenCalledWith("tok", { session: "session_20260309-081500_abc123", episodes: "0-2", task: "Place the cup on the shelf." });
    expect((within(sessions).getByLabelText("Takes to correct") as HTMLInputElement).value).toBe("");
    await waitFor(() => expect(backend.load).toHaveBeenCalledTimes(2));
  });

  it("offers nothing but the takes and the task to correct in a session recorded in one place", async () => {
    renderDialog();
    const form = (await region("Sessions")).querySelector(".wrap-up-correction") as HTMLElement;
    expect(Array.from(form.querySelectorAll("input, select, textarea")).map((control) => control.getAttribute("aria-label"))).toEqual(["Takes to correct", "Correct task"]);
    expect(within(await region("Sessions")).getAllByRole("columnheader").map((head) => head.textContent)).toEqual(["Take", "Outcome", "Time", "Task", "Operator", "Issues"]);
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
    // The take number shown is the one inside that group, which is what a correction addresses; both groups
    // number from zero, so each take also says which group it is in.
    const table = within(sessions).getAllByRole("table")[0];
    expect(within(table).getAllByRole("columnheader").map((head) => head.textContent)).toEqual(["Take", "Recorded under", "Outcome", "Time", "Task", "Operator", "Issues"]);
    expect(within(table).getAllByRole("row").slice(1).map((row) => Array.from(row.querySelectorAll("td")).slice(0, 2).map((cell) => cell.textContent))).toEqual([
      ["4", "Kitchen/2026-03-09-08-15-00"],
      ["0", "Bathroom/2026-03-09-08-40-00"],
    ]);
    fireEvent.change(holder, { target: { value: "Bathroom/2026-03-09-08-40-00" } });
    fireEvent.change(within(sessions).getByLabelText("Takes to correct"), { target: { value: "0" } });
    fireEvent.change(within(sessions).getByLabelText("Correct task"), { target: { value: "Wash the cup." } });
    fireEvent.click(within(sessions).getByRole("button", { name: "Correct" }));
    await waitFor(() => expect(backend.correct).toHaveBeenCalledWith("tok", { session: "Bathroom/2026-03-09-08-40-00", episodes: "0", task: "Wash the cup." }));
    expect(await within(sessions).findByText("Corrected episode_0, episode_1 in Bathroom/2026-03-09-08-40-00. Not found: episode_2.")).toBeTruthy();
    // The session whose only take was rejected has nothing to relabel, and its unreadable record is reported.
    expect(sessions.querySelectorAll(".wrap-up-correction")).toHaveLength(1);
    expect(within(sessions).getByText("session_nothing_kept")).toBeTruthy();
    expect(within(sessions).getByText("Unreadable take record: episode_9.json: unsupported manifest")).toBeTruthy();
  });

  it("says where each take is recorded as soon as one is held outside its session, even in a single group", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const routed = {
      ...DAY_REPORT_FIXTURE,
      report: {
        ...DAY_REPORT_FIXTURE.report,
        sessions: [
          {
            ...session,
            episodes: [
              // Kept takes carry their slot in the one group; the interrupted take keeps its place in the session.
              { ...session.episodes[0], location: "Kitchen/2026-03-09-08-15-00/episode_0" },
              { ...session.episodes[1], location: "Kitchen/2026-03-09-08-15-00/episode_1" },
              { ...session.episodes[4], location: `${session.session_id}/episode_1` },
            ],
          },
        ],
      },
    };
    renderDialog({}, fakeBackend(routed));
    const sessions = await region("Sessions");
    const table = within(sessions).getByRole("table");
    expect(within(table).getAllByRole("columnheader").map((head) => head.textContent)).toContain("Recorded under");
    expect(within(table).getAllByRole("row").slice(1).map((row) => Array.from(row.querySelectorAll("td")).slice(0, 2).map((cell) => cell.textContent))).toEqual([
      ["0", "Kitchen/2026-03-09-08-15-00"],
      ["1", "Kitchen/2026-03-09-08-15-00"],
      ["1", session.session_id],
    ]);
    // One group leaves nothing to choose for a correction.
    expect(within(sessions).queryByLabelText("Recorded under")).toBeNull();
  });
});

describe("saving the report", () => {
  it("saves the report and names the files written", async () => {
    const { backend } = renderDialog();
    await region("Takes");
    fireEvent.click(screen.getByRole("button", { name: "Save report" }));
    expect(await screen.findByText("Saved bench_a_2026-03-09_report.json and bench_a_2026-03-09_activity.json in station_status under the recording output.")).toBeTruthy();
    expect(backend.save).toHaveBeenCalledWith("tok");
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
});

describe("keyboard focus in the end-of-day dialog", () => {
  it("tabs past a collapsed session to the next thing that can take focus, and back", async () => {
    const [session] = DAY_REPORT_FIXTURE.report.sessions;
    const twoSessions = { ...DAY_REPORT_FIXTURE, report: { ...DAY_REPORT_FIXTURE.report, sessions: [{ ...session, session_id: "session_morning" }, session] } };
    renderDialog({}, fakeBackend(twoSessions));
    const sessions = await region("Sessions");
    const [collapsed, open] = Array.from(sessions.querySelectorAll("summary"));
    const notes = notesField(screen.getByRole("region", { name: "Station time" }));
    notes.focus();
    // Save notes is disabled, so the collapsed session's summary is next.
    fireEvent.keyDown(notes, { key: "Tab" });
    expect(document.activeElement).toBe(collapsed);
    // Its takes and its correction form are not on screen: the next stop is the open session.
    fireEvent.keyDown(collapsed, { key: "Tab" });
    expect(document.activeElement).toBe(open);
    fireEvent.keyDown(open, { key: "Tab" });
    expect(document.activeElement).toBe(within(open.parentElement as HTMLElement).getByLabelText("Takes to correct"));
    fireEvent.keyDown(document.activeElement as Element, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(open);
    fireEvent.keyDown(open, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(collapsed);
  });
});
