/** The task editor's catalog picker: offered when the operator chose a catalog, fills the text, preselects a
 * standing task into an empty draft only, and stays free text when no catalog is configured or loading fails. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { RecordingPanel } from "./RecordingPanel";
import type { RecordingSnapshot } from "./recordingTypes";
import type { TaskCatalog } from "./taskCatalog";

afterEach(cleanup);

const operatorProps = {
  // Edit task is offered only when a set-task action is wired; the tests below override it when they assert on it.
  actions: { onSetTask: vi.fn().mockResolvedValue({ outcome: "ok", reason: null }) },
  operatorToken: "<test-operator>",
  setOperatorToken: vi.fn(),
  clearOperatorToken: vi.fn(),
  controlToken: "<test-token>",
  setControlToken: vi.fn(),
  clearControlToken: vi.fn(),
};

function snapshot(overrides: Partial<RecordingSnapshot> = {}): RecordingSnapshot {
  return {
    batchId: "b1",
    episodePhase: "IDLE",
    currentEpisodeId: null,
    currentEpisodeElapsedMs: null,
    taskText: "",
    defaultOperator: null,
    autoFillOperatorName: false,
    savedCount: 0,
    lastResult: null,
    readyToRecord: true,
    revision: 0,
    ...overrides,
  };
}

const CATALOG: TaskCatalog = {
  fileName: "Kitchen.csv",
  tasks: [
    { name: "wash_cup", instruction: "Wash the cup in the sink." },
    { name: "pour_water", instruction: "Pour water into the cup." },
  ],
  preselected: { name: "pour_water", instruction: "Pour water into the cup." },
};

// The operator-name field is also a textbox, so the task editor's input is found by its own class.
const taskInput = () => document.querySelector(".recording-task-editor-input") as HTMLInputElement;

describe("RecordingPanel task catalog picker", () => {
  it("offers the catalog's tasks and fills the text with the picked instruction", async () => {
    render(<RecordingPanel {...operatorProps} recordingSnapshot={snapshot({ taskText: "Old task" })} loadTaskCatalog={async () => CATALOG} />);
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const picker = (await screen.findByLabelText("Task from Kitchen.csv")) as HTMLSelectElement;
    expect(picker.value).toBe("-1"); // the existing text matches no catalog task
    expect(taskInput().value).toBe("Old task"); // an existing task is never overwritten by the preselection
    fireEvent.change(picker, { target: { value: "0" } });
    expect(taskInput().value).toBe("Wash the cup in the sink.");
    expect(picker.value).toBe("0");
    // Editing the wording keeps the text and drops the picker back to "custom".
    fireEvent.change(taskInput(), { target: { value: "Wash the cup in the sink, then dry it." } });
    expect((screen.getByLabelText("Task from Kitchen.csv") as HTMLSelectElement).value).toBe("-1");
  });

  it("preselects the standing task into an empty draft, ready to save with Enter", async () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...operatorProps} recordingSnapshot={snapshot({ taskText: "" })} actions={{ onSetTask }} loadTaskCatalog={async () => CATALOG} />);
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    await screen.findByLabelText("Task from Kitchen.csv");
    expect(taskInput().value).toBe("Pour water into the cup.");
    fireEvent.keyDown(taskInput(), { key: "Enter" });
    expect(onSetTask).toHaveBeenCalledWith({ batchId: "b1" }, "Pour water into the cup.");
  });

  it("stays a plain text editor when no catalog is configured or the catalog cannot be loaded", async () => {
    const { unmount } = render(<RecordingPanel {...operatorProps} recordingSnapshot={snapshot()} loadTaskCatalog={async () => null} />);
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    expect(taskInput()).toBeTruthy();
    await Promise.resolve();
    expect(screen.queryByRole("combobox")).toBeNull();
    unmount();
    render(
      <RecordingPanel
        {...operatorProps}
        recordingSnapshot={snapshot()}
        loadTaskCatalog={async () => {
          throw new Error("backend down");
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    await Promise.resolve();
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(taskInput()).toBeTruthy();
  });

  it("fetches the catalog each time the editor opens, so a catalog imported meanwhile is seen", async () => {
    const load = vi.fn(async () => CATALOG);
    render(<RecordingPanel {...operatorProps} recordingSnapshot={snapshot({ taskText: "x" })} loadTaskCatalog={load} />);
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    await screen.findByLabelText("Task from Kitchen.csv");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    await screen.findByLabelText("Task from Kitchen.csv");
    expect(load).toHaveBeenCalledTimes(2);
  });
});
