// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Settings dialog: tabs, the token-gated station/task changes, catalog import and removal, tools, display
 * and the read-only deployment tab. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
import { SettingsPanel, type SettingsBackend, type StationSetup } from "./SettingsPanel";
import { SettingsHttpError, type OperatorSettings, type TaskFileListing } from "./settingsApi";
import { KITCHEN_FIXTURE, OPERATOR_SETTINGS_FIXTURE, SETTINGS_FIXTURE, TASK_FILES_FIXTURE, TOOLS_FIXTURE } from "./settingsTestDoubles";

afterEach(cleanup);

const MID_TRAIN = { name: "Mid.csv", format: "mid_train_csv", tasks: [{ name: "fold_towel", instruction: "Fold a towel" }] };

/** A fake backend over in-memory state, so saves show up on the next load like the real one. */
function fakeBackend(initial: Partial<StationSetup> = {}): SettingsBackend & { operator: OperatorSettings; listing: TaskFileListing } {
  const state = {
    operator: initial.operator ?? OPERATOR_SETTINGS_FIXTURE,
    listing: initial.taskFiles ?? TASK_FILES_FIXTURE,
    tools: initial.tools ?? TOOLS_FIXTURE,
  };
  return {
    get operator() {
      return state.operator;
    },
    get listing() {
      return state.listing;
    },
    loadDeployment: vi.fn(async () => SETTINGS_FIXTURE),
    loadStationSetup: vi.fn(async () => ({ operator: state.operator, tools: state.tools, taskFiles: state.listing })),
    loadTaskFile: vi.fn(async (name: string) => (name === "Mid.csv" ? MID_TRAIN : KITCHEN_FIXTURE)),
    saveOperator: vi.fn(async (_token: string, changes) => {
      state.operator = { ...state.operator, ...changes };
      return state.operator;
    }),
    importTaskFile: vi.fn(async (_token: string, source) => {
      const name = source.name;
      state.listing = { files: [...new Set([...state.listing.files, name])].sort(), imported: [...state.listing.imported, name] };
      return { ...state.listing, file: { ...MID_TRAIN, name } };
    }),
    removeTaskFile: vi.fn(async (_token: string, name: string) => {
      state.listing = { files: state.listing.files.filter((f) => f !== name), imported: state.listing.imported.filter((f) => f !== name) };
      return state.listing;
    }),
  };
}

function renderPanel(overrides: Partial<Parameters<typeof SettingsPanel>[0]> = {}, backend = fakeBackend()) {
  const props = {
    open: true,
    onClose: vi.fn(),
    sidebarWidth: 460,
    sidebarDefault: 460,
    onResetSidebar: vi.fn(),
    controlToken: "tok" as string | null,
    backend,
    ...overrides,
  };
  const utils = render(<SettingsPanel {...props} />);
  return { ...utils, props, backend };
}

const tab = (name: string) => screen.getByRole("button", { name });

describe("SettingsPanel shell", () => {
  it("renders nothing while closed and does not fetch", () => {
    const backend = fakeBackend();
    renderPanel({ open: false }, backend);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(backend.loadDeployment).not.toHaveBeenCalled();
    expect(backend.loadStationSetup).not.toHaveBeenCalled();
  });

  it("opens on the Station tab, switches tabs, and closes on Escape or Close", async () => {
    const { props } = renderPanel();
    expect(await screen.findByText("Camera type")).toBeTruthy();
    fireEvent.click(tab("Display"));
    expect(screen.getByText("Keyboard shortcuts")).toBeTruthy();
    expect(tab("Display").getAttribute("aria-current")).toBe("page");
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });
});

describe("Station tab", () => {
  it("saves the camera type with the control token, then has nothing left to save", async () => {
    const { backend } = renderPanel();
    const station = await screen.findByRole("region", { name: "Camera type" });
    const save = within(station).getByRole("button", { name: "Save camera type" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.click(within(station).getByLabelText(/OAK-D cameras/));
    expect(save.disabled).toBe(false);
    fireEvent.click(save);
    expect(backend.saveOperator).toHaveBeenCalledWith("tok", { camera_type: "oak_d" });
    expect(((await within(station).findByRole("button", { name: "Save camera type" })) as HTMLButtonElement).disabled).toBe(true);
    expect((within(station).getByLabelText(/OAK-D cameras/) as HTMLInputElement).checked).toBe(true);
  });

  it("cannot save without the control token and says where to enter it", async () => {
    const { backend } = renderPanel({ controlToken: null });
    const station = await screen.findByRole("region", { name: "Camera type" });
    fireEvent.click(within(station).getByLabelText(/SENSING GMSL cameras/));
    const save = within(station).getByRole("button", { name: "Save camera type" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(within(station).getByText(/Enter the control token in the recording panel/)).toBeTruthy();
    fireEvent.click(save);
    expect(backend.saveOperator).not.toHaveBeenCalled();
  });

  it("shows a rejected save in place and keeps the unsaved choice", async () => {
    const backend = fakeBackend();
    backend.saveOperator = vi.fn(async () => {
      throw new SettingsHttpError("control token rejected");
    });
    renderPanel({}, backend);
    const station = await screen.findByRole("region", { name: "Camera type" });
    fireEvent.click(within(station).getByLabelText(/OAK-D cameras/));
    fireEvent.click(within(station).getByRole("button", { name: "Save camera type" }));
    expect(await within(station).findByText(/Not saved: control token rejected/)).toBeTruthy();
    expect((within(station).getByLabelText(/OAK-D cameras/) as HTMLInputElement).checked).toBe(true);
  });

  it("assigns the session's cameras to roles, refuses one camera in two roles, swaps wrists, and saves only the roles", async () => {
    const { backend } = renderPanel({ cameraSources: ["Ego", "Wrist A", "Wrist B"] });
    const roles = await screen.findByRole("region", { name: "Camera roles" });
    const ego = within(roles).getByLabelText("Ego (head)") as HTMLSelectElement;
    const left = within(roles).getByLabelText("Left wrist") as HTMLSelectElement;
    const right = within(roles).getByLabelText("Right wrist") as HTMLSelectElement;
    const save = within(roles).getByRole("button", { name: "Save camera roles" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(ego, { target: { value: "Ego" } });
    fireEvent.change(left, { target: { value: "Wrist A" } });
    // A camera already used by another role cannot be picked again.
    expect((within(right).getByRole("option", { name: "Wrist A" }) as HTMLOptionElement).disabled).toBe(true);
    fireEvent.change(right, { target: { value: "Wrist B" } });
    fireEvent.click(within(roles).getByRole("button", { name: "Swap wrists" }));
    expect(left.value).toBe("Wrist B");
    expect(right.value).toBe("Wrist A");
    fireEvent.click(save);
    expect(backend.saveOperator).toHaveBeenCalledWith("tok", { camera_roles: { ego: "Ego", wrist_left: "Wrist B", wrist_right: "Wrist A" } });
    expect(((await within(roles).findByRole("button", { name: "Save camera roles" })) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByLabelText("Station readiness").textContent).toContain("3/3 roles assigned");
  });

  it("keeps a saved camera the session no longer offers selectable, marked, so it can be cleared", async () => {
    renderPanel(
      { cameraSources: ["Ego"] },
      fakeBackend({ operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_roles: { ego: "Ego", wrist_left: "Old wrist", wrist_right: null } } }),
    );
    const roles = await screen.findByRole("region", { name: "Camera roles" });
    const left = within(roles).getByLabelText("Left wrist") as HTMLSelectElement;
    expect(left.value).toBe("Old wrist");
    expect(within(left).getByRole("option", { name: "Old wrist (not in this session)" })).toBeTruthy();
    expect(screen.getByLabelText("Station readiness").textContent).toContain("1/2 assigned cameras in this session");
  });
});

describe("Tasks tab", () => {
  it("lists the catalogs, lets the operator choose the one the task editor offers, and shows its preselect card", async () => {
    const { backend } = renderPanel();
    await screen.findByText("Camera type");
    fireEvent.click(tab("Tasks"));
    const catalogs = await screen.findByRole("list", { name: "Task catalogs" });
    expect(within(catalogs).getByText("Kitchen.csv")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Preselected task" })).toBeNull();
    fireEvent.click(within(catalogs).getByLabelText("Use Kitchen.csv in the task editor"));
    expect(backend.saveOperator).toHaveBeenCalledWith("tok", { task_file: "Kitchen.csv", preselected_task: null });
    const preselect = await screen.findByRole("region", { name: "Preselected task" });
    expect(await within(preselect).findByRole("option", { name: /wash_cup/ })).toBeTruthy();
    fireEvent.change(within(preselect).getByLabelText("Preselected task"), { target: { value: "pour_water" } });
    fireEvent.click(within(preselect).getByRole("button", { name: "Save preselected task" }));
    expect(backend.saveOperator).toHaveBeenLastCalledWith("tok", { preselected_task: "pour_water" });
    expect(await within(catalogs).findByText("in task editor")).toBeTruthy();
  });

  it("imports catalogs by file, then lists it as imported and reports its task count", async () => {
    const { backend } = renderPanel();
    await screen.findByText("Camera type");
    fireEvent.click(tab("Tasks"));
    const add = await screen.findByRole("region", { name: "Add a catalog" });
    const mid = new File(["Task Name,Task Description\nfold_towel,Fold a towel\n"], "Mid.csv", { type: "text/csv" });
    fireEvent.change(within(add).getByLabelText("Choose a CSV file"), { target: { files: [mid] } });
    expect(await within(add).findByText("Imported Mid.csv: 1 task.")).toBeTruthy();
    const catalogs = screen.getByRole("list", { name: "Task catalogs" });
    expect(within(catalogs).getByText("Mid.csv")).toBeTruthy();
    expect(within(catalogs).getByText("imported")).toBeTruthy();
    // The file picker uploads the chosen file's text.
    const file = new File(["Task Name,Task Description\nfold_towel,Fold a towel\n"], "Garage.csv", { type: "text/csv" });
    fireEvent.change(within(add).getByLabelText("Choose a CSV file"), { target: { files: [file] } });
    expect(await within(add).findByText("Imported Garage.csv: 1 task.")).toBeTruthy();
    expect(backend.importTaskFile).toHaveBeenLastCalledWith("tok", { name: "Garage.csv", content: "Task Name,Task Description\nfold_towel,Fold a towel\n" });
  });

  it("shows import errors and requires deselection before removing an imported catalog", async () => {
    const backend = fakeBackend({
      operator: { ...OPERATOR_SETTINGS_FIXTURE, task_file: "Mid.csv" },
      taskFiles: { files: ["Kitchen.csv", "Mid.csv"], imported: ["Mid.csv"] },
    });
    backend.importTaskFile = vi.fn(async () => {
      throw new SettingsHttpError("Bad.csv is neither a studio catalog nor a Mid Train file");
    });
    renderPanel({}, backend);
    await screen.findByText("Camera type");
    fireEvent.click(tab("Tasks"));
    const add = await screen.findByRole("region", { name: "Add a catalog" });
    const bad = new File(["bad"], "Bad.csv", { type: "text/csv" });
    fireEvent.change(within(add).getByLabelText("Choose a CSV file"), { target: { files: [bad] } });
    expect(await within(add).findByText(/Not imported: Bad.csv is neither/)).toBeTruthy();
    const catalogs = screen.getByRole("list", { name: "Task catalogs" });
    expect(within(catalogs).getAllByRole("button", { name: "Remove" })).toHaveLength(1); // only the import
    expect(await screen.findByRole("region", { name: "Preselected task" })).toBeTruthy();
    expect((within(catalogs).getByRole("button", { name: "Remove" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Free text only" }));
    await waitFor(() => expect((within(catalogs).getByRole("button", { name: "Remove" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(within(catalogs).getByRole("button", { name: "Remove" }));
    expect(backend.removeTaskFile).toHaveBeenCalledWith("tok", "Mid.csv");
    await within(catalogs).findByText("Kitchen.csv");
    expect(within(catalogs).queryByText("Mid.csv")).toBeNull();
    expect(screen.queryByRole("region", { name: "Preselected task" })).toBeNull();
  });

  it("is read-only without the control token", async () => {
    renderPanel({ controlToken: null });
    await screen.findByText("Camera type");
    fireEvent.click(tab("Tasks"));
    const catalogs = await screen.findByRole("list", { name: "Task catalogs" });
    expect((within(catalogs).getByLabelText("Use Kitchen.csv in the task editor") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Choose a CSV file") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getAllByText(/Enter the control token in the recording panel/).length).toBeGreaterThan(0);
  });
});

describe("Tools tab", () => {
  it("lists every tool disabled with the backend's reason, and scopes fisheye rectification to SENSING stations", async () => {
    renderPanel({}, fakeBackend({ operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "oak_d" } }));
    await screen.findByText("Camera type");
    fireEvent.click(tab("Tools"));
    const tools = await screen.findByRole("region", { name: "Operator tools" });
    const buttons = within(tools).getAllByRole("button", { name: "Open" }) as HTMLButtonElement[];
    expect(buttons).toHaveLength(3);
    expect(buttons.every((button) => button.disabled)).toBe(true);
    expect(within(tools).getAllByText(/not wired in this build/)).toHaveLength(3);
    expect(within(tools).getByText("Fisheye rectification")).toBeTruthy();
    expect(within(tools).getByText(/Only fisheye lenses need this/)).toBeTruthy();
  });
});

describe("Display tab", () => {
  it("offers a side-panel reset only when the width differs from the default, and lists the shortcuts", async () => {
    const { props, unmount } = renderPanel({ sidebarWidth: 600 });
    await screen.findByText("Camera type");
    fireEvent.click(tab("Display"));
    expect(screen.getByText("600 px")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset side panel width" }));
    expect(props.onResetSidebar).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Stop and save")).toBeTruthy();
    unmount();
    renderPanel({ sidebarWidth: 460, initialTab: "display" });
    expect(screen.getByText("460 px (default)")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Reset side panel width" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Advanced tab", () => {
  it("shows the deployment settings grouped, with the variables they come from", async () => {
    renderPanel({ initialTab: "advanced" });
    expect(await screen.findByText("ws://127.0.0.1:8082")).toBeTruthy();
    expect(screen.getByText("head, controllers, hands")).toBeTruthy();
    expect(screen.getByText("head_human_camera (human)")).toBeTruthy();
    expect(screen.getByText("not configured")).toBeTruthy();
    expect(screen.queryByText("LiveKit credentials")).toBeNull();
    expect(screen.getByText("127.0.0.1:8085")).toBeTruthy();
    expect(screen.getByText(/EGOCAP_CAPTURE_SERVER_URL/)).toBeTruthy();
  });

  it("reports a station file error and a failed load in place", async () => {
    const backend = fakeBackend();
    backend.loadDeployment = vi.fn(async () => ({
      ...SETTINGS_FIXTURE,
      station: { ...SETTINGS_FIXTURE.station, path: "/etc/egocap/stations.yaml", error: "station configuration is invalid; see backend logs" },
    }));
    const { unmount } = renderPanel({ initialTab: "advanced" }, backend);
    expect(await screen.findByText("/etc/egocap/stations.yaml")).toBeTruthy();
    expect(screen.getByText(/station configuration is invalid/)).toBeTruthy();
    unmount();
    const failing = fakeBackend();
    failing.loadDeployment = vi.fn(async () => {
      throw new SettingsHttpError("settings are not resolved yet");
    });
    renderPanel({ initialTab: "advanced" }, failing);
    expect(await screen.findByText(/Deployment settings unavailable: settings are not resolved yet/)).toBeTruthy();
  });
});

describe("Deployment credential", () => {
  it("waits for a token and reloads deployment settings when one is entered", async () => {
    const { backend, props, rerender } = renderPanel({ controlToken: null, initialTab: "advanced" });
    expect(screen.getByText(/to view deployment settings/)).toBeTruthy();
    expect(backend.loadDeployment).not.toHaveBeenCalled();
    rerender(<SettingsPanel {...props} controlToken="new-token" />);
    await waitFor(() => expect(backend.loadDeployment).toHaveBeenCalledWith(expect.any(AbortSignal), "new-token"));
    expect(await screen.findByRole("region", { name: "Capture" })).toBeTruthy();
  });
});
