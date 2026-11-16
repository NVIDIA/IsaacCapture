// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Settings dialog: tabs, the token-gated station and task changes, the live device checklist, catalog import
 * and removal, display and the read-only deployment tab. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
import { SettingsPanel, type SettingsBackend, type StationSetup } from "./SettingsPanel";
import { SettingsHttpError, type OperatorSettings, type TaskFileListing } from "./settingsApi";
import { KITCHEN_FIXTURE, OPERATOR_SETTINGS_FIXTURE, SETTINGS_FIXTURE, STATIONS_FIXTURE, TASK_FILES_FIXTURE, TOOLS_FIXTURE, fakeStationRegistry } from "./settingsTestDoubles";
import type { SetupStatus } from "./setupApi";
import { SETUP_HELD, SETUP_READY } from "./setupTestDoubles";
import type { StationRecord } from "./stationsApi";

afterEach(cleanup);

const MID_TRAIN = { name: "Mid.csv", format: "mid_train_csv", tasks: [{ name: "fold_towel", instruction: "Fold a towel" }] };

/** A fake backend over in-memory state, so saves show up on the next load like the real one. */
function fakeBackend(
  initial: Partial<StationSetup> & { flow?: SetupStatus } = {},
): SettingsBackend & { operator: OperatorSettings; listing: TaskFileListing; registry: ReturnType<typeof fakeStationRegistry> } {
  const registry = fakeStationRegistry(initial.stations);
  /** As the backend does: the station in use supplies the camera type and roles every reader sees. */
  const effective = (): OperatorSettings => {
    const inUse = registry.inUse();
    return inUse === null ? state.operator : { ...state.operator, camera_type: inUse.camera_type, camera_roles: inUse.camera_roles };
  };
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
    registry,
    addStation: vi.fn(registry.addStation),
    updateStation: vi.fn(registry.updateStation),
    deleteStation: vi.fn(registry.deleteStation),
    selectStation: vi.fn(registry.selectStation),
    loadSetup: vi.fn(async () => initial.flow ?? SETUP_READY),
    loadDeployment: vi.fn(async () => SETTINGS_FIXTURE),
    loadStationSetup: vi.fn(async () => ({ operator: effective(), tools: state.tools, taskFiles: state.listing, stations: registry.listing })),
    loadTaskFile: vi.fn(async (name: string) => (name === "Mid.csv" ? MID_TRAIN : KITCHEN_FIXTURE)),
    saveOperator: vi.fn(async (_token: string, changes) => {
      state.operator = { ...state.operator, ...changes };
      return effective();
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

const BENCH: StationRecord = { id: "st_1", name: "Bench A", camera_type: "oak_d", camera_roles: { ego: "Ego", wrist_left: null, wrist_right: null }, problems: [] };
const BENCH_IN_USE = { ...STATIONS_FIXTURE, stations: [BENCH], selected: "st_1", label: "Bench A" };

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
/** The Station tab once its setup has loaded. */
const stations = () => screen.findByRole("region", { name: "Stations" });

describe("SettingsPanel shell", () => {
  it("renders nothing while closed and does not fetch", () => {
    const backend = fakeBackend();
    renderPanel({ open: false }, backend);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(backend.loadDeployment).not.toHaveBeenCalled();
    expect(backend.loadStationSetup).not.toHaveBeenCalled();
  });

  it("offers Station, Devices, Tasks, Display and Advanced, in that order", () => {
    renderPanel();
    const nav = screen.getByRole("navigation", { name: "Settings sections" });
    expect(within(nav).getAllByRole("button").map((button) => button.textContent)).toEqual(["Station", "Devices", "Tasks", "Display", "Advanced"]);
  });

  it("opens on the Station tab, switches tabs, and closes on Escape or Close", async () => {
    const { props } = renderPanel();
    expect(await stations()).toBeTruthy();
    expect(tab("Station").getAttribute("aria-current")).toBe("page");
    fireEvent.click(tab("Display"));
    expect(screen.getByText("Keyboard shortcuts")).toBeTruthy();
    expect(tab("Display").getAttribute("aria-current")).toBe("page");
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });
});

describe("Station tab", () => {
  it("is the Stations card alone", async () => {
    renderPanel({}, fakeBackend({ stations: BENCH_IN_USE }));
    await stations();
    const body = document.querySelector(".settings-content-body") as HTMLElement;
    expect(Array.from(body.children).map((part) => part.getAttribute("aria-label"))).toEqual(["Stations"]);
  });

  it("adds the first station, puts it in use, and then edits that station's cameras in its row", async () => {
    const backend = fakeBackend({ operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "oak_d" } });
    renderPanel({ cameraSources: ["Ego", "Wrist A"] }, backend);
    const card = await stations();
    fireEvent.change(within(card).getByLabelText("New station name"), { target: { value: "Bench A" } });
    fireEvent.click(within(card).getByRole("button", { name: "Add station" }));
    await waitFor(() => expect((within(card).getByLabelText("Use Bench A") as HTMLInputElement).checked).toBe(true));
    // The camera type saved before any station existed became the station's own.
    expect(backend.registry.inUse()).toMatchObject({ name: "Bench A", camera_type: "oak_d" });
    // The station setup is read again, so the rest of the dialog goes by the station in use.
    expect(backend.loadStationSetup).toHaveBeenCalledTimes(2);
    expect(backend.loadStationSetup).toHaveBeenLastCalledWith(expect.any(AbortSignal));
    fireEvent.click(within(card).getByRole("button", { name: "Edit" }));
    const editor = within(card).getByRole("group", { name: "Edit Bench A" });
    expect((within(editor).getByLabelText("Camera type") as HTMLSelectElement).value).toBe("oak_d");
    fireEvent.change(within(editor).getByLabelText("Ego (head)"), { target: { value: "Ego" } });
    fireEvent.click(within(editor).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(backend.registry.inUse()?.camera_roles.ego).toBe("Ego"));
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_1", { camera_roles: { ego: "Ego" } });
    expect(await within(card).findByText("Ego (head): Ego · Left wrist: — · Right wrist: —")).toBeTruthy();
    // Camera setup goes through the station routes only.
    expect(backend.saveOperator).not.toHaveBeenCalled();
  });

  it("shows where the headset connects in the editor of the station in use, from the page's flow", async () => {
    const setup: SetupStatus = { ...SETUP_READY, headset: { address: "10.0.0.42", saved: "10.0.0.7", changed: true } };
    renderPanel({ setup }, fakeBackend({ stations: BENCH_IN_USE }));
    const card = await stations();
    const warning = "This host's address changed from 10.0.0.7 to 10.0.0.42. Enter the new one in the headset.";
    expect(within(card).getByText(warning)).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Edit" }));
    expect(within(card).getByText("Headset connects to 10.0.0.42")).toBeTruthy();
    expect(within(card).getAllByText(warning)).toHaveLength(1);
  });

  it("says nothing about the headset when the flow was read without the control token", async () => {
    renderPanel({ setup: SETUP_READY, controlToken: null }, fakeBackend({ stations: BENCH_IN_USE }));
    const card = await stations();
    expect(within(card).queryByText(/Headset connects to/)).toBeNull();
    expect(within(card).getByText(/Enter the control token in the recording panel/)).toBeTruthy();
  });

  it("closes a station's editor on Escape and stays open itself", async () => {
    const { props } = renderPanel({}, fakeBackend({ stations: BENCH_IN_USE }));
    const card = await stations();
    fireEvent.click(within(card).getByRole("button", { name: "Edit" }));
    fireEvent.keyDown(within(card).getByLabelText("Name"), { key: "Escape" });
    expect(within(card).queryByRole("group")).toBeNull();
    expect(props.onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(within(card).getByLabelText("New station name"), { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps the changed registry on screen when reading the setup again fails", async () => {
    const backend = fakeBackend();
    renderPanel({}, backend);
    const card = await stations();
    backend.loadStationSetup = vi.fn(async () => {
      throw new SettingsHttpError("stations are not available yet");
    });
    fireEvent.change(within(card).getByLabelText("New station name"), { target: { value: "Bench A" } });
    fireEvent.click(within(card).getByRole("button", { name: "Add station" }));
    expect(await within(card).findByLabelText("Use Bench A")).toBeTruthy();
  });
});

describe("Devices tab", () => {
  it("reads the flow itself with the control token and shows the verdict and the checklist", async () => {
    const { backend } = renderPanel({ initialTab: "devices" }, fakeBackend({ flow: SETUP_HELD }));
    expect((await screen.findByRole("status")).textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
    expect(backend.loadSetup).toHaveBeenCalledWith(expect.any(AbortSignal), "tok");
    const checklist = screen.getByRole("region", { name: "Checklist" });
    expect(within(checklist).getAllByRole("region").map((group) => group.getAttribute("aria-label"))).toEqual(["Tracking", "Cameras", "System"]);
    expect(within(checklist).getByText("Left wrist: cam_left")).toBeTruthy();
  });

  it("shows the page's flow at once, before its own read answers", () => {
    const backend = fakeBackend();
    backend.loadSetup = vi.fn(() => new Promise<SetupStatus>(() => {})); // never answers
    renderPanel({ initialTab: "devices", setup: SETUP_HELD }, backend);
    expect(screen.getByRole("status").textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
  });

  it("does not read the flow while another tab is shown", async () => {
    const { backend } = renderPanel();
    await stations();
    expect(backend.loadSetup).not.toHaveBeenCalled();
    fireEvent.click(tab("Devices"));
    expect(await screen.findByRole("status")).toBeTruthy();
    expect(backend.loadSetup).toHaveBeenCalledTimes(1);
  });

  it("lists every calibration tool disabled with the backend's reason, and scopes fisheye rectification to SENSING stations", async () => {
    renderPanel({ initialTab: "devices" }, fakeBackend({ operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "oak_d" } }));
    const tools = await screen.findByRole("region", { name: "Calibration tools" });
    const buttons = within(tools).getAllByRole("button", { name: "Open" }) as HTMLButtonElement[];
    expect(buttons).toHaveLength(3);
    expect(buttons.every((button) => button.disabled)).toBe(true);
    expect(within(tools).getAllByText(/not wired in this build/)).toHaveLength(3);
    expect(within(tools).getByText("Fisheye rectification")).toBeTruthy();
    expect(within(tools).getByText(/Only fisheye lenses need this/)).toBeTruthy();
  });
});

describe("Tasks tab", () => {
  it("lists the catalogs, lets the operator choose the one the task editor offers, and shows its preselect card", async () => {
    const { backend } = renderPanel();
    await stations();
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
    await stations();
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
    await stations();
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
    await stations();
    fireEvent.click(tab("Tasks"));
    const catalogs = await screen.findByRole("list", { name: "Task catalogs" });
    expect((within(catalogs).getByLabelText("Use Kitchen.csv in the task editor") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Choose a CSV file") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getAllByText(/Enter the control token in the recording panel/).length).toBeGreaterThan(0);
  });
});

describe("Display tab", () => {
  it("offers a side-panel reset only when the width differs from the default, and lists the shortcuts", async () => {
    const { props, unmount } = renderPanel({ sidebarWidth: 600 });
    await stations();
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

describe("SettingsPanel focus ownership", () => {
  it("moves focus inside, wraps Tab both ways and restores the invoking control", async () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();
    const { rerender, props } = renderPanel();
    const dialog = screen.getByRole("dialog");
    expect(document.activeElement).toBe(dialog);
    await stations();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(document.activeElement).toBe(tab("Station"));
    const controls = Array.from(dialog.querySelectorAll<HTMLElement>('button, input, select, textarea, a[href], [tabindex]'))
      .filter((element) => element.tabIndex >= 0 && !element.matches(":disabled"));
    fireEvent.keyDown(document.activeElement!, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(controls.at(-1));
    fireEvent.keyDown(document.activeElement!, { key: "Tab" });
    expect(document.activeElement).toBe(controls[0]);
    trigger.focus();
    expect(document.activeElement).toBe(dialog);
    // Updating callbacks must not lose ownership of the original invoking element.
    rerender(<SettingsPanel {...props} onClose={vi.fn()} />);
    rerender(<SettingsPanel {...props} open={false} />);
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });

  it("restores the invoking control on unmount", () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();
    const { unmount } = renderPanel();
    unmount();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });
});
