// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Settings dialog: tabs, the token-gated station and task changes, the live device checklist, catalog import
 * and removal, display and the read-only deployment tab. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
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
      // As the backend does: a role change names only the roles it moves.
      state.operator = { ...state.operator, ...changes, camera_roles: { ...state.operator.camera_roles, ...changes.camera_roles } };
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

  it("edits the camera setup itself while there are no stations, saving only what changed", async () => {
    const backend = fakeBackend({ operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "oak_d", camera_roles: { ego: "Ego", wrist_left: "Wrist A", wrist_right: null } } });
    renderPanel({ cameraSources: ["Ego", "Wrist A", "Wrist B"] }, backend);
    await stations();
    const body = document.querySelector(".settings-content-body") as HTMLElement;
    expect(Array.from(body.children).map((part) => part.getAttribute("aria-label"))).toEqual(["Stations", "Camera setup"]);
    const card = () => within(screen.getByRole("region", { name: "Camera setup" }));
    const save = () => card().getByRole("button", { name: "Save camera setup" }) as HTMLButtonElement;
    const select = (label: string) => card().getByLabelText(label) as HTMLSelectElement;
    expect(card().getByText(/Adding a station moves this setup onto the station/)).toBeTruthy();
    expect(select("Camera type").value).toBe("oak_d");
    expect(save().disabled).toBe(true); // nothing changed yet
    // The cameras offered are the session's, and one camera fills one role only.
    expect(Array.from(select("Right wrist").options).map((option) => [option.value, option.disabled])).toEqual([["", false], ["Ego", true], ["Wrist A", true], ["Wrist B", false]]);
    fireEvent.click(card().getByRole("button", { name: "Swap wrists" }));
    expect(select("Right wrist").value).toBe("Wrist A");
    fireEvent.click(save());
    await waitFor(() => expect(backend.saveOperator).toHaveBeenCalledWith("tok", { camera_roles: { wrist_left: null, wrist_right: "Wrist A" } }));
    await waitFor(() => expect(save().disabled).toBe(true));
    expect(select("Right wrist").value).toBe("Wrist A");
    fireEvent.change(select("Camera type"), { target: { value: "sensing_gmsl" } });
    fireEvent.click(save());
    await waitFor(() => expect(backend.saveOperator).toHaveBeenLastCalledWith("tok", { camera_type: "sensing_gmsl" }));
    // The first station takes the setup over, and from then on holds it: the card is gone.
    await waitFor(() => expect(backend.operator.camera_type).toBe("sensing_gmsl"));
    await waitFor(() => expect(save().textContent).toBe("Save camera setup"));
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench A" } });
    fireEvent.click(screen.getByRole("button", { name: "Add station" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Camera setup" })).toBeNull());
    expect(backend.registry.inUse()).toMatchObject({ camera_type: "sensing_gmsl", camera_roles: { ego: "Ego", wrist_left: null, wrist_right: "Wrist A" } });
  });

  it("shows why a camera setup save was refused and keeps what was chosen", async () => {
    const backend = fakeBackend();
    backend.saveOperator = vi.fn(async () => {
      throw new SettingsHttpError("camera 'Ego' is not offered by the capture session");
    });
    renderPanel({ cameraSources: ["Ego"] }, backend);
    await stations();
    const card = screen.getByRole("region", { name: "Camera setup" });
    fireEvent.change(within(card).getByLabelText("Ego (head)"), { target: { value: "Ego" } });
    fireEvent.click(within(card).getByRole("button", { name: "Save camera setup" }));
    expect(await within(card).findByText("Not saved: camera 'Ego' is not offered by the capture session")).toBeTruthy();
    expect((within(card).getByLabelText("Ego (head)") as HTMLSelectElement).value).toBe("Ego");
  });

  it("shows the camera setup read-only without the control token", async () => {
    renderPanel({ controlToken: null, cameraSources: ["Ego"] });
    await stations();
    const card = screen.getByRole("region", { name: "Camera setup" });
    expect((within(card).getByLabelText("Camera type") as HTMLSelectElement).disabled).toBe(true);
    expect((within(card).getByLabelText("Ego (head)") as HTMLSelectElement).disabled).toBe(true);
    expect((within(card).getByRole("button", { name: "Save camera setup" }) as HTMLButtonElement).disabled).toBe(true);
    expect(within(card).getByText(/Enter the control token in the recording panel/)).toBeTruthy();
  });

  it("shows the setup saved with the settings once the last station is deleted, not that station's", async () => {
    const backend = fakeBackend({ stations: BENCH_IN_USE, operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "sensing_gmsl" } });
    renderPanel({ cameraSources: ["Ego"] }, backend);
    const card = await stations();
    const reads: Array<() => void> = [];
    vi.mocked(backend.loadStationSetup).mockImplementation(
      () => new Promise<StationSetup>((resolve) => reads.push(() => resolve({ operator: backend.operator, tools: TOOLS_FIXTURE, taskFiles: backend.listing, stations: backend.registry.listing }))),
    );
    fireEvent.click(within(card).getByRole("button", { name: "Delete" }));
    fireEvent.click(within(card).getByRole("button", { name: "Delete Bench A" }));
    // The card appears with the registry change, before the setup has been read again without the station.
    const setup = await screen.findByRole("region", { name: "Camera setup" });
    expect(reads).toHaveLength(1);
    await act(async () => reads[0]());
    expect((within(setup).getByLabelText("Camera type") as HTMLSelectElement).value).toBe("sensing_gmsl");
    expect((within(setup).getByLabelText("Ego (head)") as HTMLSelectElement).value).toBe("");
    expect((within(setup).getByRole("button", { name: "Save camera setup" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("holds the camera setup and a first station when the setup cannot be read again after the last station is deleted", async () => {
    const backend = fakeBackend({ stations: BENCH_IN_USE, operator: { ...OPERATOR_SETTINGS_FIXTURE, camera_type: "sensing_gmsl" } });
    renderPanel({ cameraSources: ["Ego"] }, backend);
    const card = await stations();
    vi.mocked(backend.loadStationSetup).mockRejectedValueOnce(new Error("offline"));
    fireEvent.click(within(card).getByRole("button", { name: "Delete" }));
    fireEvent.click(within(card).getByRole("button", { name: "Delete Bench A" }));
    // What is on screen is still the deleted station's setup, so it is neither shown as the site's nor inherited.
    const retry = await screen.findByRole("button", { name: "Read again" });
    expect(screen.queryByRole("region", { name: "Camera setup" })).toBeNull();
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench B" } });
    expect((screen.getByRole("button", { name: "Add station" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(retry);
    const setup = await screen.findByRole("region", { name: "Camera setup" });
    expect((within(setup).getByLabelText("Camera type") as HTMLSelectElement).value).toBe("sensing_gmsl");
    expect(screen.queryByRole("button", { name: "Read again" })).toBeNull();
    expect((screen.getByRole("button", { name: "Add station" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("does not add the first station while a camera setup save is still on its way", async () => {
    const backend = fakeBackend();
    const saved: Array<() => void> = [];
    const realSave = backend.saveOperator;
    backend.saveOperator = vi.fn((token: string, changes) => new Promise<OperatorSettings>((resolve) => saved.push(() => resolve(realSave(token, changes)))));
    renderPanel({ cameraSources: ["Ego"] }, backend);
    await stations();
    const setup = screen.getByRole("region", { name: "Camera setup" });
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench A" } });
    const add = () => screen.getByRole("button", { name: "Add station" }) as HTMLButtonElement;
    expect(add().disabled).toBe(false);
    fireEvent.change(within(setup).getByLabelText("Ego (head)"), { target: { value: "Ego" } });
    fireEvent.click(within(setup).getByRole("button", { name: "Save camera setup" }));
    // The first station inherits the saved setup, so it waits for the save to answer.
    await waitFor(() => expect(add().disabled).toBe(true));
    await act(async () => saved[0]());
    await waitFor(() => expect(add().disabled).toBe(false));
  });

  it("does not let an older read of the station setup replace a newer change", async () => {
    const backend = fakeBackend({ stations: BENCH_IN_USE });
    renderPanel({ cameraSources: ["Ego", "Wrist A"] }, backend);
    const card = await stations();
    // From here each read of the setup answers only when the test lets it, with the registry as it was when sent.
    const reads: Array<() => void> = [];
    vi.mocked(backend.loadStationSetup).mockImplementation(() => {
      const picture = { operator: backend.operator, tools: TOOLS_FIXTURE, taskFiles: backend.listing, stations: backend.registry.listing };
      return new Promise<StationSetup>((resolve) => reads.push(() => resolve(picture)));
    });
    const editButton = () => within(card).getByRole("button", { name: "Edit" }) as HTMLButtonElement;
    // A rename starts a read; a camera change is saved while that read is still on its way.
    fireEvent.click(editButton());
    fireEvent.change(within(card).getByLabelText("Name"), { target: { value: "Bench Z" } });
    fireEvent.click(within(card).getByRole("button", { name: "Save" }));
    await within(card).findByLabelText("Use Bench Z");
    await waitFor(() => expect(editButton().disabled).toBe(false));
    fireEvent.click(editButton());
    fireEvent.change(within(card).getByLabelText("Left wrist"), { target: { value: "Wrist A" } });
    fireEvent.click(within(card).getByRole("button", { name: "Save" }));
    const assigned = "Ego (head): Ego · Left wrist: Wrist A · Right wrist: —";
    await within(card).findByText(assigned);
    expect(reads).toHaveLength(2);
    // The newer read answers first, then the older one, which still shows no left wrist camera.
    await act(async () => reads[1]());
    await act(async () => reads[0]());
    expect(within(card).getByText(assigned)).toBeTruthy();
  });

  it("reads the station setup again when a change is saved while a read is on its way", async () => {
    const backend = fakeBackend({ stations: BENCH_IN_USE });
    renderPanel({}, backend);
    const card = await stations();
    const reads: Array<(setup: StationSetup) => void> = [];
    const current = (): StationSetup => ({ operator: backend.operator, tools: TOOLS_FIXTURE, taskFiles: backend.listing, stations: backend.registry.listing });
    vi.mocked(backend.loadStationSetup).mockImplementation(() => new Promise<StationSetup>((resolve) => reads.push(resolve)));
    fireEvent.click(within(card).getByLabelText("Use no station"));
    await waitFor(() => expect(reads).toHaveLength(1));
    const before = current(); // read before the requirement is saved
    fireEvent.click(tab("Devices"));
    fireEvent.click(await screen.findByLabelText("Require calibration files before recording"));
    await waitFor(() => expect(reads).toHaveLength(2));
    await act(async () => reads[0](before));
    expect((screen.getByLabelText("Require calibration files before recording") as HTMLInputElement).checked).toBe(true);
    await act(async () => reads[1](current()));
    expect((screen.getByLabelText("Require calibration files before recording") as HTMLInputElement).checked).toBe(true);
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

  it("turns the calibration requirement on with the control token, and the checklist is read again at once", async () => {
    const backend = fakeBackend({ flow: SETUP_READY });
    renderPanel({ initialTab: "devices" }, backend);
    const box = (await screen.findByLabelText("Require calibration files before recording")) as HTMLInputElement;
    expect(box.checked).toBe(false);
    await waitFor(() => expect(backend.loadSetup).toHaveBeenCalledTimes(1));
    fireEvent.click(box);
    await waitFor(() => expect(backend.saveOperator).toHaveBeenCalledWith("tok", { require_calibration: true }));
    await waitFor(() => expect(backend.loadSetup).toHaveBeenCalledTimes(2));
    expect(box.checked).toBe(true);
    expect(backend.operator.require_calibration).toBe(true);
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
