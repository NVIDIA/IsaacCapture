/**
 * The capture page's main data body: an expandable Head/Controllers/Hands
 * tree with complete measurement tables, replacing the old plain-text
 * tracking summary. Uses native `details`/`summary` so expansion state is
 * owned by the DOM itself -- as long as each branch's element stays
 * mounted (see the `key`-stability note on each branch below), expansion,
 * focus and scroll survive a live-to-stale-to-live transition without any
 * React state of its own.
 */
import { useEffect, useState } from "react";
import type { AcceptedTracking } from "./useCaptureSession";
import type { ControllerDisplay, DecodedTracking, HandDisplay, HeadDisplay } from "./trackingDecoder";

/** A sample older than this is no longer presented as current -- shared
 * with App.tsx's CameraGrid so tracking and camera staleness use one
 * consistent threshold. */
export const TRACKING_STALE_AFTER_MS = 2_000;

/** A cell's value is one of three distinct states -- a present-but-invalid
 * observation ("invalid") must never collapse into the same dash used for
 * absent/unavailable delivery, and neither may ever show a plausible-
 * looking number that isn't actually a live, valid measurement. */
type CellState = "ok" | "invalid" | "unavailable";

/** Renders one numeric table cell, formatted to `digits` decimals, or an
 * em dash for a missing/nonfinite value. */
function numCell(value: number | null, digits: number, key: number) {
  return (
    <td key={key} className="num-cell">
      {value === null || !Number.isFinite(value) ? "—" : value.toFixed(digits)}
    </td>
  );
}

/** Renders one pose-table row: a label cell plus up to 4 numeric columns
 * (X/Y/Z/W, with W left as "—" for a 3-component vector), or one merged
 * "invalid"/"—" cell spanning every numeric column when the pose itself
 * isn't usable. `columns` is the number of numeric columns in the table
 * (4 for Head's X/Y/Z/W, so callers share one table shape). */
function PoseRow({
  label,
  vector,
  digits,
  state,
  columns,
}: {
  label: string;
  vector: number[] | null;
  digits: number;
  state: CellState;
  columns: number;
}) {
  if (state !== "ok") {
    return (
      <tr>
        <th scope="row">{label}</th>
        <td className="num-cell" colSpan={columns}>
          {state === "invalid" ? "invalid" : "—"}
        </td>
      </tr>
    );
  }
  const cells = Array.from({ length: columns }, (_, i) => numCell(vector === null || i >= vector.length ? null : vector[i], digits, i));
  return (
    <tr>
      <th scope="row">{label}</th>
      {cells}
    </tr>
  );
}

/** Classifies one pose's display state from its delivered/decoded presence
 * (`current`) and its own native validity flag. */
function poseState(current: unknown, valid: boolean): CellState {
  if (current === null) return "unavailable";
  return valid ? "ok" : "invalid";
}

/** The Head branch of the tracking tree: always expanded, since its
 * compact pose table is the first useful measurement a visitor sees. */
function HeadBranch({ head }: { head: HeadDisplay | null }) {
  // `head === null` means delivery/sample unavailable, distinct from
  // `head.present === false`, a native fact that the headset genuinely
  // isn't reporting right now -- both still render this same branch so
  // its expansion/focus never move when the distinction changes.
  const current = head !== null && head.present ? head : null;
  const positionState = poseState(current, current?.isValid ?? false);
  const summaryValue = head === null ? "unavailable" : !head.present ? "absent" : positionState === "ok" ? `pos ${vecPreview(current!.position)}` : "invalid";
  return (
    <details className="tree-branch" open>
      <summary className="tree-summary">
        <span className="tree-summary-name">Head</span>
        <span className="tree-summary-value">{summaryValue}</span>
      </summary>
      <div className="data-table-wrapper">
        <table className="data-table">
          <caption className="sr-only">Head pose</caption>
          <thead>
            <tr>
              <th scope="col">Measurement</th>
              <th scope="col">X</th>
              <th scope="col">Y</th>
              <th scope="col">Z</th>
              <th scope="col">W</th>
            </tr>
          </thead>
          <tbody>
            <PoseRow label="Position (m)" vector={current?.position ?? null} digits={3} state={positionState} columns={4} />
            <PoseRow label="Orientation (XYZW)" vector={current?.orientation ?? null} digits={4} state={positionState} columns={4} />
          </tbody>
        </table>
      </div>
      <p className="tree-flags">
        Valid: {current === null ? "—" : current.isValid ? "yes" : "no"}
        {"   "}
        Tracked: {current === null ? "—" : current.isTracked ? "yes" : "no"}
      </p>
    </details>
  );
}

/** Formats a position vector as a short inline preview for a summary row;
 * null/unavailable renders nothing extra since the caller already states
 * the branch's availability in words. */
function vecPreview(v: number[] | null): string {
  return v === null ? "—" : v.map((x) => x.toFixed(2)).join(", ");
}

const CONTROLLER_INPUTS: { key: keyof ControllerDisplay; label: string }[] = [
  { key: "primaryClick", label: "Primary click" },
  { key: "secondaryClick", label: "Secondary click" },
  { key: "thumbstickX", label: "Thumbstick X" },
  { key: "thumbstickY", label: "Thumbstick Y" },
  { key: "thumbstickClick", label: "Thumbstick click" },
  { key: "menuClick", label: "Menu click" },
  { key: "squeezeValue", label: "Squeeze" },
  { key: "triggerValue", label: "Trigger" },
];

/** One controller side's nested branch: pose and inputs tables side by
 * side (stacked on narrow widths). */
function ControllerBranch({ label, controller }: { label: string; controller: ControllerDisplay | null }) {
  // Gate on native presence too: a decoded-but-absent controller can still
  // carry nonzero retained field values that must never leak into the
  // table as if they were real measurements (see trackingDecoder.ts).
  const current = controller !== null && controller.present ? controller : null;
  const toggleLabel = controller === null ? "unavailable" : controller.present ? "present" : "absent";
  return (
    <details className="tree-branch tree-branch--nested">
      <summary className="tree-summary">
        <span className="tree-summary-name">{label}</span>
        <span className="tree-summary-value">{toggleLabel}</span>
      </summary>
      <div className="tree-table-pair">
        <div className="data-table-wrapper">
          <table className="data-table">
            <caption className="sr-only">{label} pose</caption>
            <thead>
              <tr>
                <th scope="col">Measurement</th>
                <th scope="col">X</th>
                <th scope="col">Y</th>
                <th scope="col">Z</th>
                <th scope="col">W</th>
              </tr>
            </thead>
            <tbody>
              <PoseRow label="Grip position" vector={current?.gripPosition ?? null} digits={3} state={poseState(current, current?.gripValid ?? false)} columns={4} />
              <PoseRow label="Grip orientation" vector={current?.gripOrientation ?? null} digits={4} state={poseState(current, current?.gripValid ?? false)} columns={4} />
              <PoseRow label="Aim position" vector={current?.aimPosition ?? null} digits={3} state={poseState(current, current?.aimValid ?? false)} columns={4} />
              <PoseRow label="Aim orientation" vector={current?.aimOrientation ?? null} digits={4} state={poseState(current, current?.aimValid ?? false)} columns={4} />
            </tbody>
          </table>
        </div>
        <div className="data-table-wrapper">
          <table className="data-table">
            <caption className="sr-only">{label} inputs</caption>
            <thead>
              <tr>
                <th scope="col">Control</th>
                <th scope="col">Value</th>
              </tr>
            </thead>
            <tbody>
              {CONTROLLER_INPUTS.map(({ key, label: inputLabel }) => {
                const value = current?.[key] as number | null | undefined;
                return (
                  <tr key={key}>
                    <th scope="row">{inputLabel}</th>
                    {numCell(current === null ? null : (value ?? null), 2, 0)}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}

/** The standard OpenXR XR_EXT_hand_tracking 26-joint name/order (Palm,
 * Wrist, then 4 thumb + 5-per-finger joints for index/middle/ring/little)
 * -- the wire protocol identifies this ordering scheme by reference
 * (HandJointOrder.OPENXR_26, validated in captureProtocol.ts), not by
 * sending names, so the matching label set for that one supported scheme
 * lives here. */
const OPENXR_26_JOINT_NAMES = [
  "Palm",
  "Wrist",
  "Thumb Metacarpal",
  "Thumb Proximal",
  "Thumb Distal",
  "Thumb Tip",
  "Index Metacarpal",
  "Index Proximal",
  "Index Intermediate",
  "Index Distal",
  "Index Tip",
  "Middle Metacarpal",
  "Middle Proximal",
  "Middle Intermediate",
  "Middle Distal",
  "Middle Tip",
  "Ring Metacarpal",
  "Ring Proximal",
  "Ring Intermediate",
  "Ring Distal",
  "Ring Tip",
  "Little Metacarpal",
  "Little Proximal",
  "Little Intermediate",
  "Little Distal",
  "Little Tip",
] as const;

/** One hand's nested branch: the full 26-joint table, grouped by
 * Position/Orientation column headers. */
function HandBranch({ label, hand }: { label: string; hand: HandDisplay | null }) {
  const current = hand !== null && hand.present ? hand : null;
  const validCount = current === null ? 0 : current.joints.filter((j) => j.valid).length;
  const toggleLabel = hand === null ? "unavailable" : hand.present ? `${validCount}/26 joints valid` : "absent";
  return (
    <details className="tree-branch tree-branch--nested">
      <summary className="tree-summary">
        <span className="tree-summary-name">{label}</span>
        <span className="tree-summary-value">{toggleLabel}</span>
      </summary>
      <div className="data-table-wrapper">
        <table className="data-table data-table--hand">
          <caption className="sr-only">{label} joints</caption>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Joint
              </th>
              <th scope="col" colSpan={3}>
                Position (m)
              </th>
              <th scope="col" colSpan={4}>
                Orientation (XYZW)
              </th>
              <th scope="col" rowSpan={2}>
                Radius (m)
              </th>
              <th scope="col" rowSpan={2}>
                Valid
              </th>
            </tr>
            <tr>
              <th scope="col">X</th>
              <th scope="col">Y</th>
              <th scope="col">Z</th>
              <th scope="col">X</th>
              <th scope="col">Y</th>
              <th scope="col">Z</th>
              <th scope="col">W</th>
            </tr>
          </thead>
          <tbody>
            {OPENXR_26_JOINT_NAMES.map((name, i) => {
              const joint = current?.joints[i] ?? null;
              const state: CellState = current === null ? "unavailable" : joint !== null && joint.valid ? "ok" : "invalid";
              return (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  {state === "ok"
                    ? [0, 1, 2].map((j) => numCell(joint!.position?.[j] ?? null, 3, j))
                    : [
                        <td key="pos" className="num-cell" colSpan={3}>
                          {state === "invalid" ? "invalid" : "—"}
                        </td>,
                      ]}
                  {state === "ok"
                    ? [0, 1, 2, 3].map((j) => numCell(joint!.orientation?.[j] ?? null, 4, j))
                    : [
                        <td key="ori" className="num-cell" colSpan={4}>
                          {state === "invalid" ? "invalid" : "—"}
                        </td>,
                      ]}
                  <td className="num-cell">{state === "ok" ? (joint!.radius?.toFixed(3) ?? "—") : "—"}</td>
                  <td className={joint?.valid ? "" : "tone-warning"}>{current === null ? "—" : joint?.valid ? "yes" : "no"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** The top-level "Controllers" branch's summary text: each side's
 * presence, without exposing any individual field value. */
function controllerSummary(tracking: DecodedTracking | null): string {
  if (tracking === null) return "Left unavailable   Right unavailable";
  const side = (c: ControllerDisplay) => (c.present ? "present" : "absent");
  return `Left ${side(tracking.controllerLeft)}   Right ${side(tracking.controllerRight)}`;
}

/** The top-level "Hands" branch's summary text: each side's valid-joint
 * count, or its presence/availability when there is nothing to count. */
function handSummary(tracking: DecodedTracking | null): string {
  if (tracking === null) return "Left unavailable   Right unavailable";
  const side = (h: HandDisplay) => (h.present ? `${h.joints.filter((j) => j.valid).length}/26 valid` : "absent");
  return `Left ${side(tracking.handLeft)}   Right ${side(tracking.handRight)}`;
}

/** The page's main tracking tree: Head, Controllers and Hands as
 * top-level disclosures, driven by the single accepted tracking sample
 * the connection hook currently considers current. */
export function CaptureTree({
  trackingSample,
  diagnostic,
}: {
  trackingSample: AcceptedTracking | null;
  /** Why no/an old sample is being served, e.g. an oversized sample the
   * backend dropped -- shown ahead of the generic empty/stale label when
   * present, distinct from simply never having received a sample. */
  diagnostic: string | null;
}) {
  const [, forceTick] = useState(0);
  useEffect(() => {
    // Age/stale masking must keep advancing purely from wall-clock time,
    // not only when a new prop arrives -- without this tick, a sample that
    // crosses the stale threshold between heartbeats would stay displayed
    // as live until something unrelated happened to rerender this tree.
    const id = setInterval(() => forceTick((n) => n + 1), 1_000);
    return () => clearInterval(id);
  }, []);

  const elapsedSinceSnapshot = trackingSample === null ? 0 : performance.now() - trackingSample.snapshotReceivedAtMonotonic;
  const effectiveAge =
    trackingSample === null || trackingSample.snapshotAgeMs === null ? null : trackingSample.snapshotAgeMs + elapsedSinceSnapshot;
  const stale = effectiveAge === null || effectiveAge > TRACKING_STALE_AFTER_MS;
  const unavailable = trackingSample === null || stale;
  const tracking = unavailable ? null : trackingSample.decoded;
  const ageLabel = unavailable || effectiveAge === null ? null : `Sample age ${Math.round(effectiveAge)} ms`;
  const captionText = unavailable
    ? (diagnostic ?? (trackingSample === null ? "No tracking sample yet." : "Tracking stale."))
    : null;

  return (
    <section className="capture-tree" aria-label="Tracking data">
      <div className="capture-tree-heading">
        <h2 className="group-label">Tracking data</h2>
        {ageLabel !== null && <span className="tree-age">{ageLabel}</span>}
      </div>
      {captionText !== null && <p className="tree-caption" role="status">{captionText}</p>}
      <HeadBranch head={tracking === null ? null : tracking.head} />
      <details className="tree-branch">
        <summary className="tree-summary">
          <span className="tree-summary-name">Controllers</span>
          <span className="tree-summary-value">{controllerSummary(tracking)}</span>
        </summary>
        <ControllerBranch label="Left controller" controller={tracking === null ? null : tracking.controllerLeft} />
        <ControllerBranch label="Right controller" controller={tracking === null ? null : tracking.controllerRight} />
      </details>
      <details className="tree-branch">
        <summary className="tree-summary">
          <span className="tree-summary-name">Hands</span>
          <span className="tree-summary-value">{handSummary(tracking)}</span>
        </summary>
        <HandBranch label="Left hand" hand={tracking === null ? null : tracking.handLeft} />
        <HandBranch label="Right hand" hand={tracking === null ? null : tracking.handRight} />
      </details>
    </section>
  );
}
