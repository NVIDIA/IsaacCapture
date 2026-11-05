import { useEffect, useRef, useState } from "react";

import type { VisualizationInfo } from "./captureProtocol";

interface PoseViserPanelProps {
  visualization: VisualizationInfo | null;
  /** Changes when the backend restarts, which is the only time the embedded view should be rebuilt. */
  backendInstanceId: string | null;
}

/** How big the 3D view is drawn: inline in its column, a centred pop-up over the page, or the whole window. */
export type ViserSize = "normal" | "large" | "full";

const SIZES: ReadonlyArray<{ size: ViserSize; label: string }> = [
  { size: "normal", label: "Normal" },
  { size: "large", label: "Large" },
  { size: "full", label: "Full screen" },
];

/** The backend's Viser 3D tracking view, embedded by iframe. Purely presentational: it never parses poses and has
 * no connection of its own to report -- an iframe load event does not prove Viser's live WebSocket is connected.
 * The iframe keeps one identity across snapshots, tracking loss, recording changes and size changes, so its orbit
 * camera is not reset; Reset view reloads it, and the backend frames every newly connected browser on the default
 * camera. Sizing is CSS on the wrapper only, and the pop-up backdrop is a conditional sibling before the wrapper
 * (an empty slot when not shown), so the iframe's position in the tree never changes. The size is not remembered
 * across page loads: reopening the page straight into a pop-up or full screen would hide the recording controls. */
export function PoseViserPanel({ visualization, backendInstanceId }: PoseViserPanelProps) {
  const [retries, setRetries] = useState(0);
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const [frameFocused, setFrameFocused] = useState(false);
  const [size, setSize] = useState<ViserSize>("normal");

  // Keys typed inside the iframe never reach this page, so the recording shortcuts stop working while it has focus.
  // The parent window sees only a blur whose active element is the iframe; it is the one signal available here.
  useEffect(() => {
    const onBlur = () => setFrameFocused(document.activeElement === frameRef.current);
    const onFocus = () => setFrameFocused(false);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  // The pop-up and full screen cover the page, so the page behind must not scroll, and Escape closes them. Escape
  // typed inside the iframe never reaches this page, which is why a close button is always shown while covering.
  const covering = size !== "normal";
  useEffect(() => {
    if (!covering) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSize("normal");
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [covering]);

  if (visualization === null || visualization.status === "disabled") return null;

  const ready = visualization.status === "ready" && visualization.url !== null;

  return (
    <section className="pose-viser" aria-label="3D tracking view">
      <div className="pose-viser-header">
        <h3 className="group-sublabel">3D tracking view</h3>
        {ready && (
          <div className="pose-viser-sizes" role="group" aria-label="3D view size">
            {SIZES.map(({ size: option, label }) => (
              <button
                key={option}
                type="button"
                className="pose-viser-size"
                aria-pressed={size === option}
                onClick={() => setSize(option)}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
      {ready ? (
        <>
          {size === "large" && (
            <div className="pose-viser-backdrop" aria-hidden="true" onClick={() => setSize("normal")} />
          )}
          {/* Viser always draws its own control strip down the right edge. The iframe is made that much wider and
              the strip is cropped by the viewport, so only the 3D scene shows. */}
          <div className={`pose-viser-viewport pose-viser-viewport--${size}`}>
            <iframe
              ref={frameRef}
              key={`${backendInstanceId ?? ""}|${visualization.url}|${retries}`}
              className="pose-viser-frame"
              title="3D tracking view"
              src={visualization.url ?? undefined}
            />
            {covering && (
              <button type="button" className="pose-viser-exit" onClick={() => setSize("normal")}>
                {size === "full" ? "Exit full screen" : "Close"}
              </button>
            )}
          </div>
          {frameFocused && (
            <p className="pose-viser-note" role="status">
              Shortcuts are paused while the 3D view has focus.{" "}
              <button
                type="button"
                className="refresh-button"
                onClick={() => {
                  frameRef.current?.blur();
                  window.focus();
                  setFrameFocused(false);
                }}
              >
                Resume shortcuts
              </button>
            </p>
          )}
          <div className="pose-viser-actions">
            <button type="button" className="refresh-button" onClick={() => setRetries((n) => n + 1)}>
              Reset view
            </button>
          </div>
        </>
      ) : (
        <p className="empty-note">3D view unavailable{visualization.diagnostic !== null ? `: ${visualization.diagnostic}` : "."}</p>
      )}
      {visualization.status === "ready" && visualization.diagnostic !== null && (
        <p className="pose-viser-note" role="status">
          {visualization.diagnostic}
        </p>
      )}
    </section>
  );
}
