import { useEffect, useRef, useState } from "react";

import type { VisualizationInfo } from "./captureProtocol";

interface PoseViserPanelProps {
  visualization: VisualizationInfo | null;
  /** Changes when the backend restarts, which is the only time the embedded view should be rebuilt. */
  backendInstanceId: string | null;
}

/** The backend's Viser 3D tracking view, embedded by iframe. Purely presentational: it never parses poses and has
 * no connection of its own to report -- an iframe load event does not prove Viser's live WebSocket is connected.
 * The iframe keeps one identity across snapshots, tracking loss and recording changes, so its orbit camera is not
 * reset; Reset view reloads it, and the backend frames every newly connected browser on the default camera. */
export function PoseViserPanel({ visualization, backendInstanceId }: PoseViserPanelProps) {
  const [retries, setRetries] = useState(0);
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const [frameFocused, setFrameFocused] = useState(false);

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

  if (visualization === null || visualization.status === "disabled") return null;

  return (
    <section className="pose-viser" aria-label="3D tracking view">
      <h3 className="group-sublabel">3D tracking view</h3>
      {visualization.status === "ready" && visualization.url !== null ? (
        <>
          {/* Viser always draws its own control strip down the right edge. The iframe is made that much wider and
              the strip is cropped by the viewport, so only the 3D scene shows. */}
          <div className="pose-viser-viewport">
            <iframe
              ref={frameRef}
              key={`${backendInstanceId ?? ""}|${visualization.url}|${retries}`}
              className="pose-viser-frame"
              title="3D tracking view"
              src={visualization.url}
            />
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
