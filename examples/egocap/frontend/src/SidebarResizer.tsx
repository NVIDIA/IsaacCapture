/**
 * The draggable divider between the main column and the side panel. Pulling it left widens the panel and narrows the
 * main column (and the other way round); it also works from the keyboard and resets on double-click.
 */

import { useState, type KeyboardEvent, type PointerEvent } from "react";

const KEY_STEP_PX = 24;
const KEY_STEP_LARGE_PX = 96;

interface SidebarResizerProps {
  width: number;
  min: number;
  max: number;
  /** A new panel width was requested (the caller clamps it). */
  onResize: (width: number) => void;
  /** Double-click: back to the default width. */
  onReset: () => void;
  /** A drag started or ended, so the page can stop the 3D view's iframe from swallowing the pointer meanwhile. */
  onDragChange: (dragging: boolean) => void;
}

export function SidebarResizer({ width, min, max, onResize, onReset, onDragChange }: SidebarResizerProps) {
  const [dragging, setDragging] = useState(false);

  const end = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    setDragging(false);
    onDragChange(false);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? KEY_STEP_LARGE_PX : KEY_STEP_PX;
    // The panel is on the right, so moving the divider left (ArrowLeft) makes it wider.
    const next =
      event.key === "ArrowLeft" ? width + step : event.key === "ArrowRight" ? width - step : event.key === "Home" ? min : event.key === "End" ? max : null;
    if (next === null) return;
    event.preventDefault();
    onResize(next);
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize side panel"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={width}
      tabIndex={0}
      className={`op-resizer${dragging ? " op-resizer--dragging" : ""}`}
      title="Drag to resize the side panel (double-click to reset)"
      onPointerDown={(event) => {
        event.preventDefault();
        // Capture, so the drag keeps tracking even when the pointer passes over the 3D view's iframe.
        event.currentTarget.setPointerCapture?.(event.pointerId);
        setDragging(true);
        onDragChange(true);
      }}
      onPointerMove={(event) => {
        // The panel is flush with the right edge of the window, so its width is the distance from the pointer to it.
        if (dragging) onResize(window.innerWidth - event.clientX);
      }}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
    />
  );
}
