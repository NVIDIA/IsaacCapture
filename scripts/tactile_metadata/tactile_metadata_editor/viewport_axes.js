// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  "use strict";

  const key = "__tactileMetadataViewportAxes";
  globalThis[key]?.stop();

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  Object.assign(svg.style, {
    position: "absolute",
    top: "16px",
    right: "16px",
    width: "112px",
    height: "112px",
    zIndex: "10",
    pointerEvents: "none",
    overflow: "visible",
  });
  svg.setAttribute("viewBox", "0 0 112 112");

  const background = document.createElementNS(ns, "circle");
  background.setAttribute("cx", "56");
  background.setAttribute("cy", "56");
  background.setAttribute("r", "49");
  background.setAttribute("fill", "rgba(255,255,255,0.72)");
  background.setAttribute("stroke", "rgba(0,0,0,0.2)");
  svg.appendChild(background);

  const axes = [
    ["x", "#e53935", [1, 0, 0]],
    ["y", "#43a047", [0, 1, 0]],
    ["z", "#1e88e5", [0, 0, 1]],
  ].map(([name, color, vector]) => {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", "56");
    line.setAttribute("y1", "56");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "3");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);

    const end = document.createElementNS(ns, "circle");
    end.setAttribute("r", "4");
    end.setAttribute("stroke", color);
    end.setAttribute("stroke-width", "2");
    svg.appendChild(end);

    const label = document.createElementNS(ns, "text");
    label.textContent = name;
    label.setAttribute("fill", color);
    label.setAttribute("font-size", "15");
    label.setAttribute("font-family", "sans-serif");
    label.setAttribute("font-weight", "700");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("dominant-baseline", "central");
    svg.appendChild(label);
    return { color, end, label, line, vector };
  });

  const multiply = (a, b) => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];

  const rotate = (q, v) => {
    const tx = 2 * (q[1] * v[2] - q[2] * v[1]);
    const ty = 2 * (q[2] * v[0] - q[0] * v[2]);
    const tz = 2 * (q[0] * v[1] - q[1] * v[0]);
    return [
      v[0] + q[3] * tx + q[1] * tz - q[2] * ty,
      v[1] + q[3] * ty + q[2] * tx - q[0] * tz,
      v[2] + q[3] * tz + q[0] * ty - q[1] * tx,
    ];
  };

  const state = {
    controls: null,
    raf: 0,
    stopped: false,
    stop() {
      this.stopped = true;
      cancelAnimationFrame(this.raf);
      this.controls?.removeEventListener("update", draw);
      svg.remove();
    },
  };
  globalThis[key] = state;

  const draw = () => {
    const mutable = globalThis.__viserMutable;
    const camera = mutable?.camera;
    const root = mutable?.nodeRefFromName?.[""];
    const parent = mutable?.canvas?.parentElement;
    if (!camera || !root || !parent) return;
    if (svg.parentElement !== parent) parent.appendChild(svg);

    const cameraQuaternion = camera.quaternion;
    const rootQuaternion = root.quaternion;
    const cameraFromWorld = [
      -cameraQuaternion.x,
      -cameraQuaternion.y,
      -cameraQuaternion.z,
      cameraQuaternion.w,
    ];
    const cameraFromVisualization = multiply(cameraFromWorld, [
      rootQuaternion.x,
      rootQuaternion.y,
      rootQuaternion.z,
      rootQuaternion.w,
    ]);

    for (const [index, axis] of axes.entries()) {
      const direction = rotate(cameraFromVisualization, axis.vector);
      const x = 56 + 36 * direction[0];
      const y = 56 - 36 * direction[1];
      const projected = Math.hypot(direction[0], direction[1]);
      const fallback = [[10, -8], [-10, -8], [0, 12]][index];
      const labelOffset =
        projected > 0.12
          ? [8 * direction[0] / projected, -8 * direction[1] / projected]
          : fallback;
      axis.line.setAttribute("x2", x.toFixed(2));
      axis.line.setAttribute("y2", y.toFixed(2));
      axis.end.setAttribute("cx", x.toFixed(2));
      axis.end.setAttribute("cy", y.toFixed(2));
      axis.end.setAttribute("fill", direction[2] >= 0 ? axis.color : "white");
      axis.label.setAttribute("x", (x + labelOffset[0]).toFixed(2));
      axis.label.setAttribute("y", (y + labelOffset[1]).toFixed(2));
    }
  };

  const frame = () => {
    if (state.stopped) return;
    const controls = globalThis.__viserMutable?.cameraControl;
    if (controls !== state.controls) {
      state.controls?.removeEventListener("update", draw);
      controls?.addEventListener("update", draw);
      state.controls = controls;
    }
    draw();
    state.raf = requestAnimationFrame(frame);
  };
  frame();
})();
