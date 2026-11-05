// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Draw every frame of a page from build_pages.py in headless Chrome, as PNGs.
// Usage: node capture.mjs <page.html> <outdir> <cssWidth> <cssHeight> <dpr>
// Writes <outdir>/NN-<label>/0000.png ... and <outdir>/meta.json. Node built-ins only.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [page, out, cssW, cssH, dpr] = process.argv.slice(2);
if (!dpr) {
  console.error("usage: node capture.mjs <page.html> <outdir> <cssWidth> <cssHeight> <dpr>");
  process.exit(2);
}
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9400 + Math.floor(Math.random() * 400);
const profile = mkdtempSync(join(tmpdir(), "cts-media-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--force-color-profile=srgb", "about:blank"],
  { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));

try {
  let targets = [];
  for (let i = 0; i < 100 && !targets.some(t => t.type === "page"); i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(200); }
  }
  const ws = new WebSocket(targets.find(t => t.type === "page").webSocketDebuggerUrl);
  await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = fail; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.error || r.result.exceptionDetails) throw new Error(JSON.stringify(r.error || r.result.exceptionDetails));
    return r.result.result.value;
  };
  await send("Emulation.setDeviceMetricsOverride", { width: Number(cssW), height: Number(cssH), deviceScaleFactor: Number(dpr), mobile: false });
  await send("Page.navigate", { url: "file://" + resolve(page) });
  for (let i = 0; i < 100; i++) {
    try { if (await ev("document.readyState === 'complete' && typeof render === 'function'")) break; } catch {}
    await sleep(100);
  }
  await ev("document.fonts.ready.then(() => true)");
  const steps = await ev(`setup(${Number(cssW)}, ${Number(cssH)})`);
  const fps = await ev("DATA.fps");
  const meta = { fps, size: [Number(cssW) * Number(dpr), Number(cssH) * Number(dpr)], steps: [] };
  for (const [si, s] of steps.entries()) {
    const dir = `${out}/${String(si + 1).padStart(2, "0")}-${s.label}`;
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    for (let f = 0; f < s.n; f++) {
      const url = await ev(`render(${si}, ${f}), cv.toDataURL("image/png")`);
      writeFileSync(`${dir}/${String(f).padStart(4, "0")}.png`, Buffer.from(url.split(",")[1], "base64"));
    }
    meta.steps.push({ dir, label: s.label, frames: s.n });
    console.log(dir, s.n);
  }
  writeFileSync(`${out}/meta.json`, JSON.stringify(meta, null, 1));
  ws.close();
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
