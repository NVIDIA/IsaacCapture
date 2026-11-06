// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * PoC demo entry point for browsing and downloading recorded files -- not
 * part of the final system design, and not a recording-service
 * responsibility of its own. Polls the backend's /recordings/api/list so
 * newly-appearing subfolders (a session still being written, or one that
 * just finished) show up without a manual refresh.
 */

import { fetchRecordingListing, recordingFileUrl, RecordingsApiError, type RecordingEntry } from "./api";

const POLL_INTERVAL_MS = 4000;

let currentPath = "";
// Bumped by every navigation/poll; a response is only rendered if this is
// still the same generation when it arrives, so an overlapping poll tick or
// a response for a directory the user has since navigated away from can
// never render against the (by then) wrong `currentPath`.
let requestGeneration = 0;

function formatBytes(size: number | null): string {
  if (size === null) return "";
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function formatTimestamp(mtimeSeconds: number): string {
  return new Date(mtimeSeconds * 1000).toLocaleString();
}

function pathSegments(path: string): string[] {
  return path.split("/").filter((segment) => segment.length > 0);
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`#${id} element missing from index.html`);
  return element as T;
}

function setPath(path: string): void {
  currentPath = path;
  void refresh();
}

function renderBreadcrumbs(): void {
  const nav = requireElement<HTMLElement>("breadcrumbs");
  nav.innerHTML = "";

  const root = document.createElement("button");
  root.type = "button";
  root.textContent = "recordings";
  root.addEventListener("click", () => setPath(""));
  nav.appendChild(root);

  const segments = pathSegments(currentPath);
  const upTo: string[] = [];
  for (const segment of segments) {
    upTo.push(segment);
    const path = upTo.join("/");

    const sep = document.createElement("span");
    sep.className = "sep";
    sep.textContent = "/";
    nav.appendChild(sep);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = segment;
    button.addEventListener("click", () => setPath(path));
    nav.appendChild(button);
  }
}

function clearGrid(): void {
  requireElement<HTMLUListElement>("grid").innerHTML = "";
  requireElement<HTMLElement>("empty").hidden = true; // don't show "No recordings here yet." under an error banner
}

function renderEntries(path: string, entries: RecordingEntry[]): void {
  const grid = requireElement<HTMLUListElement>("grid");
  const empty = requireElement<HTMLElement>("empty");
  grid.innerHTML = "";
  empty.hidden = entries.length !== 0;

  for (const entry of entries) {
    const entryPath = path ? `${path}/${entry.name}` : entry.name;
    const li = document.createElement("li");
    li.className = "card";

    const link = entry.is_dir ? document.createElement("button") : document.createElement("a");
    if (link instanceof HTMLButtonElement) {
      link.type = "button";
      link.className = "folder";
      link.addEventListener("click", () => setPath(entryPath));
    } else {
      link.href = recordingFileUrl(entryPath);
      link.setAttribute("download", "");
    }

    const icon = document.createElement("span");
    icon.className = "icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = entry.is_dir ? "\u{1F4C1}" : "\u{1F4C4}";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = entry.name;

    link.append(icon, name);

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${entry.is_dir ? "folder" : formatBytes(entry.size)} · ${formatTimestamp(entry.mtime)}`;

    li.append(link, meta);
    grid.appendChild(li);
  }
}

async function refresh(): Promise<void> {
  renderBreadcrumbs();
  const path = currentPath;
  const generation = ++requestGeneration;
  const errorEl = requireElement<HTMLElement>("error");
  let listing;
  try {
    listing = await fetchRecordingListing(path);
  } catch (caught) {
    if (generation !== requestGeneration) return; // superseded by a newer navigation/poll; ignore
    clearGrid(); // don't leave the previous directory's stale entries on screen under the error
    if (caught instanceof RecordingsApiError && caught.status === 404 && path !== "") {
      // The directory we were viewing vanished (e.g. a take renamed away at
      // finalization) -- go up to the parent instead of re-polling a dead
      // path forever with the error banner stuck on screen.
      setPath(pathSegments(path).slice(0, -1).join("/"));
      return;
    }
    errorEl.hidden = false;
    errorEl.textContent = caught instanceof RecordingsApiError ? caught.message : String(caught);
    return;
  }
  if (generation !== requestGeneration) return; // superseded; a newer refresh already owns the view
  errorEl.hidden = true;
  renderEntries(path, listing.entries);
}

// Serial poll loop (not setInterval): each tick waits for the previous
// refresh to finish before scheduling the next, so a slow/hung request
// can never pile up overlapping in-flight polls. refresh() already handles
// listing errors internally; the try/catch here is a backstop so a tick
// that throws for some other reason (e.g. a DOM element missing) can't
// permanently kill the loop.
async function pollLoop(): Promise<void> {
  for (;;) {
    try {
      await refresh();
    } catch {
      // swallow and keep polling -- a stuck loop is worse than a skipped tick
    }
    await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

void pollLoop();
