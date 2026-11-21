// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Thin client for the standalone recordings.py backend module's
 * endpoints -- nothing shared with the main frontend's API surface. The
 * listing and every file need the run's control token; `signIn` exchanges
 * it once for a same-site cookie, which the listing calls and the plain
 * download links then carry by themselves. */

export interface RecordingEntry {
  name: string;
  is_dir: boolean;
  size: number | null;
  mtime: number;
}

export interface RecordingListing {
  path: string;
  entries: RecordingEntry[];
}

export class RecordingsApiError extends Error {
  /** HTTP status of the failed response, or `null` for a network error/timeout
   * (no response was ever received to have a status). */
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.status = status;
  }
}

const FETCH_TIMEOUT_MS = 10000;

/** Encodes each path component so a filename containing #, ?, % or similar
 * can't be misread as a URL fragment/query/escape -- while leaving the "/"
 * separators themselves intact. */
function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

/** Fetches the listing for `path` ("" for the recordings root). Throws on
 * a non-200 response, network failure, or timeout, so callers can
 * distinguish "not found / offline" from a parsed empty directory -- and so
 * a hung request can't stall the poll loop indefinitely. */
export async function fetchRecordingListing(path: string): Promise<RecordingListing> {
  const url = path ? `/recordings/api/list/${encodePath(path)}` : "/recordings/api/list";
  let response: Response;
  try {
    response = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
  } catch (error) {
    throw new RecordingsApiError(`network error fetching ${url}: ${String(error)}`);
  }
  if (!response.ok) {
    throw new RecordingsApiError(`${url} returned ${response.status}`, response.status);
  }
  return (await response.json()) as RecordingListing;
}

/** HTTP status the backend answers with until the page has signed in. */
export const SIGN_IN_REQUIRED = 401;

/** Exchanges the control token for the session cookie. Throws with the
 * backend's status when the token is refused, so the form can say so. */
export async function signIn(controlToken: string): Promise<void> {
  const url = "/recordings/api/session";
  let response: Response;
  try {
    response = await fetch(url, { method: "POST", headers: { "X-Control-Token": controlToken }, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
  } catch (error) {
    throw new RecordingsApiError(`network error signing in: ${String(error)}`);
  }
  if (!response.ok) {
    throw new RecordingsApiError(response.status === SIGN_IN_REQUIRED ? "That control token was not accepted." : `${url} returned ${response.status}`, response.status);
  }
}

/** Download/view URL for a recorded file, given its full path
 * (as joined from listing entries, e.g. "sess1/episode_0/head_camera.mp4"). */
export function recordingFileUrl(path: string): string {
  return `/recordings/files/${encodePath(path)}`;
}
