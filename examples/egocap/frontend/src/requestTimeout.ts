// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * A bound on how long a request to the backend may go unanswered. Without one a backend that accepts a request
 * and never replies leaves a refresh loop waiting forever, or a button busy for good.
 */
import { SettingsHttpError } from "./settingsApi";

/** How long a read may take: the backend answers these from memory or a short walk of the day's files. */
export const READ_TIMEOUT_MS = 8_000;
/** How long a change may take: one can end a recording batch or rewrite a session's metadata first. */
export const CHANGE_TIMEOUT_MS = 30_000;
/** What the page says when the bound is reached. A change may still have been applied, so it says only that. */
export const NO_ANSWER = "the backend did not answer in time";

/** Run a request under a time bound and the caller's own signal; reaching the bound rejects with `NO_ANSWER`. */
export async function withTimeout<T>(signal: AbortSignal | undefined, ms: number, request: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  /** Pass the caller's abandonment on to the request. */
  const abandon = () => controller.abort();
  if (signal?.aborted) abandon();
  else signal?.addEventListener("abort", abandon, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, ms);
  try {
    return await request(controller.signal);
  } catch (error) {
    // The abort this bound caused is reported as what it means; the caller's own abort stays an abort.
    if (timedOut) throw new SettingsHttpError(NO_ANSWER);
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abandon);
  }
}
