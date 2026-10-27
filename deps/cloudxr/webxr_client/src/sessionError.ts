/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const STORAGE_KEY = 'cxr.isaac.lastSessionError';

export function readSessionError(): string {
  try {
    return sessionStorage.getItem(STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveSessionError(message: string): void {
  try {
    if (message) sessionStorage.setItem(STORAGE_KEY, message);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Restricted storage must not prevent error reporting or session cleanup.
  }
}
