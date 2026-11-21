// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** A backend reason as a sentence: its messages start lower-case because they are also read mid-sentence. */
export function sentence(reason: string): string {
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}
