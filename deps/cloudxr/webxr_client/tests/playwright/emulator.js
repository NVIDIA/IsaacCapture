/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Production uses the emulator chunk; development names the IWER module chunk.
const EMULATOR_BUNDLE_PATTERN = /(?:emulator|iwer.*index_js).*\.js/;

module.exports = { EMULATOR_BUNDLE_PATTERN };
