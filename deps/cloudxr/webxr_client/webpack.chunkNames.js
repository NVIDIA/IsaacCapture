/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Human-readable chunk names for production builds.
 *
 * Target layout (two JS artifacts):
 * - ``bundle.js`` — main application (UIKit, MSDF text, Lucide icons, runtime)
 * - ``bundle.emulator.js`` — desktop XR / IWER code and its transitive deps
 *
 * Routing (no per-package allowlists):
 *   lazy ``import()`` under ``@pmndrs/xr/dist`` → ``bundle.emulator.js``
 *   LoadIWER's explicit lazy imports → ``bundle.emulator.js``
 *   other dynamic imports (UIKit → msdf, zustand, fonts, …) → eager → ``bundle.js``
 *
 * The MSDF web worker is inlined via ``asset/inline`` (no separate worker file).
 * OOB/--host-client sync downloads ``index.html``, ``bundle.js``, and
 * ``bundle.emulator.js``.
 */

/** Only non-main async chunk basename. */
const EMULATOR_CHUNK = 'emulator';

/**
 * Parser rules: eager dynamic imports everywhere except ``@pmndrs/xr/dist``, which
 * keeps the single lazy ``import('./emulate.js')`` boundary for IWER.
 *
 * @type {import('webpack').RuleSetRule[]}
 */
const eagerExceptEmulatorParserRules = [
  {
    test: /[\\/]@pmndrs[\\/]xr[\\/]dist[\\/]/,
    parser: { javascript: { dynamicImportMode: 'lazy' } },
  },
  {
    test: /\.(tsx?|jsx?|mjs|cjs)$/,
    parser: { javascript: { dynamicImportMode: 'eager' } },
  },
];

/** @type {import('webpack').Configuration['optimization']} */
const chunkOptimization = {
  chunkIds: 'named',
  splitChunks: {
    chunks: 'async',
    cacheGroups: {
      default: false,
      defaultVendors: false,
      // After eagerExceptEmulatorParserRules, async chunks are emulator-only.
      emulator: {
        name: EMULATOR_CHUNK,
        chunks: 'async',
        enforce: true,
      },
    },
  },
};

/** @type {import('webpack').Configuration['output']['chunkFilename']} */
function chunkFilename(pathData) {
  const name = pathData.chunk?.name;
  if (name === EMULATOR_CHUNK) {
    return 'bundle.emulator.js';
  }
  const id = String(pathData.chunk?.id ?? 'unknown');
  throw new Error(
    `Unexpected async chunk "${id}" (name=${name ?? ''}). ` +
      'Only bundle.emulator.js is allowed besides bundle.js. ' +
      'A dependency introduced a lazy import outside @pmndrs/xr/dist.'
  );
}

/** Inline MSDF worker as a data URL inside ``bundle.js`` (no extra worker file). */
const msdfInlineRules = {
  rules: [
    {
      test: /[\\/]@zappar[\\/]msdf-generator[\\/]dist[\\/]worker\.js$/,
      type: 'asset/inline',
    },
  ],
};

module.exports = {
  chunkFilename,
  chunkOptimization,
  eagerExceptEmulatorParserRules,
  msdfInlineRules,
};
