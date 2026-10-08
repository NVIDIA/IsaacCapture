/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Builds the REAL app (src/index.tsx / src/index.html - same entry as webpack.dev.js) with
 * `@nvidia/cloudxr` aliased to tests/mock/cloudxr-mock-alias.ts, so CloudXRComponent.tsx's
 * `CloudXR.createSession(...)` call transparently gets MockCloudXR instead of a real session.
 * No production chunk-splitting invariant to preserve here (unlike webpack.dev.js/prod.js), so
 * this only reuses webpack.common.js's entry/HtmlWebpackPlugin/DefinePlugin/CopyWebpackPlugin -
 * output goes to build-app-mock/, never build/, so it can't collide with a real dev build.
 */

const path = require('path');
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  // production, not development: src/index.tsx wraps the app in
  // <React.StrictMode>, which double-invokes effects in a dev build - that calls
  // CloudXRComponent.tsx's establishSession() (and so CloudXR.createSession())
  // twice, leaving tests/mock/cloudxr-mock-alias.ts's activeSession pointing at
  // whichever MockCloudXR instance was created last while the mounted component
  // may be running against the other one. window.__mockCloudXRFail() then
  // silently no-ops against a session nothing is actually using. A production
  // build's NODE_ENV=production makes React skip StrictMode's double-invoke
  // entirely, so this is the actual fix, not a workaround.
  mode: 'production',
  devtool: 'source-map',
  resolve: {
    alias: {
      // Exact match ($) only: the shim itself (and MockCloudXR.ts) reach the real SDK via the
      // concrete '@nvidia/cloudxr/build/cloudxr.js' subpath, which must NOT be swallowed by this
      // alias (a prefix-match here would redirect that subpath back to the shim too, circularly).
      '@nvidia/cloudxr$': path.resolve(__dirname, './tests/mock/cloudxr-mock-alias.ts'),
    },
  },
  output: {
    filename: 'bundle.[contenthash:8].js',
    path: path.resolve(__dirname, './build-app-mock'),
    clean: true,
  },
  devServer: {
    static: { directory: path.resolve(__dirname, './build-app-mock') },
    open: false,
    port: 8082,
  },
});
