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
 * Standalone build for tests/mock/CloudXRComponentTest and tests/mock/StreamAttachTimeoutTest -
 * both mount the REAL CloudXRComponent (helpers/react/CloudXRComponent.tsx) against MockCloudXR
 * via the same '@nvidia/cloudxr' alias webpack.app-mock.js uses for the full app, but as a
 * minimal Canvas/XR harness instead of App.tsx's production UI. Own multi-entry output in
 * build-component-mock/, never build/ or the other demo pages' output dirs.
 */

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development',
  devtool: 'eval-source-map',
  entry: {
    componentTest: './tests/mock/CloudXRComponentTest.tsx',
    streamAttachTimeoutTest: './tests/mock/StreamAttachTimeoutTest.tsx',
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: { loader: 'ts-loader', options: { transpileOnly: true } },
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    alias: {
      '@helpers': path.resolve(__dirname, './helpers'),
      // Exact match ($) only - see webpack.app-mock.js for why prefix-match would be wrong
      // (the shim/MockCloudXR.ts reach the real SDK via a subpath that must not be swallowed).
      '@nvidia/cloudxr$': path.resolve(__dirname, './tests/mock/cloudxr-mock-alias.ts'),
    },
  },
  output: {
    filename: 'bundle.[name].js',
    path: path.resolve(__dirname, './build-component-mock'),
    clean: true,
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: 'CloudXRComponentTest.html',
      template: './tests/mock/CloudXRComponentTest.html',
      chunks: ['componentTest'],
    }),
    new HtmlWebpackPlugin({
      filename: 'StreamAttachTimeoutTest.html',
      template: './tests/mock/StreamAttachTimeoutTest.html',
      chunks: ['streamAttachTimeoutTest'],
    }),
  ],
  devServer: {
    static: { directory: path.resolve(__dirname, './build-component-mock') },
    open: false,
    port: 8083,
  },
};
