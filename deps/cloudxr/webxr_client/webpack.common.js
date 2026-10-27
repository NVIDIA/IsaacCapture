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

const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');
const { eagerExceptEmulatorParserRules } = require('./webpack.chunkNames.js');

function git(cmd) {
  try {
    return execSync(`git ${cmd}`, { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString()
      .trim();
  } catch {
    return '';
  }
}
let TELEOP_VERSION = '';
try {
  TELEOP_VERSION = fs.readFileSync(path.resolve(__dirname, '../../../VERSION'), 'utf8').trim();
} catch {}
// Source of truth for the CloudXR SDK is deps/cloudxr/.env.default's
// CXR_WEB_SDK_VERSION (also controls which tarball `npm install`
// consumes). Local builds that don't set SDK_VERSION report the installed SDK's own version;
// this package's version is not the SDK's.
function installedSdkVersion() {
  try {
    return require('@nvidia/cloudxr/package.json').version;
  } catch {
    return 'unknown';
  }
}
const CLIENT_SDK_VERSION = process.env.SDK_VERSION || installedSdkVersion();
const CLIENT_GIT_REF =
  process.env.CLIENT_GIT_REF || git('rev-parse --abbrev-ref HEAD') || 'unknown';
const CLIENT_GIT_SHA = (process.env.CLIENT_GIT_SHA || git('rev-parse HEAD') || 'unknown').slice(
  0,
  12
);
const CLIENT_BUILD_TIME = new Date().toISOString();

// WebXR input profile assets are used by default when @webxr-input-profiles/assets is installed.
// Set USE_LOCAL_WEBXR_ASSETS=0 to skip bundling local assets (build needs internet at runtime to load assets).
const useLocalWebxrAssets = process.env.USE_LOCAL_WEBXR_ASSETS !== '0';
let webxrAssetsPackagePath = null;
let WEBXR_ASSETS_VERSION = '';
if (useLocalWebxrAssets) {
  try {
    webxrAssetsPackagePath = require.resolve('@webxr-input-profiles/assets/package.json');
    const webxrAssetsPackage = require(webxrAssetsPackagePath);
    WEBXR_ASSETS_VERSION = webxrAssetsPackage.version;
  } catch {
    console.warn(
      'webpack: @webxr-input-profiles/assets not found; building without WebXR input profile assets (controller models will use fallback or be disabled).'
    );
  }
}

module.exports = {
  entry: {
    main: ['./src/index.tsx'],
  },

  // Enable webpack 5 persistent filesystem caching for faster incremental builds
  cache: {
    type: 'filesystem',
    buildDependencies: {
      config: [__filename],
    },
  },

  // Module rules define how different file types are processed
  module: {
    rules: [
      ...eagerExceptEmulatorParserRules,
      {
        test: /\.tsx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            // Only transpile, don't type-check (faster builds)
            transpileOnly: true,
          },
        },
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.svg$/,
        type: 'asset/inline',
      },
    ],
  },

  // Resolve configuration for module resolution
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    alias: {
      // @helpers can be used instead of relative paths to the helpers directory
      '@helpers': path.resolve(__dirname, './helpers'),
    },
  },

  // Output configuration for bundled files
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, './build'),
  },

  // Webpack plugins that extend webpack's functionality
  plugins: [
    // Generates HTML file and automatically injects bundled JavaScript
    new HtmlWebpackPlugin({
      template: './src/index.html',
      favicon: './favicon.ico',
    }),

    // Inject environment variables
    new webpack.DefinePlugin({
      'process.env.WEBXR_ASSETS_VERSION': JSON.stringify(WEBXR_ASSETS_VERSION),
      'process.env.CLIENT_TELEOP_VERSION': JSON.stringify(TELEOP_VERSION),
      'process.env.CLIENT_SDK_VERSION': JSON.stringify(CLIENT_SDK_VERSION),
      'process.env.CLIENT_GIT_REF': JSON.stringify(CLIENT_GIT_REF),
      'process.env.CLIENT_GIT_SHA': JSON.stringify(CLIENT_GIT_SHA),
      'process.env.CLIENT_BUILD_TIME': JSON.stringify(CLIENT_BUILD_TIME),
    }),

    // Copies WebXR input profile assets when available; always copies favicon
    new CopyWebpackPlugin({
      patterns: [
        ...(webxrAssetsPackagePath
          ? [
              {
                from: path.join(
                  path.dirname(webxrAssetsPackagePath),
                  'dist',
                  'profiles',
                  'profilesList.json'
                ),
                to: `npm/@webxr-input-profiles/assets@${WEBXR_ASSETS_VERSION}/dist/profiles/profilesList.json`,
                toType: 'file',
              },
              ...[
                'meta-quest-touch-plus',
                'meta-quest-touch-plus-v2',
                'oculus-touch-v2',
                'oculus-touch-v3',
                'pico-4u',
                'generic-hand',
                'generic-trigger-squeeze-thumbstick',
              ].map(profile => ({
                from: path.join(path.dirname(webxrAssetsPackagePath), 'dist', 'profiles', profile),
                to: `npm/@webxr-input-profiles/assets@${WEBXR_ASSETS_VERSION}/dist/profiles/${profile}`,
              })),
            ]
          : []),
        { from: './favicon.ico', to: 'favicon.ico' },
      ],
    }),
  ],
};
