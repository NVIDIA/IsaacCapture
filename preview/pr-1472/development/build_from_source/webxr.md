<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build WebXR Client

#### TIP
A prebuilt CloudXR web client is available at [https://nvidia.github.io/IsaacCapture/client/main/](https://nvidia.github.io/IsaacCapture/client/main/).
You only need to build from source if you want to customize the client.

The Isaac Teleop WebXR client is built with React, Three.js, and WebXR. It streams VR/AR content
from a CloudXR Runtime server to the browser.  For more details about CloudXR.js SDK and its advanced features, see the
[CloudXR.js documentation](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html).

The source is located at [deps/cloudxr/webxr_client/](https://github.com/NVIDIA/IsaacCapture/tree/main/deps/cloudxr/webxr_client).

> ##### Steps
> 
> * [Prerequisites](#prerequisites)
> * [1. Download CloudXR.js SDK](#download-cloudxr-js-sdk)
> * [2. Install dependencies](#install-dependencies)
> * [3. Build & Run](#build-run)

## Prerequisites

- Node.js (v20 or higher). Ubuntu’s `nodejs` apt package is older (18.x on Ubuntu 24.04), so
  install a current release from [Node.js downloads](https://nodejs.org/en/download) (for
  example with nvm) and check it with `node --version`. `npm` ships with it.

## 1. Download CloudXR.js SDK

From the project root:

```bash
source scripts/setup_cloudxr_env.sh
scripts/download_cloudxr_sdk.sh
```

This will automatically download the CloudXR.js SDK and place it in `deps/cloudxr/`. The
package.json is configured to install the SDK from the downloaded local package.

## 2. Install dependencies

From the `deps/cloudxr/webxr_client/` directory:

```bash
npm install
```

## 3. Build & Run

### Production build

From `deps/cloudxr/webxr_client/`:

```bash
npm run build
```

Output in `build/`:

- `index.html` — entry page
- `bundle.js` — main application (UIKit, MSDF text, icons; loaded on first paint)
- `bundle.emulator.js` — desktop XR / IWER stack (DevUI, synthetic environments,
  and related deps); loaded only when emulation is used

### Development build (one-shot)

Outputs to `./build/` with source maps for debugging:

```bash
npm run dev
```

### Development server (with hot reload)

Starts a webpack dev server on `http://localhost:8080` with hot module replacement:

```bash
npm run dev-server
```

For **HTTPS** (required by WebXR on most browsers):

```bash
npm run dev-server:https
```

This starts the dev server with a self-signed certificate on `https://localhost:8080`.  You can follow the same
instructions to use the WebXR client as documented in the [Quick Start Guide](../../teleop/quick_start.md#connect-quest-pico).

<!-- References -->
