<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand Reconstruction Workflow

`hand_reconstruction.yaml` is an OSMO workflow that runs a two-stage hand reconstruction pipeline:

1. **vipe-infer** — Runs ViPE inference on a source video to produce camera pose and depth estimates.
2. **dynhamr-reconstruct** — Feeds the ViPE output into Dyn-HaMR for full hand mesh reconstruction, then runs a jumpy-hand quality control check on the results.

## Prerequisites

- OSMO CLI installed and authenticated (`osmo login …`)
- Bucket and image registry credentials stored in OSMO
- An OSMO pool with GPU resources available

### Data files

The reconstruction pipeline requires two sets of external data files, stored in an S3 bucket referenced by the `assets_url` parameter (see [Template Parameters](#template-parameters)):

- **MANO_RIGHT.pkl**
- **BMC/**

See [`Isaac Teleop Documentation`](https://nvidia.github.io/IsaacCapture/main/references/egocentric_hand_reconstruction.html) for detailed setup instructions.

### Container images

The workflow requires two container images (`vipe_image` and `dynhamr_image`). Build them locally following the instructions in [`Isaac Teleop Documentation`](https://nvidia.github.io/IsaacCapture/main/references/egocentric_hand_reconstruction.html):

```bash
./docker/vipe.sh build
./docker/dynhamr.sh build
```

Then push the images to your container registry so OSMO can pull them at runtime:

```bash
docker tag ego_vipe:latest CONTAINER_REGISTRY/ego_vipe:TAG
docker tag ego_dynhamr:latest CONTAINER_REGISTRY/ego_dynhamr:TAG

docker push CONTAINER_REGISTRY/ego_vipe:TAG
docker push CONTAINER_REGISTRY/ego_dynhamr:TAG
```

### OSMO

This workflow runs on [OSMO](https://github.com/NVIDIA/OSMO). You need a working OSMO deployment and the OSMO CLI installed.

#### Deploy OSMO

Follow the [official deployment guide](https://nvidia.github.io/OSMO/main/deployment_guide/getting_started/infrastructure_setup.html) to set up your own OSMO deployment.

For a quick single-machine setup, use the [Brev launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-36a6a7qnkOMOP2vgiBRaw2e3jpW). Provision L40S or L40 machines with the Hyperstack provider.

##### Using the OSMO CLI locally with a Brev deployment

To interact with a Brev-hosted OSMO instance from your local machine:

1. **Install the Brev CLI** per the [official guide](https://docs.nvidia.com/brev/latest/cli/getting-started).

2. **Forward the OSMO port** from your Brev instance (OSMO listens on port 8000):
   ```bash
   brev port-forward YOUR_BREV_INSTANCE --port LOCAL_PORT:8000
   ```

3. **Add a hosts entry** so the CLI and browser can resolve the OSMO hostname:
   ```bash
   echo "127.0.0.1 quick-start.osmo" | sudo tee -a /etc/hosts
   ```

4. **Install the OSMO CLI**:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash
   ```

5. **Log in**:
   ```bash
   osmo login http://quick-start.osmo:LOCAL_PORT --method=dev --username=testuser
   ```

See your Brev instance page for additional Brev CLI commands.

#### Store credentials in OSMO

OSMO needs credentials to access the S3 buckets (source data, output destination, assets) and the container image registry. Register them with `osmo credential set`:

```bash
# S3 bucket access
osmo credential set BUCKET_CREDENTIAL \
    --type DATA \
    --payload \
        endpoint=BUCKET_URL \
        region=us-east-1 \
        access_key_id=ACCESS_KEY_ID \
        access_key=ACCESS_KEY

# Container registry access
osmo credential set REGISTRY_CREDENTIAL \
    --type REGISTRY \
    --payload \
        registry=REGISTRY_URL \
        username=USERNAME \
        auth=API_KEY
```

See the [OSMO credentials documentation](https://nvidia.github.io/OSMO/main/user_guide/getting_started/credentials.html) for details.

## Template Parameters

The workflow uses `{{placeholder}}` template variables that are filled at submission time via `--set-string`:

| Parameter | Required | Description |
|---|---|---|
| `workflow_name` | No | Unique name for the workflow instance. Defaults to `hand-reconstruction`. |
| `experiment_id` | Yes | Folder name of the experiment to process. Must contain at least one `.mp4` file. Appended to `source_url` to form the input path and used to namespace the output under `dest_url`. |
| `source_url` | Yes | S3 URL prefix for source data. The experiment is fetched from `<source_url>/<experiment_id>`. |
| `dest_url` | Yes | S3 URL where outputs are uploaded. Results land at `<dest_url>/<experiment_id>/…` |
| `assets_url` | Yes | S3 URL for shared reconstruction assets. Must contain `MANO_RIGHT.pkl` and a `BMC/` directory. |
| `vipe_image` | Yes | Container image for the ViPE inference task. |
| `dynhamr_image` | Yes | Container image for the Dyn-HaMR reconstruction task. |

## Usage

### Submit a single experiment

```bash
osmo workflow submit hand_reconstruction.yaml \
    --pool POOL_NAME \
    --set-string \
        workflow_name=hand-rec-my-experiment \
        experiment_id=EXPERIMENT_ID \
        source_url=s3://INPUT_S3_PATH \
        dest_url=s3://OUTPUT_S3_PATH \
        assets_url=s3://ASSETS_S3_PATH \
        vipe_image=CONTAINER_REGISTRY/ego_vipe:TAG \
        dynhamr_image=CONTAINER_REGISTRY/ego_dynhamr:TAG
```

### Monitor a running workflow

```bash
osmo workflow logs WORKFLOW_ID -n 100
```

This streams the trailing 100 lines and continues to follow new output.

### Check workflow status

Browse the OSMO dashboard or use:

```bash
osmo workflow list --name hand-rec --status RUNNING
```

## Pipeline Data Flow

```
source_url/experiment_id/
  └── *.mp4                          ← input video
  └── *.hdf5                         ← optional sensor data (not used)

        ┌─────────────┐
        │  vipe-infer │  → ViPE pose estimates
        └──────┬──────┘
               │
    ┌──────────┴──────────┐
    │ dynhamr-reconstruct │  + assets_url (MANO_RIGHT.pkl, BMC/)
    └──────────┬──────────┘
               │
               ▼
dest_url/experiment_id/
  ├── logs/                          ← Dyn-HaMR reconstruction logs
  │   └── video-custom/<date>/<run>/
  │       ├── smooth_fit/*_world_results.npz
  │       ├── cameras.json
  │       ├── track_info.json
  │       ├── *_smooth_fit*grid*.mp4
  │       ├── *_src_cam.mp4
  │       └── qc_results/
  ├── vipe_results/                  ← ViPE output
  ├── *.hdf5                         ← sensor data (passthrough)
  ├── video_file.txt
  └── video_name.txt
```
