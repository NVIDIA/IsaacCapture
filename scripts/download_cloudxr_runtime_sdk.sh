#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Downloads the CloudXR Runtime SDK if not already present using NGC.
# The SDK tarball (CloudXR-<VERSION>-Linux-<ARCH>-sdk.tar.gz) is placed in deps/cloudxr/
# for use by Dockerfile.runtime.
#
# Two ways to obtain the SDK:
# 1) Local tarball: place CloudXR-<VERSION>-Linux-<ARCH>-sdk.tar.gz in deps/cloudxr/.
# 2) Public NGC: RC versions use nvidia/cloudxr-runtime-for-isaac-teleop directly.
#    Other versions try nvidia/cloudxr-runtime first and fall back to the unlisted resource.
# Optional: set CXR_DOWNLOAD_EXP=1 to also download CloudXR-exp-<VERSION>-....

set -Eeuo pipefail

on_error() {
    local exit_code="$?"
    local line_no="$1"
    echo "Error: ${BASH_SOURCE[0]} failed at line ${line_no} (exit ${exit_code})" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

# Ensure we're in the git root
if [[ -z "${GIT_ROOT:-}" ]]; then
    GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
    if [[ -z "$GIT_ROOT" ]]; then
        echo "Error: Could not determine git root. Set GIT_ROOT before sourcing." >&2
        exit 1
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [[ -z "${CXR_RUNTIME_SDK_VERSION:-}" ]]; then
    echo -e "${RED}Error: CXR_RUNTIME_SDK_VERSION is not set${NC}"
    exit 1
fi

# SDK configuration (shared)
CXR_DEPLOYMENT_DIR="$GIT_ROOT/deps/cloudxr"

case "$(uname -m)" in
    x86_64)        ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) echo -e "${RED}Error: Unsupported architecture '$(uname -m)'.${NC}"; exit 1 ;;
esac

SDK_FILE="CloudXR-${CXR_RUNTIME_SDK_VERSION}-Linux-${ARCH}-sdk.tar.gz"
EXP_SDK_FILE="CloudXR-exp-${CXR_RUNTIME_SDK_VERSION}-Linux-${ARCH}-sdk.tar.gz"

# Remote names on NGC, newest convention first. 6.3.0-rc4 renamed the published
# tarballs to a "-external" infix; the payload is unchanged. Downloads are always
# saved as $SDK_FILE / $EXP_SDK_FILE, which is what CMake and Dockerfile.runtime-ngc
# expect on disk.
SDK_REMOTE_FILES=(
    "CloudXR-external-${CXR_RUNTIME_SDK_VERSION}-Linux-${ARCH}-sdk.tar.gz"
    "$SDK_FILE"
)
EXP_SDK_REMOTE_FILES=(
    "CloudXR-exp-external-${CXR_RUNTIME_SDK_VERSION}-Linux-${ARCH}-sdk.tar.gz"
    "$EXP_SDK_FILE"
)

is_valid_sdk_bundle() {
    local dir="$1"
    [[ -f "$dir/$SDK_FILE" ]] || return 1
    if [[ "${CXR_DOWNLOAD_EXP:-0}" == "1" ]]; then
        [[ -f "$dir/$EXP_SDK_FILE" ]] || return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Local tarball: place $SDK_FILE in deps/cloudxr/
# -----------------------------------------------------------------------------
install_from_local_tarball() {
    if ! is_valid_sdk_bundle "$CXR_DEPLOYMENT_DIR"; then
        return 1
    fi
    echo -e "${GREEN}✓ CloudXR Runtime SDK found at $CXR_DEPLOYMENT_DIR/$SDK_FILE${NC}"
    if [[ "${CXR_DOWNLOAD_EXP:-0}" == "1" ]]; then
        echo -e "${GREEN}✓ CloudXR Experimental Runtime SDK found at $CXR_DEPLOYMENT_DIR/$EXP_SDK_FILE${NC}"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# NGC download helper
# -----------------------------------------------------------------------------
download_ngc_file() {
    local url="$1"
    local out_path="$2"
    local label="$3"

    local -a curl_args=(--fail --location --output "$out_path"
        --connect-timeout 10 --max-time 120
        --retry 3 --retry-delay 5)
    if ! curl "${curl_args[@]}" "$url"; then
        rm -f "$out_path"
        return 1
    fi
    if [[ ! -s "$out_path" ]]; then
        echo -e "${RED}Error: Downloaded ${label} is empty${NC}"
        rm -f "$out_path"
        return 1
    fi
    # Record where this came from. The tarball is fetched, not checked in, so
    # without this the only evidence of its origin is a shell variable that
    # stops existing when the script exits -- and the SBOM has to call the
    # download location unknown.
    printf '%s\n' "$url" > "${out_path}.source"
}

# Try each remote name in turn; the first that resolves wins.
download_ngc_first_match() {
    local base="$1"
    local out_path="$2"
    local label="$3"
    shift 3
    local -a remote_files=("$@")

    [[ -s "$out_path" ]] && return 0

    local remote
    for remote in "${remote_files[@]}"; do
        echo -e "${YELLOW}Downloading ${label} (${remote})...${NC}"
        if download_ngc_file "${base}${remote}" "$out_path" "$label"; then
            return 0
        fi
    done

    echo -e "${RED}Error: Failed to download ${label}${NC}"
    echo -e "${RED}Tried: ${remote_files[*]}${NC}"
    return 1
}

# -----------------------------------------------------------------------------
# Public NGC is anonymous. RC versions go directly to the unlisted Isaac Teleop resource;
# non-RC versions try listed first and fall back to unlisted without credentials.
# -----------------------------------------------------------------------------
install_from_public_ngc() {
    local resource="$1"
    local visibility="$2"

    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl not found. Please install it first.${NC}"
        echo -e "To use a local SDK instead, place $SDK_FILE in deps/cloudxr/"
        return 1
    fi

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Downloading CloudXR Runtime SDK from ${visibility} NGC resource${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    mkdir -p "$CXR_DEPLOYMENT_DIR"

    local base="https://api.ngc.nvidia.com/v2/resources/org/nvidia/${resource}/${CXR_RUNTIME_SDK_VERSION}/files?redirect=true&path="
    download_ngc_first_match \
        "$base" \
        "$CXR_DEPLOYMENT_DIR/$SDK_FILE" \
        "CloudXR Runtime SDK" \
        "${SDK_REMOTE_FILES[@]}" || return 1
    if [[ "${CXR_DOWNLOAD_EXP:-0}" == "1" ]]; then
        download_ngc_first_match \
            "$base" \
            "$CXR_DEPLOYMENT_DIR/$EXP_SDK_FILE" \
            "CloudXR Experimental Runtime SDK" \
            "${EXP_SDK_REMOTE_FILES[@]}" || return 1
    fi

    echo -e "${GREEN}✓ CloudXR Runtime SDK installed successfully${NC}"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# One listed -> unlisted sweep over the public NGC resources.
install_from_ngc() {
    local resource visibility

    if [[ "$CXR_RUNTIME_SDK_VERSION" == *-rc* ]]; then
        resource="cloudxr-runtime-for-isaac-teleop"
        visibility="unlisted"
    else
        resource="cloudxr-runtime"
        visibility="listed"
    fi

    echo "Trying ${visibility} public NGC..."
    if install_from_public_ngc "$resource" "$visibility"; then
        return 0
    fi

    if [[ "$visibility" == "listed" ]]; then
        resource="cloudxr-runtime-for-isaac-teleop"
        visibility="unlisted"
        echo "Cannot install from listed public NGC, trying unlisted public NGC..."
        # Do not combine a partial listed download with the unlisted SDK bundle.
        rm -f "$CXR_DEPLOYMENT_DIR/$SDK_FILE" "$CXR_DEPLOYMENT_DIR/$EXP_SDK_FILE"
        if install_from_public_ngc "$resource" "$visibility"; then
            return 0
        fi
    fi

    return 1
}

# Prefer local tarball if present; otherwise use NGC
if install_from_local_tarball; then
    exit 0
fi

echo "Cannot install from local tarball, trying public NGC..."

# NGC 404s transiently, and curl's --retry never covers 4xx. Retry the whole
# sweep rather than each request: a 404 on an individual remote name is how
# download_ngc_first_match discovers the published name, so per-request retry
# would cost a delay on every successful run.
NGC_ATTEMPTS=3
NGC_RETRY_DELAY_S=5
for (( attempt = 1; attempt <= NGC_ATTEMPTS; attempt++ )); do
    if install_from_ngc; then
        exit 0
    fi
    if (( attempt < NGC_ATTEMPTS )); then
        echo -e "${YELLOW}NGC download failed (attempt ${attempt}/${NGC_ATTEMPTS}); retrying in ${NGC_RETRY_DELAY_S}s...${NC}"
        rm -f "$CXR_DEPLOYMENT_DIR/$SDK_FILE" "$CXR_DEPLOYMENT_DIR/$EXP_SDK_FILE"
        sleep "$NGC_RETRY_DELAY_S"
    fi
done

echo "Cannot install from public NGC after ${NGC_ATTEMPTS} attempts, exiting..."
exit 1
