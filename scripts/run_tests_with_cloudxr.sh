#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test Runner Script with CloudXR
# Runs Isaac Teleop tests using docker-compose with CloudXR runtime
#
# Usage:
#   ./scripts/run_tests_with_cloudxr.sh [--build] [--python-version <version>]
#
# Options:
#   --build           Force rebuild of test container (no cache)
#   --python-version  Python version for test container (e.g. 3.11)
#   --help            Show this help message

set -euo pipefail

#==============================================================================
# Configuration - Edit these lists to add/remove tests
#
# Note: Only tests will be executed with the CloudXR runtime on a runner with
# GPU support. If the tests doesn't require GPU support, please add them to
# the ctest configuration in `build-ubuntu.yml` instead, so that they can be
# run on a runner without GPU support immediately after the build.
#==============================================================================
# Paths are relative to /app/tests in the test image, which is a copy of
# examples/oxr/python/ -- so they carry the namespace prefix.
CXR_PYTHON_GPU_TESTS=(
    "isaaccapture_examples/oxr/test_package_version.py"
    "isaaccapture_examples/oxr/test_extensions.py"
    "isaaccapture_examples/oxr/test_modular.py"
)

CXR_NATIVE_GPU_TESTS=(
    "install/examples/native_openxr/xdev_list/xdev_list"
    "install/examples/oxr/cpp/oxr_session_sharing"
    "install/examples/oxr/cpp/oxr_simple_api_demo"
)
#==============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Make sure to run this script from the root of the repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$GIT_ROOT" || exit 1

# Default values
FORCE_BUILD=false
EXIT_CODE=0
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Compose files: shared runtime base + test overrides
COMPOSE_RUNTIME="deps/cloudxr/docker-compose.runtime.yaml"
COMPOSE_TEST="deps/cloudxr/docker-compose.test.yaml"

# Use a different project name to isolate volumes from run_cloudxr.sh
COMPOSE_PROJECT="isaacteleop-test"

# Environment file for test configuration
ENV_TEST="deps/cloudxr/.env.test"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --python-version)
            if [[ -z "${2:-}" ]]; then
                echo -e "${RED}--python-version requires a value${NC}"
                exit 1
            fi
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Test Runner Script with CloudXR"
            echo ""
            echo "Usage: $0 [--build] [--python-version <version>]"
            echo ""
            echo "Options:"
            echo "  --build           Force rebuild of test container (no cache)"
            echo "  --python-version  Python version for test container (e.g. 3.11)"
            echo "  --help            Show this help message"
            echo ""
            echo "Tests to run (edit CXR_PYTHON_GPU_TESTS/CXR_NATIVE_GPU_TESTS in this script):"
            echo ""
            echo "Python tests:"
            for test in "${CXR_PYTHON_GPU_TESTS[@]}"; do
                echo "  - $test"
            done
            echo ""
            echo "Native tests:"
            for test in "${CXR_NATIVE_GPU_TESTS[@]}"; do
                echo "  - $test"
            done
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
echo ""
echo -e "${BLUE}=========================================="
echo "  Isaac Teleop Test Runner with CloudXR"
echo -e "==========================================${NC}"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

# CSV-mode hosts need the NVIDIA runtime; legacy GPU hooks also work with runc.
# Match keys: Go templates treat a missing runtime's zero-valued struct as true.
DOCKER_RUNTIMES=$(docker info --format '{{range $name, $_ := .Runtimes}}{{println $name}}{{end}}')
CXR_TEST_DOCKER_RUNTIME=runc
if grep -Fxq nvidia <<< "$DOCKER_RUNTIMES"; then
    CXR_TEST_DOCKER_RUNTIME=nvidia
fi
export CXR_TEST_DOCKER_RUNTIME
log_info "Test Docker runtime: $CXR_TEST_DOCKER_RUNTIME"

# Set up environment
log_info "Setting up environment..."

# Source shared CloudXR environment setup, which defines ENV_DEFAULT and ENV_LOCAL
source scripts/setup_cloudxr_env.sh

# Cleanup function - ensures containers are stopped
cleanup_containers() {
    log_info "Cleaning up containers..."
    docker compose \
        -p "$COMPOSE_PROJECT" \
        --env-file "$ENV_DEFAULT" \
        ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
        ${ENV_TEST:+--env-file "$ENV_TEST"} \
        -f "$COMPOSE_RUNTIME" \
        -f "$COMPOSE_TEST" \
        down -v --remove-orphans 2>/dev/null || true
    log_success "Cleanup complete"
}

# Set up trap to ensure cleanup on exit
trap cleanup_containers EXIT

log_info "CXR_PYTHON_GPU_TESTS:"
for test in "${CXR_PYTHON_GPU_TESTS[@]}"; do
    log_info "  - $test"
done
log_info "CXR_NATIVE_GPU_TESTS:"
for test in "${CXR_NATIVE_GPU_TESTS[@]}"; do
    log_info "  - $test"
done
log_info "Test container Python version: $PYTHON_VERSION"

# Compose interpolation reads environment variables, not shell locals.
export PYTHON_VERSION
export CXR_BUILD_CONTEXT="$GIT_ROOT"

# Join arrays into comma-separated strings for docker-compose environment variables
CXR_PYTHON_GPU_TESTS_ENV=$(IFS=','; echo "${CXR_PYTHON_GPU_TESTS[*]}")
export CXR_PYTHON_GPU_TESTS="$CXR_PYTHON_GPU_TESTS_ENV"

CXR_NATIVE_GPU_TESTS_ENV=$(IFS=','; echo "${CXR_NATIVE_GPU_TESTS[*]}")
export CXR_NATIVE_GPU_TESTS="$CXR_NATIVE_GPU_TESTS_ENV"

# Set CloudXR runtime network mode
export CXR_RUNTIME_NETWORK_MODE="bridge"

# Create/update .env file with test configuration
log_info "Writing test configuration to $ENV_TEST..."
if [[ "${CI:-false}" = "true" ]]; then
    log_info "CI environment detected, auto-accepting CloudXR EULA"
fi
{
    echo "CXR_PYTHON_GPU_TESTS=$CXR_PYTHON_GPU_TESTS"
    echo "CXR_NATIVE_GPU_TESTS=$CXR_NATIVE_GPU_TESTS"
    # In CI, auto-accept EULA
    if [[ "${CI:-false}" = "true" ]]; then
        echo "ACCEPT_CLOUDXR_EULA=Y"
    fi
} > "$ENV_TEST"

# Verify install directory exists and has required artifacts
log_info "Verifying build artifacts..."

if [[ ! -d "install/wheels" ]]; then
    log_error "install/wheels not found. Please build first:"
    echo "  cmake -B build"
    echo "  cmake --build build --parallel"
    echo "  cmake --install build"
    exit 1
fi

WHEEL_COUNT=$(find install/wheels -name "isaaccapture-*.whl" | wc -l)
if [[ "$WHEEL_COUNT" -ne 1 ]]; then
    log_error "Expected exactly one isaaccapture wheel in install/wheels/, found $WHEEL_COUNT"
    exit 1
fi

log_success "Found $WHEEL_COUNT isaaccapture wheel(s) in install/wheels/"

# Select the exact CI wheel while resolving its transition dependency locally.
WHEEL_PATH=$(find install/wheels -name "isaaccapture-*.whl")
export ISAACTELEOP_PIP_SPEC="/workspace/${WHEEL_PATH}[cloudxr]"
export ISAACTELEOP_PIP_FIND_LINKS="/workspace/install/wheels"
export ISAACTELEOP_PIP_DEBUG=0
log_info "Using ISAACTELEOP_PIP_SPEC=$ISAACTELEOP_PIP_SPEC"
log_info "Using ISAACTELEOP_PIP_FIND_LINKS=$ISAACTELEOP_PIP_FIND_LINKS"
log_info "Using ISAACTELEOP_PIP_DEBUG=$ISAACTELEOP_PIP_DEBUG"

WHEEL_BASENAME=$(basename "$WHEEL_PATH")
EXPECTED_ISAACTELEOP_VERSION=$(echo "$WHEEL_BASENAME" | sed -E 's/^isaaccapture-([^-]+)-.*/\1/' | tr '_' '-')
if [[ -z "$EXPECTED_ISAACTELEOP_VERSION" ]]; then
    log_error "Failed to derive expected version from wheel name: $WHEEL_BASENAME"
    exit 1
fi
export EXPECTED_ISAACTELEOP_VERSION
log_info "Expected isaaccapture version from wheel artifact: $EXPECTED_ISAACTELEOP_VERSION"

# Build test container
log_info "Building test container..."

BUILD_ARGS="-q"
if [[ "$FORCE_BUILD" = true ]]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

docker build \
    $BUILD_ARGS \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    -t isaacteleop-tests:latest \
    -f deps/cloudxr/Dockerfile.test \
    .

log_success "Test container built successfully"

# Start CloudXR runtime services
log_info "Starting CloudXR runtime services..."

docker compose \
    -p "$COMPOSE_PROJECT" \
    --env-file "$ENV_DEFAULT" \
    ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
    ${ENV_TEST:+--env-file "$ENV_TEST"} \
    -f "$COMPOSE_RUNTIME" \
    -f "$COMPOSE_TEST" \
    up --build -d cloudxr-runtime

# Wait for CloudXR runtime to be healthy
log_info "Waiting for CloudXR runtime to be healthy..."

MAX_WAIT=60
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if docker compose \
        -p "$COMPOSE_PROJECT" \
        --env-file "$ENV_DEFAULT" \
        ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
        ${ENV_TEST:+--env-file "$ENV_TEST"} \
        -f "$COMPOSE_RUNTIME" \
        -f "$COMPOSE_TEST" \
        ps cloudxr-runtime 2>/dev/null | grep -q "healthy"; then
        log_success "CloudXR runtime is healthy"
        break
    fi

    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done
echo ""

if [[ $WAITED -ge $MAX_WAIT ]]; then
    log_error "CloudXR runtime failed to become healthy within ${MAX_WAIT}s"
    log_info "Container logs:"
    docker compose \
        -p "$COMPOSE_PROJECT" \
        --env-file "$ENV_DEFAULT" \
        ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
        ${ENV_TEST:+--env-file "$ENV_TEST"} \
        -f "$COMPOSE_RUNTIME" \
        -f "$COMPOSE_TEST" \
        logs cloudxr-runtime
    exit 1
fi

# Run tests
log_info "Running tests..."
echo ""

if docker compose \
    -p "$COMPOSE_PROJECT" \
    --env-file "$ENV_DEFAULT" \
    ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
    ${ENV_TEST:+--env-file "$ENV_TEST"} \
    -f "$COMPOSE_RUNTIME" \
    -f "$COMPOSE_TEST" \
    run --rm isaacteleop-tests; then
    log_success "All tests passed!"
    EXIT_CODE=0
else
    log_error "Some tests failed"
    EXIT_CODE=1
fi

# Output test results for CI
if [[ "${CI:-false}" = "true" ]]; then
    echo ""
    echo "::group::Container Logs"
    docker compose \
        -p "$COMPOSE_PROJECT" \
        --env-file "$ENV_DEFAULT" \
        ${ENV_LOCAL:+--env-file "$ENV_LOCAL"} \
        ${ENV_TEST:+--env-file "$ENV_TEST"} \
        -f "$COMPOSE_RUNTIME" \
        -f "$COMPOSE_TEST" \
        logs
    echo "::endgroup::"
fi

echo ""
echo -e "${BLUE}=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}  ✅ Tests Completed Successfully${NC}"
else
    echo -e "${RED}  ❌ Tests Failed${NC}"
fi
echo -e "${BLUE}==========================================${NC}"
echo ""

exit $EXIT_CODE
