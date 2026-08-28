# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Defaults target ROS 2 Jazzy, whose system Python matches the wheel's floor.
# ROS_DISTRO and PYTHON_VERSION must be overridden together: rclpy is built
# against the distro's own interpreter.
ARG ROS_DISTRO=jazzy
ARG PYTHON_VERSION=3.12
ARG MANUS_ACCEPT_EULA=

# -----------------------------------------------------------------------------
# Base: common runtime deps + uv + Python (shared by builder and runtime)
# -----------------------------------------------------------------------------
FROM ros:${ROS_DISTRO}-ros-base AS base
ARG ROS_DISTRO
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libbz2-1.0 \
    libreadline8 \
    libsqlite3-0 \
    libffi8 \
    liblzma5 \
    zlib1g \
    libssl3 \
    libncursesw6 \
    libgdbm6 \
    libnss3 \
    libuuid1 \
    libexpat1 \
    libusb-1.0-0-dev \
    libudev-dev \
    libvulkan1 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libxrender-dev \
    libxfixes-dev \
    libxxf86vm-dev \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

# Install uv via standalone script (avoids PEP 668 "externally managed" pip on Ubuntu 24.04 / Jazzy)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN uv python install ${PYTHON_VERSION} --quiet \
    && uv python find ${PYTHON_VERSION}

# -----------------------------------------------------------------------------
# Stage 1: fetch Sharpa Wave URDF assets
# -----------------------------------------------------------------------------
FROM base AS sharpa_urdfs

WORKDIR /tmp/teleop_ros2
COPY examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py scripts/
COPY examples/teleop_ros2/configs/sharpa_wave_left_dexpilot.yml \
     examples/teleop_ros2/configs/sharpa_wave_right_dexpilot.yml \
     configs/
RUN python3 scripts/fetch_sharpa_wave_urdfs.py --output-dir /opt/sharpa-wave-urdfs

# -----------------------------------------------------------------------------
# Stage 2: build and install
# -----------------------------------------------------------------------------
FROM base AS builder
ARG ROS_DISTRO
ARG PYTHON_VERSION
ARG TARGETARCH
ARG MANUS_ACCEPT_EULA

# Compiler cache. CCACHE_DIR points at a BuildKit cache mount kept on the
# runner's local BuildKit storage (not the image, not GitHub Actions cache),
# so unchanged translation units are reused across builds. Bounded so it
# self-evicts LRU instead of growing without limit.
ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=5G

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    clang-format \
    cmake \
    curl \
    git \
    libvulkan-dev \
    patchelf \
    pkg-config \
    python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-ament-cmake-auto \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/isaaccapture
COPY . /opt/isaaccapture

# Remove any install/ or build/ copied in from the host context (an extracted
# build-ubuntu artifact tree can carry both). A stale install/wheels/ would
# pollute find-links with an isaaccapture wheel that is not the one this build
# produces; a stale build/ would skip a clean CMake reconfigure. The build tree
# is rebuilt from scratch every time -- ccache keeps recompiles cheap -- so each
# build reconfigures against the current image rather than trusting cached
# configure results. Ament's Python tools come from apt, so the interface
# package build below pins the ROS distro's system Python instead of uv's.
RUN --mount=type=cache,target=/ccache,id=isaacteleop-teleop-ros2-ccache-${ROS_DISTRO}-${TARGETARCH} \
    rm -rf /opt/isaaccapture/install /opt/isaaccapture/build \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/isaaccapture/install \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_EXAMPLE_TELEOP_ROS2=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_VIZ=OFF \
        -DBUILD_PLUGINS=ON \
        -DBUILD_PLUGIN_OAK_CAMERA=OFF \
        -DENABLE_CLANG_FORMAT_CHECK=OFF \
        -DENABLE_CLOUDXR_BUNDLE_CHECK=ON \
        -DISAAC_TELEOP_PYTHON_VERSION=${PYTHON_VERSION} \
    && cmake --build build --config Release -j$(nproc) \
    && cmake --install build --config Release \
    && bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash \
        && colcon build \
            --base-paths examples/teleop_ros2/teleop_ros2_interfaces \
            --build-base build/teleop_ros2_interfaces \
            --install-base /opt/isaaccapture/install \
            --merge-install \
            --cmake-args \
                -DCMAKE_BUILD_TYPE=Release \
                -DPython3_EXECUTABLE=/usr/bin/python3" \
    && ccache --show-stats

RUN --mount=type=cache,target=/ccache,id=isaacteleop-teleop-ros2-ccache-${ROS_DISTRO}-${TARGETARCH} \
    src/plugins/wuji_glove/install.sh --build-dir /opt/isaaccapture/build \
    && if [ "${MANUS_ACCEPT_EULA}" = "YES" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        src/plugins/manus/install_manus.sh --container \
            --build-dir /opt/isaaccapture/build; \
    fi \
    && ccache --show-stats

RUN set -eu; \
    wuji_plugin=/opt/isaaccapture/install/plugins/wuji_glove/wuji_glove_plugin; \
    test -x "$wuji_plugin"; \
    test -f /opt/isaaccapture/install/plugins/wuji_glove/plugin.yaml; \
    ldd_output="$(ldd "$wuji_plugin")"; \
    printf '%s\n' "$ldd_output" | awk '/not found/ { missing=1 } END { exit missing ? 1 : 0 }'; \
    if [ "${MANUS_ACCEPT_EULA}" = "YES" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        manus_plugin=/opt/isaaccapture/install/plugins/manus/manus_hand_plugin; \
        test -x "$manus_plugin"; \
        test -f /opt/isaaccapture/install/plugins/manus/plugin.yaml; \
        ldd_output="$(ldd "$manus_plugin")"; \
        printf '%s\n' "$ldd_output" | awk '/not found/ { missing=1 } END { exit missing ? 1 : 0 }'; \
    fi

COPY --chmod=644 <<EOF /usr/local/bin/teleop-env-setup
source /opt/ros/${ROS_DISTRO}/setup.bash
if [[ -f /opt/isaaccapture/install/local_setup.bash ]]; then
    source /opt/isaaccapture/install/local_setup.bash
fi
EOF

COPY --chmod=755 <<'EOF' /usr/local/bin/teleop-entrypoint
#!/bin/bash
set -e
source /usr/local/bin/teleop-env-setup
exec "$@"
EOF

# -----------------------------------------------------------------------------
# Stage 3: runtime base (install tree only)
# -----------------------------------------------------------------------------
FROM base AS runtime_base
ARG PYTHON_VERSION
ARG TARGETARCH
ARG MANUS_ACCEPT_EULA

# install_manus.sh supplies development headers in the builder; the final
# image only needs the shared library used by libManusSDK.so.
RUN if [ "${MANUS_ACCEPT_EULA}" = "YES" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libzmq5 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /opt/isaaccapture/install/examples/teleop_ros2

# The node launches the CloudXR runtime in-process via CloudXRLauncher, so this
# image needs GPU access and the NVIDIA Vulkan/EGL ICDs. Run with --gpus all.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
COPY deps/cloudxr/runtime/data/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
COPY deps/cloudxr/runtime/data/nvidia_icd.json /etc/vulkan/icd.d/
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Pre-install heavy Python dependencies so they are cached independently of C++ builds.
# Copy requirements from the context, avoiding any files from the 'builder' stage.
# Base requirements include the local transition package; final uv sync installs them.
COPY src/core/python/requirements-retargeters.txt \
     src/core/python/requirements-grounding.txt \
     src/core/python/requirements-wuji.txt \
     src/core/python/requirements-cloudxr.txt \
     /tmp/
RUN uv venv --python=${PYTHON_VERSION} && \
    uv pip install setuptools wheel && \
    uv pip install \
        -r /tmp/requirements-retargeters.txt \
        -r /tmp/requirements-grounding.txt \
        -r /tmp/requirements-wuji.txt \
        -r /tmp/requirements-cloudxr.txt \
        msgpack msgpack-numpy && \
    rm /tmp/requirements-retargeters.txt \
       /tmp/requirements-grounding.txt \
       /tmp/requirements-wuji.txt \
       /tmp/requirements-cloudxr.txt

COPY --from=builder /opt/isaaccapture/install /opt/isaaccapture/install
COPY --from=sharpa_urdfs /opt/sharpa-wave-urdfs/ /opt/isaaccapture/install/examples/teleop_ros2/assets/urdf/sharpa_standalone/
COPY --from=builder /usr/local/bin/teleop-env-setup /usr/local/bin/teleop-env-setup
COPY --from=builder /usr/local/bin/teleop-entrypoint /usr/local/bin/teleop-entrypoint

ENV ISAAC_TELEOP_PLUGIN_PATH=/opt/isaaccapture/install/plugins
ENV BASH_ENV=/usr/local/bin/teleop-env-setup

RUN echo "source /usr/local/bin/teleop-env-setup" >> /etc/bash.bashrc

# Pre-install Python dependencies so uv run doesn't download at every launch.
# Pin isaaccapture to the exact version of the locally-built wheel so the
# resolver cannot prefer a published PyPI release (whose dep metadata may
# diverge from what this image was actually built with). Extract the version
# from the wheel filename rather than hard-coding it.
RUN set -eu; \
    echo "== isaaccapture wheels found in find-links =="; \
    ls -la /opt/isaaccapture/install/wheels/ || true; \
    wheels=$(ls /opt/isaaccapture/install/wheels/isaaccapture-*.whl 2>/dev/null); \
    count=$(printf '%s\n' "$wheels" | grep -c .); \
    if [ "$count" -ne 1 ]; then \
        echo "ERROR: expected exactly 1 isaaccapture wheel in find-links, found $count:"; \
        printf '  %s\n' $wheels; \
        exit 1; \
    fi; \
    wheel=$wheels; \
    version=$(basename "$wheel" | sed -E 's/^isaaccapture-([^-]+)-.*$/\1/'); \
    echo "== pinning isaaccapture==${version} =="; \
    uv sync --no-dev --upgrade-package "isaaccapture==${version}"

# Test-only image: keep the production runtime free of source tests.
FROM runtime_base AS test
COPY tests/python/examples/teleop_ros2/ tests/

# Production image.
FROM runtime_base AS runtime

# Use --no-sync because the venv is already provisioned above.
ENTRYPOINT ["/usr/local/bin/teleop-entrypoint", "uv", "run", "--no-sync", \
            "python", "-m", "isaaccapture_examples.teleop_ros2"]
