#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EXPERIMENTAL, do not merge: build wheels against CUDA 13.4 inside the CUDA 13.3 ci-wheel image.
# RAPIDS does not publish CUDA 13.4 CI images yet, so this installs the 13.4 compiler, libraries and
# tileiras from NVIDIA's RHEL8 repo and points the build at them. Source it before generating
# requirements; it is a no-op for CUDA 12 jobs.

if [[ "${RAPIDS_CUDA_VERSION%%.*}" != "13" ]]; then
  return 0
fi

rapids-logger "Installing CUDA 13.4 toolkit and tileiras (experimental, image has CUDA ${RAPIDS_CUDA_VERSION})"

sudo_cmd=()
if [[ "$(id -u)" -ne 0 ]]; then
  sudo_cmd=(sudo)
fi

if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/yum.repos.d/; then
  repo_arch="x86_64"
  if [[ "$(arch)" == "aarch64" ]]; then
    repo_arch="sbsa"
  fi
  "${sudo_cmd[@]}" dnf install -y dnf-plugins-core
  "${sudo_cmd[@]}" dnf config-manager --add-repo \
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${repo_arch}/cuda-rhel8.repo"
fi

"${sudo_cmd[@]}" dnf install -y --setopt=install_weak_deps=False \
  cuda-compiler-13-4 \
  cuda-driver-devel-13-4 \
  cuda-libraries-devel-13-4 \
  cuda-nvtx-13-4 \
  cuda-profiler-api-13-4 \
  cuda-tileiras-13-4

export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Selects the CUDA 13.4 entries of dependencies.yaml; artifact names and the wheel suffix only use
# the major version, so they stay cuda13 / cu13.
export RAPIDS_CUDA_VERSION="13.4.0"

nvcc --version
tileiras --version || true
