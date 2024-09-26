#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

./build.sh tests --allgpuarch --no-nvtx --build-metrics=tests --incl-cache-stats
cmake --install cpp/build --component testing
