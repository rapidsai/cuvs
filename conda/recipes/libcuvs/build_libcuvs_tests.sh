#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

./build.sh tests bench-ann --allgpuarch --no-nvtx --build-metrics=tests_bench --incl-cache-stats
cmake --install cpp/build --component testing
