#!/usr/bin/env bash
# Copyright (c) 2024, NVIDIA CORPORATION.

./build.sh bench-ann --allgpuarch --no-shared-libs --no-nvtx --build-metrics=bench_ann --incl-cache-stats
cmake --install cpp/build --component ann_bench
