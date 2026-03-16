#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuvs/cuvs

for i in $(seq 1 10); do
  echo "===== Run $i of 10 ====="
  PYTHONUNBUFFERED=1 pytest --cache-clear -vs "$@" tests
done
