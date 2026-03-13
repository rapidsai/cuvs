#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuvs/cuvs

PYTHONUNBUFFERED=1 pytest --cache-clear -vs --deselect tests/test_hnsw_ace.py::test_hnsw_ace_tiny_memory_limit_triggers_disk_mode --deselect tests/test_cagra_ace.py::test_cagra_ace_tiny_memory_limit_triggers_disk_mode "$@" tests
