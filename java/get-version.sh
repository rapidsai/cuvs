#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

cd "$(dirname "$0")/.."

RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

echo "$RAPIDS_VERSION_MAJOR_MINOR.$(python3 -c "from packaging.version import Version; import os; print(Version(os.getenv('RAPIDS_VERSION')).micro)")"
