#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

cd "$(dirname "$0")/.."

RAPIDS_VERSION="$(rapids-version)"
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
RAPIDS_VERSION_PATCH="$(echo "$RAPIDS_VERSION" | awk -F. '{ print $3 }' | sed -E 's/^0*(.)$/\1/')"

echo "$RAPIDS_VERSION_MAJOR_MINOR.$RAPIDS_VERSION_PATCH"
