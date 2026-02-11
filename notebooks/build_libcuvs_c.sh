#!/bin/bash
# Build libcuvs and pack into libcuvs_c.tar.gz
# Usage: ./build_libcuvs_c.sh [INSTALL_DIR]
# Examples:
#   ./build_libcuvs_c.sh                    # uses repo_root/cuvs_install
#   STATIC=1 ./build_libcuvs_c.sh           # enable DCUVS_STATIC_RAPIDS_LIBRARIES=ON
#
# This script lives in notebooks/; CUVS_REPO is auto-detected from parent dir.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUVS_REPO="${CUVS_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
INSTALL_DIR="${1:-${CUVS_REPO}/cuvs_install}"
STATIC="${STATIC:-0}"

cd "$CUVS_REPO"

CMAKE_EXTRA=()
[[ "${STATIC}" == "1" ]] && CMAKE_EXTRA+=(--cmake-args="-DCUVS_STATIC_RAPIDS_LIBRARIES=ON")

echo "Building libcuvs -> ${INSTALL_DIR}"
INSTALL_PREFIX="$INSTALL_DIR" ./build.sh libcuvs "${CMAKE_EXTRA[@]}"

echo "Creating libcuvs_c.tar.gz"
tar czvf libcuvs_c.tar.gz -C "$INSTALL_DIR" .
ls -lh libcuvs_c.tar.gz
