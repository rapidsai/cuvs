#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NUMARGS=$#
ARGS=$*
function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "Usage:"
    echo "    $0 [--NEIGHBORS_ANN_VAMANA_TEST]"
    exit 0
fi

DEST=${RAPIDS_DATASET_ROOT_DIR:-"$PWD"}

# get test data for NEIGHBORS_ANN_VAMANA_TEST
if hasArg "--NEIGHBORS_ANN_VAMANA_TEST"; then
    echo "Downloading test data for NEIGHBORS_ANN_VAMANA_TEST"
    echo "Destination: ${DEST}"
    URL_PREFIX=https://data.rapids.ai/cuvs/tests/data
    SUBDIR=neighbors/ann_vamana/randomized_codebooks
    FILE_LIST=(
        384_int8_pq_pivots.bin
        384_int8_pq_pivots.bin_rotation_matrix.bin
        64_float_pq_pivots.bin
        64_float_pq_pivots.bin_rotation_matrix.bin
    )
    for f in "${FILE_LIST[@]}"; do
        wget --no-verbose --directory-prefix="${DEST}/${SUBDIR}" "${URL_PREFIX}/${SUBDIR}/$f"
    done
fi
