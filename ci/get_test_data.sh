#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
