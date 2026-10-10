#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: quick compute-sanitizer pass on the host-input product quantizer
# tests (H100 CAGRA hang investigation, #2783).

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate Python testing dependencies (nightly packages)"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
rapids-mamba-retry create --yes -n cudagdb -c conda-forge "cuda-sanitizer-api=12.9" "cuda-version=12.9" || true

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env
nvidia-smi
nvidia-smi -q | grep -iE "Addressing Mode|Driver Version|Product Name|Bus Id" || true

DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}"
CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
CI_DIR="${PWD}/ci"
export PYTHONPATH="${CI_DIR}:${PYTHONPATH:-}"

set +e

pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
python -m pytest --collect-only -q -p no:cacheprovider tests/test_product_quantizer.py 2>/dev/null \
  | grep "::" | sed "s|^cuvs/||" > "${DIAG_DIR}/pq_all.txt"
grep -E "test_product_quantizer\[False-" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_host.txt"
grep -F "test_product_quantizer[False-True-True-kmeans-True-7-64-700]" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_one.txt"
grep -E "test_product_quantizer\[False-.*-64-700\]" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_host_small.txt"
wc -l "${DIAG_DIR}"/pq_*.txt
cat "${DIAG_DIR}/pq_one.txt"
popd > /dev/null || exit 1

SANITIZER=$(find /opt/conda/envs/cudagdb -name compute-sanitizer -type f -perm -u+x 2>/dev/null | head -1)
echo "compute-sanitizer: ${SANITIZER}"
"${SANITIZER}" --version || true

postcheck() {
  echo "---- post-check in a fresh process"
  timeout 300 python "${CI_DIR}/b2_postcheck.py" 2>&1 | tail -5
}

# run_sanitizer <name> <timeout> <tool args...> -- <test id file>
run_sanitizer() {
  local name=$1 tmo=$2
  shift 2
  local args=()
  while [[ $1 != "--" ]]; do args+=("$1"); shift; done
  shift
  local ids=$1
  local out="${DIAG_DIR}/san-${name}.log"
  local start
  start=$(date +%s)
  rapids-logger "SANITIZER ${name}: ${args[*]} on $(wc -l < "${ids}") tests"
  pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
  # shellcheck disable=SC2046
  timeout "${tmo}" "${SANITIZER}" --target-processes all --print-limit 200 --error-exitcode 0 \
    "${args[@]}" python -m pytest -v -p no:cacheprovider -p hang_sync_plugin $(cat "${ids}") > "${out}" 2>&1
  local rc=$?
  popd > /dev/null || exit 1
  echo "sanitizer ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  echo "---- error records (first 300 '=========' lines)"
  grep -a "=========" "${out}" | head -300
  echo "---- test results and summary"
  grep -aE "PASSED|FAILED|ERROR|hang-sync|passed|failed" "${out}" | cut -c1-200 | tail -40
  postcheck
}

postcheck
run_sanitizer memcheck-one 15m --tool memcheck --leak-check no --padding 32 -- "${DIAG_DIR}/pq_one.txt"
run_sanitizer initcheck-one 15m --tool initcheck -- "${DIAG_DIR}/pq_one.txt"
run_sanitizer memcheck-host 25m --tool memcheck --leak-check no --padding 32 -- "${DIAG_DIR}/pq_host.txt"
run_sanitizer initcheck-host-small 15m --tool initcheck -- "${DIAG_DIR}/pq_host_small.txt"
exit 0
