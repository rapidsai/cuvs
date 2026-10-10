#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: find which test_product_quantizer case leaves the GPU in a state
# that makes a following test_cagra.py hang on H100 (#2772, #2732, #2783).
# Phase 1 (one role per job): compute-sanitizer / per-test sync on the PQ tests.
# Phase 2 (all jobs): bisect buckets of PQ tests, each followed by test_cagra.py
# in the same process, with stall detection.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

SCRIPT_START=$(date +%s)

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate Python testing dependencies (nightly packages)"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

rapids-logger "Create debugging tools environments"
rapids-mamba-retry create --yes -n dbg -c conda-forge gdb py-spy || true
rapids-mamba-retry create --yes -n cudagdb -c conda-forge "cuda-sanitizer-api=12.9" "cuda-version=12.9" || true
DBG_BIN=/opt/conda/envs/dbg/bin

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "System info"
nvidia-smi
nvidia-smi -q | grep -iE "Addressing Mode|Driver Version|Product Name|Bus Id" || true
nproc
lscpu | grep -E "Model name" || true
echo "RUNNER_NAME=${RUNNER_NAME:-unset} HOSTNAME=$(hostname) GITHUB_RUN_ID=${GITHUB_RUN_ID:-} GITHUB_JOB=${GITHUB_JOB:-}"

# Work out which copy of the job this is, so each copy runs a different sanitizer.
ROLE=""
if command -v gh > /dev/null && [[ -n "${GH_TOKEN:-}" ]]; then
  for _ in 1 2 3; do
    ROLE=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/attempts/${GITHUB_RUN_ATTEMPT:-1}/jobs?per_page=100" \
      --jq "[.jobs[] | select(.name | test(\"h100\"))] | sort_by(.id) | map(.runner_name) | index(\"${RUNNER_NAME:-none}\")" 2>/dev/null || true)
    [[ -n "${ROLE}" && "${ROLE}" != "null" ]] && break
    ROLE=""
    sleep 10
  done
fi
if [[ -z "${ROLE}" ]]; then
  ROLE=$(( $(hostname | cksum | cut -d' ' -f1) % 4 ))
  echo "ROLE from hostname hash"
fi
ROLE=$(( ROLE % 4 ))
rapids-logger "This job has ROLE=${ROLE}"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"
DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}"
CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
CI_DIR="${PWD}/ci"
export PYTHONPATH="${CI_DIR}:${PYTHONPATH:-}"
PYTEST_ARGS=(-v -p no:cacheprovider -o faulthandler_timeout=240 -p hang_sync_plugin)

descendants() {
  local p=$1
  echo "$p"
  for c in $(pgrep -P "$p" 2>/dev/null); do
    descendants "$c"
  done
}

thread_times() {
  local p=$1
  for t in /proc/"${p}"/task/*; do
    stat=$(cat "$t"/stat 2>/dev/null) || continue
    rest=${stat##*) }
    # shellcheck disable=SC2086
    set -- $rest
    echo "$(basename "$t") $(cat "$t"/comm 2>/dev/null) state=$1 utime=${12} stime=${13} wchan=$(cat "$t"/wchan 2>/dev/null)"
  done
}

dump_diag() {
  local root=$1
  local pids p
  pids=$(descendants "${root}")
  echo "########## HANG DIAGNOSTICS $(date -u +%FT%TZ) root=${root} pids: ${pids}"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv || true
  for p in ${pids}; do
    thread_times "${p}" | sort -t= -k4 -n -r | head -5
    if grep -q python /proc/"${p}"/cmdline 2>/dev/null; then
      echo "---- py-spy dump --native ${p}"
      timeout 120 "${DBG_BIN}/py-spy" dump --native --pid "${p}" 2>&1 | head -45 || true
    fi
  done
  echo "########## END HANG DIAGNOSTICS"
}

# run_watched <name> <stall_seconds> <cmd...>; returns 124 on a stall.
run_watched() {
  local name=$1 stall=$2
  shift 2
  local log="${DIAG_DIR}/${name}.log"
  local start
  start=$(date +%s)
  echo ">>>>> START ${name}"
  "$@" > "${log}" 2>&1 &
  local pid=$!
  local last_size=-1 last_change now size hung=0
  last_change=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
    size=$(stat -c %s "${log}" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [[ "${size}" != "${last_size}" ]]; then
      last_size=${size}
      last_change=${now}
    elif (( now - last_change > stall )); then
      hung=1
      break
    fi
  done
  local rc=0
  if (( hung )); then
    echo "!!!!! HANG DETECTED in ${name} after $(( $(date +%s) - start ))s (no output for ${stall}s)"
    grep -aE "PASSED|FAILED|SKIPPED|ERROR|hang-sync" "${log}" | tail -5
    tail -n 3 "${log}"
    dump_diag "${pid}"
    for p in $(descendants "${pid}"); do kill -9 "${p}" 2>/dev/null || true; done
    wait "${pid}" 2>/dev/null
    rc=124
    sleep 5
  else
    wait "${pid}"
    rc=$?
  fi
  # Compact log: failures, sync reports and the summary line.
  grep -aE "FAILED|ERROR |hang-sync|Timeout|Error|error:|passed|failed" "${log}" | grep -v "PASSED" | tail -40 || true
  echo "<<<<< END ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  return ${rc}
}

set +e

pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
python -m pytest --collect-only -q -p no:cacheprovider tests/test_product_quantizer.py 2>/dev/null \
  | grep "::" | sed "s|^cuvs/||" > "${DIAG_DIR}/pq_all.txt"
grep -E "test_product_quantizer\[False-" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_host.txt"
grep -E "test_product_quantizer\[True-" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_dev.txt"
grep -E "test_recall|test_extreme_cases|test_pq_dataset_params" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_recall.txt"
grep -E "test_product_quantizer\[False-.*-64-700\]" "${DIAG_DIR}/pq_all.txt" > "${DIAG_DIR}/pq_host_small.txt"
wc -l "${DIAG_DIR}"/pq_*.txt
head -3 "${DIAG_DIR}/pq_host.txt"
popd > /dev/null || exit 1

SANITIZER=$(find /opt/conda/envs/cudagdb -name compute-sanitizer -type f -perm -u+x 2>/dev/null | head -1)
echo "compute-sanitizer: ${SANITIZER}"
"${SANITIZER}" --version || true

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
  echo "---- error records (first 300 lines of '=========' output, excluding the banner)"
  grep -a "=========" "${out}" | grep -v "COMPUTE-SANITIZER" | head -300
  echo "---- summary lines"
  grep -aE "ERROR SUMMARY|RACECHECK SUMMARY|hang-sync|passed|failed" "${out}" | tail -20
  echo "---- post-check in a fresh process"
  timeout 300 python "${CI_DIR}/b2_postcheck.py" 2>&1 | tail -5
}


# One bucket per GPU (the poison may persist across processes, so buckets are
# never mixed on one GPU). Sanitizer passes that run PQ cases from other buckets
# are done after the bisect loop.
BUCKETS=(host recall dev none)
BUCKET=${BUCKETS[${ROLE}]}

phase1_pre() {
  case "${ROLE}" in
    0)
      run_sanitizer memcheck-host 60m --tool memcheck --leak-check no --padding 32 -- "${DIAG_DIR}/pq_host.txt"
      run_sanitizer initcheck-host-small 25m --tool initcheck -- "${DIAG_DIR}/pq_host_small.txt"
      ;;
    1)
      run_sanitizer memcheck-recall 30m --tool memcheck --leak-check no --padding 32 -- "${DIAG_DIR}/pq_recall.txt"
      ;;
  esac
}

phase1_post() {
  case "${ROLE}" in
    0)
      run_sanitizer synccheck-host 30m --tool synccheck -- "${DIAG_DIR}/pq_host.txt"
      ;;
    1)
      run_sanitizer racecheck-host-small 40m --tool racecheck --racecheck-report all -- "${DIAG_DIR}/pq_host_small.txt"
      ;;
    2)
      pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
      HANG_SYNC_EACH=1 run_watched "sync-each-pq" 360 python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" tests/test_product_quantizer.py
      popd > /dev/null || exit 1
      timeout 300 python "${CI_DIR}/b2_postcheck.py" 2>&1 | tail -5
      run_sanitizer synccheck-all 45m --tool synccheck -- "${DIAG_DIR}/pq_all.txt"
      ;;
    3)
      run_sanitizer memcheck-all 70m --tool memcheck --leak-check no --padding 32 -- "${DIAG_DIR}/pq_all.txt"
      ;;
  esac
}

rapids-logger "Phase 1 (before bisect): ROLE=${ROLE} BUCKET=${BUCKET}"
phase1_pre

rapids-logger "Phase 2: bucket ${BUCKET} (PQ subset, then test_cagra.py in the same process), repeated"
case "${BUCKET}" in
  none) ids=() ;;
  host) mapfile -t ids < "${DIAG_DIR}/pq_host.txt" ;;
  dev) mapfile -t ids < "${DIAG_DIR}/pq_dev.txt" ;;
  recall) mapfile -t ids < "${DIAG_DIR}/pq_recall.txt" ;;
esac
PHASE2_END=$(( SCRIPT_START + 165 * 60 ))
n=0
RUNS=0
HANGS=0
FAILS=0
HANG_ITERS=""
SEQ=""
while (( $(date +%s) < PHASE2_END )); do
  n=$((n + 1))
  rapids-logger "Iteration ${n}: bucket=${BUCKET} (${#ids[@]} PQ tests) then test_cagra.py"
  pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
  run_watched "iter${n}-${BUCKET}" 360 python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" "${ids[@]}" tests/test_cagra.py
  rc=$?
  popd > /dev/null || exit 1
  RUNS=$((RUNS + 1))
  if (( rc == 124 )); then
    HANGS=$((HANGS + 1))
    HANG_ITERS="${HANG_ITERS} ${n}"
    SEQ="${SEQ} ${n}:HANG"
  elif (( rc != 0 )); then
    FAILS=$((FAILS + 1))
    SEQ="${SEQ} ${n}:rc${rc}"
  else
    SEQ="${SEQ} ${n}:ok"
  fi
  echo "B2-PROGRESS role=${ROLE} bucket=${BUCKET} seq:${SEQ}"
done

rapids-logger "Phase 1 (after bisect): ROLE=${ROLE}"
phase1_post

rapids-logger "B2 BISECT SUMMARY (ROLE=${ROLE}, runner ${RUNNER_NAME:-?})"
echo "B2-SUMMARY bucket=${BUCKET} runs=${RUNS} hangs=${HANGS} other_failures=${FAILS} hang_iterations:${HANG_ITERS}"
echo "B2-SEQUENCE ${SEQ}"
exit 0
