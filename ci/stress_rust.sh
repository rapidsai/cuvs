#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CI-only: repeat the Rust test suite with parallel test threads (issue #2233).
set -uo pipefail

STRESS_MINUTES=${STRESS_MINUTES:-110}
ENV_NAME=${ENV_NAME:-rust}
RUST_DIR=${RUST_DIR:-rust}

. /opt/conda/etc/profile.d/conda.sh
set +u
conda activate "${ENV_NAME}"
set -u

rapids-logger "Installing gdb and compute-sanitizer"
rapids-mamba-retry install -y -n "${ENV_NAME}" gdb "cuda-sanitizer-api=${RAPIDS_CUDA_VERSION%.*}" || \
  rapids-mamba-retry install -y -n "${ENV_NAME}" gdb || true
which gdb compute-sanitizer || true

nvidia-smi || true
nproc
cat /proc/sys/kernel/core_pattern || true
ulimit -c unlimited || true
ulimit -c

cd "${RUST_DIR}" || exit 1
export RUST_BACKTRACE=full
cargo test -p cuvs --lib --no-run 2>&1 | tee norun.log
TEST_BIN=$(sed -n 's/.*Executable unittests src\/lib.rs (\(.*\))/\1/p' norun.log | tail -1)
TEST_BIN=$(realpath "${TEST_BIN}")
echo "TEST_BIN=${TEST_BIN}"
"${TEST_BIN}" --list 2>/dev/null | tail -3
HAVE_STRESS=0
if "${TEST_BIN}" --list --ignored 2>/dev/null | grep -q "stress_"; then HAVE_STRESS=1; fi

LOGDIR=$(mktemp -d)
mkdir -p "${LOGDIR}/cores"
GDB_CMDS=(-q -batch -ex "set pagination off" -ex "set print thread-events off"
  -ex "handle SIGPIPE nostop noprint pass" -ex "handle SIG32 nostop noprint pass"
  -ex "handle SIG33 nostop noprint pass" -ex "handle SIG34 nostop noprint pass"
  -ex "handle SIGUSR1 nostop noprint pass" -ex "handle SIGUSR2 nostop noprint pass"
  -ex run -ex "info threads" -ex "thread apply all bt" -ex "info sharedlibrary")

THREADS_LIST=("" 8 16 32 64)
deadline=$(( $(date +%s) + STRESS_MINUTES * 60 ))
iter=0
fails=0
crashes=0
declare -A per_mode_runs per_mode_fails
while [ "$(date +%s)" -lt "${deadline}" ]; do
  iter=$((iter + 1))
  t=${THREADS_LIST[$((iter % ${#THREADS_LIST[@]}))]}
  targs=()
  [ -n "${t}" ] && targs=(--test-threads="${t}")
  mode="native"
  if (( iter % 4 == 0 )); then mode="gdb"; fi
  if (( iter % 6 == 0 )) && (( HAVE_STRESS )); then mode="stress"; fi
  key="${mode}-t${t:-default}"
  per_mode_runs[$key]=$(( ${per_mode_runs[$key]:-0} + 1 ))
  log="${LOGDIR}/iter${iter}.log"
  start=$(date +%s)
  case "${mode}" in
    native)
      (cd "${LOGDIR}/cores" && timeout 900 "${TEST_BIN}" "${targs[@]}" > "${log}" 2>&1); rc=$? ;;
    gdb)
      timeout 1800 gdb "${GDB_CMDS[@]}" --args "${TEST_BIN}" "${targs[@]}" > "${log}" 2>&1; rc=$?
      # gdb returns 0 even when the inferior crashed; detect the signal in the log.
      if grep -q "received signal SIG" "${log}"; then rc=139; fi ;;
    stress)
      CUVS_STRESS_THREADS=16 CUVS_STRESS_ITERS=20 timeout 1200 "${TEST_BIN}" --ignored --test-threads=2 stress_ > "${log}" 2>&1; rc=$? ;;
  esac
  dur=$(( $(date +%s) - start ))
  summary=$(grep -E "^test result:" "${log}" | tail -1)
  echo "ITER ${iter} mode=${mode} threads=${t:-default} rc=${rc} dur=${dur}s ${summary}"
  if [ "${rc}" -ne 0 ]; then
    fails=$((fails + 1))
    per_mode_fails[$key]=$(( ${per_mode_fails[$key]:-0} + 1 ))
    if [ "${rc}" -ge 128 ] && [ "${rc}" -ne 124 ]; then crashes=$((crashes + 1)); fi
    echo "::group::FAILURE iter ${iter} (${key}) rc=${rc}"
    if [ "${mode}" = "gdb" ]; then
      sed -n '/received signal SIG/,$p' "${log}" | head -2000
      grep -E "^test .*(FAILED|ok)$" "${log}" | tail -40
    else
      grep -vE "^test .* \.\.\. ok$" "${log}" | tail -300
    fi
    echo "::endgroup::"
    ls -la "${LOGDIR}/cores"
    for core in "${LOGDIR}"/cores/core*; do
      [ -e "${core}" ] || continue
      gdb -q -batch -ex "info threads" -ex "thread apply all bt" "${TEST_BIN}" "${core}" 2>&1 | head -3000
      rm -f "${core}"
    done
  fi
done

echo "STRESS SUMMARY: iterations=${iter} failures=${fails} crashes=${crashes}"
for k in "${!per_mode_runs[@]}"; do
  echo "  ${k}: runs=${per_mode_runs[$k]} fails=${per_mode_fails[$k]:-0}"
done

if command -v compute-sanitizer > /dev/null; then
  rapids-logger "compute-sanitizer memcheck, --test-threads=8"
  timeout 2400 compute-sanitizer --tool memcheck --error-exitcode 1 "${TEST_BIN}" --test-threads=8 2>&1 | tail -200
  echo "memcheck rc=${PIPESTATUS[0]}"
  rapids-logger "compute-sanitizer racecheck-free: synccheck skipped; initcheck on ivf_flat"
  timeout 1200 compute-sanitizer --tool memcheck --error-exitcode 1 "${TEST_BIN}" --test-threads=16 ivf_flat 2>&1 | tail -100
  echo "memcheck ivf_flat rc=${PIPESTATUS[0]}"
fi

if [ "${fails}" -ne 0 ]; then exit 1; fi
