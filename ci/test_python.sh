#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: reproducer for the H100 CAGRA Python test hang (#2772, #2732).
# Uses nightly conda packages, repeats the CAGRA tests and the cuvs_bench CLI
# test, and dumps host/GPU stacks when a hang is detected.

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

rapids-logger "Create debugging tools environments"
rapids-mamba-retry create --yes -n dbg -c conda-forge gdb py-spy || true
rapids-mamba-retry create --yes -n cudagdb -c conda-forge "cuda-gdb=12.9" "cuda-sanitizer-api=12.9" "cuda-version=12.9" || true
DBG_BIN=/opt/conda/envs/dbg/bin
CUDA_GDB=$(ls /opt/conda/envs/cudagdb/bin/cuda-gdb 2>/dev/null || true)

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "System info"
nvidia-smi
nvidia-smi -q || true
nproc
ulimit -a
env | grep -E "^(CUDA|OMP|NVIDIA|RAPIDS|LD_)" | sort || true
ldconfig -p | grep -E "cudadebugger|libcuda\.so" || true
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || true
lscpu || true
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
taskset -p $$ || true
free -g || true

rapids-logger "HMM / pointer attribute probe"
nvidia-smi -q | grep -iE "Addressing Mode|Driver Version|Product Name" || true
cat /proc/driver/nvidia/version 2>/dev/null || true
grep -iE "hmm|ats" /proc/driver/nvidia/params 2>/dev/null || true
cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm 2>/dev/null || true
python - <<'PYEOF' || true
import numpy as np
import cupy as cp
rt = cp.cuda.runtime
dev = 0
print("pageableMemoryAccess", rt.deviceGetAttribute(rt.cudaDevAttrPageableMemoryAccess, dev))
print("concurrentManagedAccess", rt.deviceGetAttribute(rt.cudaDevAttrConcurrentManagedAccess, dev))
a = np.zeros(1 << 20, dtype=np.float32)
attr = rt.pointerGetAttributes(a.ctypes.data)
print("numpy ptr", hex(a.ctypes.data), "type", attr.type, "devicePointer", hex(attr.devicePointer), "hostPointer", hex(attr.hostPointer))
PYEOF

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"
DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}"

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
    echo "$(basename "$t") $(cat "$t"/comm 2>/dev/null) state=$1 utime=${12} stime=${13} wchan=$(cat "$t"/wchan 2>/dev/null) syscall=$(cut -c1-120 < "$t"/syscall 2>/dev/null)"
  done
}

dump_diag() {
  local root=$1 log=$2
  local pids p size0
  pids=$(descendants "${root}")
  echo "########## HANG DIAGNOSTICS $(date -u +%FT%TZ) root=${root} pids: ${pids}"
  ps -eo pid,ppid,stat,wchan:32,etime,time,pcpu,rss,cmd --forest | cut -c1-250 || true
  cat /proc/loadavg
  nvidia-smi || true
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,power.draw,memory.used --format=csv -lms 500 & local smipid=$!
  sleep 3; kill "${smipid}" 2>/dev/null || true
  # Passive information first (no ptrace).
  for p in ${pids}; do
    echo "---- passive thread info ${p} (sample 1)"
    thread_times "${p}"
    sleep 2
    echo "---- passive thread info ${p} (sample 2)"
    thread_times "${p}"
    grep -E "^(VmRSS|VmSwap|Threads|voluntary|nonvoluntary)" /proc/"${p}"/status || true
  done
  size0=$(stat -c %s "${log}")
  # One ptrace-based sample, then check whether the process resumes on its own.
  for p in ${pids}; do
    if grep -q python /proc/"${p}"/cmdline 2>/dev/null; then
      echo "---- py-spy dump --native ${p}"
      timeout 120 "${DBG_BIN}/py-spy" dump --native --pid "${p}" 2>&1 | head -60 || true
    fi
  done
  sleep 30
  echo "---- log size before/after the py-spy attach: ${size0} -> $(stat -c %s "${log}")"
  for p in ${pids}; do
    if grep -q python /proc/"${p}"/cmdline 2>/dev/null; then
      echo "---- py-spy dump --native ${p} (30 s later)"
      timeout 120 "${DBG_BIN}/py-spy" dump --native --pid "${p}" 2>&1 | head -40 || true
      thread_times "${p}" | head -3
    fi
  done
  # GPU core dump through the user-triggered coredump pipe, then inspect it offline.
  for p in ${pids}; do
    pipe="${DIAG_DIR}/corepipe.${p}"
    if [[ ! -p "${pipe}" ]]; then
      echo "---- pipes in ${DIAG_DIR}:"; find "${DIAG_DIR}" -maxdepth 1 -type p
      pipe=$(find "${DIAG_DIR}" -maxdepth 1 -type p -name "corepipe*" | tail -1)
    fi
    if [[ -n "${pipe}" && -p "${pipe}" ]]; then
      echo "---- triggering GPU coredump through ${pipe}"
      timeout 30 sh -c "echo 1 > '${pipe}'" || echo "pipe write timed out"
      for _ in $(seq 1 60); do
        if ! kill -0 "${p}" 2>/dev/null; then break; fi
        sleep 5
      done
      find "${DIAG_DIR}" -maxdepth 1 -name "core*" -exec ls -la {} + || true
      for core in "${DIAG_DIR}"/core.cuda.*; do
        [[ -f "${core}" ]] || continue
        if [[ -n "${CUDA_GDB}" ]]; then
          timeout 600 "${CUDA_GDB}" -batch -nx \
            -ex "set pagination off" \
            -ex "target cudacore ${core}" \
            -ex "info cuda kernels" \
            -ex "info cuda blocks" \
            -ex "info cuda warps" \
            -ex "bt" \
            -ex "info cuda lanes" 2>&1 | head -300 || true
        fi
      done
    else
      echo "---- no coredump pipe ${pipe}"; find "${DIAG_DIR}" -maxdepth 1 | head
    fi
  done
  echo "########## END HANG DIAGNOSTICS"
}

HANGS=0
HANG_LIST=""
# run_watched <name> <stall_seconds> <cmd...>
run_watched() {
  local name=$1 stall=$2
  shift 2
  local log="${DIAG_DIR}/${name}.log"
  local start
  start=$(date +%s)
  echo ">>>>> START ${name}: $*"
  "$@" > "${log}" 2>&1 &
  local pid=$!
  tail -n +1 -F "${log}" 2>/dev/null &
  local tailpid=$!
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
  sleep 2
  kill "${tailpid}" 2>/dev/null || true
  wait "${tailpid}" 2>/dev/null || true
  local rc=0
  if (( hung )); then
    HANGS=$((HANGS + 1))
    HANG_LIST="${HANG_LIST} ${name}"
    echo "!!!!! HANG DETECTED in ${name} after $(( $(date +%s) - start ))s (no output for ${stall}s)"
    dump_diag "${pid}" "${log}"
    for p in $(descendants "${pid}"); do kill -9 "${p}" 2>/dev/null || true; done
    wait "${pid}" 2>/dev/null
    rc=124
    echo "---- nvidia-smi after kill"
    sleep 5
    nvidia-smi || true
  else
    wait "${pid}"
    rc=$?
  fi
  echo "<<<<< END ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  return ${rc}
}

set +e

CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
CI_DIR="${PWD}/ci"
PYTEST_ARGS=(-v -p no:cacheprovider -o faulthandler_timeout=240 -o junit_family=xunit2)

# CPU hogs to emulate a CPU-starved runner (noisy neighbors), which widens host-side race windows.
HOG_PIDS=()
start_hogs() {
  local n=$1
  for _ in $(seq 1 "${n}"); do
    sh -c 'while :; do :; done' &
    HOG_PIDS+=($!)
  done
}
stop_hogs() {
  for p in "${HOG_PIDS[@]}"; do kill "${p}" 2>/dev/null || true; done
  HOG_PIDS=()
}

rapids-logger "compute-sanitizer on the host-input product quantizer tests"
SANITIZER=$(find /opt/conda/envs/cudagdb -name compute-sanitizer -type f -perm -u+x 2>/dev/null | head -1)
echo "compute-sanitizer: ${SANITIZER}"
pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
python -m pytest --collect-only -q -p no:cacheprovider tests/test_product_quantizer.py 2>/dev/null \
  | grep -E "test_product_quantizer\[False-.*-64-700\]" | sed "s|^cuvs/||" > "${DIAG_DIR}/pq_host_ids.txt" || true
head -3 "${DIAG_DIR}/pq_host_ids.txt"
wc -l "${DIAG_DIR}/pq_host_ids.txt"
if [[ -n "${SANITIZER}" ]]; then
  # shellcheck disable=SC2046
  timeout 45m "${SANITIZER}" --tool memcheck --print-limit 100 --error-exitcode 0 \
    python -m pytest -q -p no:cacheprovider $(cat "${DIAG_DIR}/pq_host_ids.txt") 2>&1 | tail -150
  # shellcheck disable=SC2046
  timeout 30m "${SANITIZER}" --tool initcheck --print-limit 100 --error-exitcode 0 \
    python -m pytest -q -p no:cacheprovider $(head -8 "${DIAG_DIR}/pq_host_ids.txt") 2>&1 | tail -150
fi
popd > /dev/null || exit 1

# GPU core dumps on demand: writing to the pipe makes the driver dump the GPU state of the process.
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_PIPE="${DIAG_DIR}/corepipe.%p"
export CUDA_COREDUMP_FILE="${DIAG_DIR}/core.cuda.%p"
export CUDA_COREDUMP_GENERATION_FLAGS="skip_global_memory,skip_shared_memory,skip_local_memory"

# Total budget for the repeat loop (seconds).
BUDGET=$(( 120 * 60 ))
LOOP_START=$(date +%s)
i=0
while (( $(date +%s) - LOOP_START < BUDGET )); do
  i=$((i + 1))
  NHOGS=$(( (i % 2) * 12 ))
  rapids-logger "Iteration ${i}, ${NHOGS} CPU hogs (hangs so far: ${HANGS}:${HANG_LIST})"
  start_hogs "${NHOGS}"
  cat /proc/loadavg

  pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
  # Host-input product quantizer tests followed by the CAGRA tests in one process.
  PYTHONPATH="${CI_DIR}:${PYTHONPATH:-}" run_watched "iter${i}-pq-cagra" 480 \
      python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" -p hang_sync_plugin \
      tests/test_product_quantizer.py tests/test_cagra.py
  popd > /dev/null || exit 1
  stop_hogs

  if (( HANGS >= 2 )); then
    break
  fi
done
stop_hogs

rapids-logger "HANG SUMMARY: ${HANGS} hang(s) in ${i} iteration(s):${HANG_LIST}"
if (( HANGS > 0 )); then
  exit 1
fi
exit 0
