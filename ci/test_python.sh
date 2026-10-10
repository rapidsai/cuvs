#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: GPU/driver-side characterization of the H100 CAGRA Python test
# hang (#2772, #2732, #2783). Each matrix copy runs one experiment, selected by
# the suffix of DEPENDENCIES (for example "latest-nsys"), on the reproducer
#   pytest tests/test_product_quantizer.py tests/test_cagra.py

set -euo pipefail

EXP="${RAPIDS_DEPENDENCIES#*-}"
if [[ "${EXP}" == "${RAPIDS_DEPENDENCIES}" ]]; then
  EXP=baseline
fi
RAPIDS_DEPENDENCIES="${RAPIDS_DEPENDENCIES%%-*}"
export RAPIDS_DEPENDENCIES

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "EXPERIMENT: ${EXP} (dependencies=${RAPIDS_DEPENDENCIES})"

conda config --set channel_priority strict

rapids-logger "Generate Python testing dependencies (nightly packages)"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
rapids-mamba-retry create --yes -n dbg -c conda-forge py-spy || true
DBG_BIN=/opt/conda/envs/dbg/bin
NSYS=""
if [[ "${EXP}" == nsys* ]]; then
  rapids-mamba-retry create --yes -n nsys -c conda-forge "nsight-systems>=2025" || true
  NSYS=$(find /opt/conda/envs/nsys -name nsys -type f -perm -u+x 2>/dev/null | grep -v "/host-" | head -1 || true)
  echo "nsys: ${NSYS}"
  [[ -n "${NSYS}" ]] && "${NSYS}" --version
fi

set +u
conda activate test
set -u

rapids-print-env

DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}" "${RAPIDS_ARTIFACTS_DIR:-${PWD}/artifacts}"
ART_DIR="${RAPIDS_ARTIFACTS_DIR:-${PWD}/artifacts}"

rapids-logger "Static GPU / driver information"
nvidia-smi
nvidia-smi -q -d PERFORMANCE,UTILIZATION,PAGE_RETIREMENT,ECC,CLOCK,ROW_REMAPPER || true
nvidia-smi -q | grep -iE "Addressing Mode|Driver Version|Product Name|VBIOS|Bus Id|Persistence|Compute Mode|MIG|Link (Gen|Width)|Bar1|Max|Current" | head -40 || true
uname -a; cat /proc/cmdline 2>/dev/null || true
cat /proc/driver/nvidia/version 2>/dev/null || true
for f in /proc/driver/nvidia/gpus/*/information; do echo "== ${f}"; cat "${f}"; done 2>/dev/null || true
echo "== /proc/driver/nvidia/params"; cat /proc/driver/nvidia/params 2>/dev/null || true
echo "== nvidia_uvm parameters"
for f in /sys/module/nvidia_uvm/parameters/*; do echo "$(basename "${f}")=$(cat "${f}" 2>/dev/null)"; done 2>/dev/null || true
echo "== dmesg"; (dmesg 2>&1 | grep -iE "xid|nvrm|uvm|nvidia" | tail -30) || true
nproc; free -g || true
python - <<'PYEOF' || true
import cupy as cp
rt = cp.cuda.runtime
for a in ("PageableMemoryAccess", "ConcurrentManagedAccess", "PageableMemoryAccessUsesHostPageTables", "HostNativeAtomicSupported", "DirectManagedMemAccessFromHost"):
    print(a, rt.deviceGetAttribute(getattr(rt, "cudaDevAttr" + a), 0))
print("driver", rt.driverGetVersion(), "runtime", rt.runtimeGetVersion())
PYEOF

descendants() {
  local p=$1
  echo "$p"
  for c in $(pgrep -P "$p" 2>/dev/null); do
    descendants "$c"
  done
}

# The CUDA process of a watched command (the deepest python descendant).
cuda_pid() {
  local p last=""
  for p in $(descendants "$1"); do
    if grep -q python /proc/"${p}"/cmdline 2>/dev/null; then last=${p}; fi
  done
  echo "${last}"
}

nv_irqs() {
  awk '/nvidia/ {s=0; for (i=2; i<=NF; i++) if ($i ~ /^[0-9]+$/) s+=$i; print $1 s}' /proc/interrupts 2>/dev/null | tr '\n' ' '
}

vmstat_sel() {
  grep -E "^(pgfault|pgmajfault|pgmigrate_success|pgmigrate_fail|numa_hint_faults|numa_pages_migrated|thp_migration_success|pgfree|nr_mlock|pgactivate)" /proc/vmstat 2>/dev/null | tr '\n' ' '
}

thread_line() {
  local t=$1 stat rest
  stat=$(cat "$t"/stat 2>/dev/null) || return
  rest=${stat##*) }
  # shellcheck disable=SC2086
  set -- $rest
  # fields after comm: 1=state 8=minflt 10=majflt 12=utime 13=stime 37=processor
  echo "$(basename "$t") $(cat "$t"/comm 2>/dev/null) state=$1 minflt=$8 majflt=${10} utime=${12} stime=${13} cpu=${37} wchan=$(cat "$t"/wchan 2>/dev/null) ctxsw=$(grep -E "ctxt" "$t"/status 2>/dev/null | awk '{print $2}' | tr '\n' '/') syscall=$(cut -d' ' -f1-3 < "$t"/syscall 2>/dev/null)"
}

# GPU/driver-side sampling of a stalled process. Passive (no ptrace) first.
gpu_diag() {
  local p=$1 log=$2 n
  echo "########## B3 GPU/DRIVER DIAG $(date -u +%FT%TZ) pid=${p}"
  nvidia-smi -q -d PERFORMANCE,UTILIZATION,PAGE_RETIREMENT,ECC,CLOCK,ROW_REMAPPER 2>&1 | grep -vE "^\s*$" | head -120 || true
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
  echo "---- dmesg"; (dmesg 2>&1 | grep -iE "xid|nvrm|uvm" | tail -20) || true
  echo "---- process status"; grep -E "^(State|VmRSS|VmLck|VmPin|Threads|voluntary|nonvoluntary)" /proc/"${p}"/status || true
  echo "---- /proc/stat cpu, nvidia irqs, vmstat; main thread every 2 s"
  for n in 1 2 3 4 5 6; do
    echo "[s${n} $(date +%T.%N | cut -c1-12)] $(head -1 /proc/stat)"
    echo "[s${n}] irq: $(nv_irqs)"
    echo "[s${n}] vm: $(vmstat_sel)"
    echo "[s${n}] main: $(thread_line /proc/"${p}"/task/"${p}")"
    echo "[s${n}] nvidia-smi: $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw,memory.used,pcie.link.gen.gpucurrent,clocks_event_reasons.active --format=csv,noheader 2>&1)"
    sleep 2
  done
  echo "---- all threads (minflt/stime deltas over 5 s)"
  for t in /proc/"${p}"/task/*; do thread_line "$t"; done > "${DIAG_DIR}/threads.a"
  sleep 5
  for t in /proc/"${p}"/task/*; do thread_line "$t"; done > "${DIAG_DIR}/threads.b"
  paste -d'\n' "${DIAG_DIR}/threads.a" "${DIAG_DIR}/threads.b" | awk 'NR%2==1{a=$0; next} {print "A " a; print "B " $0}' | grep -E "state=R|minflt=[0-9]{4,}" | head -20 || true
  echo "---- kernel stacks (main thread, 3 samples)"
  for n in 1 2 3; do head -30 /proc/"${p}"/task/"${p}"/stack 2>&1; echo "--"; sleep 1; done
  echo "---- SIGSTOP 15 s / SIGCONT (no ptrace), then watch the log for 60 s"
  local size0
  size0=$(stat -c %s "${log}")
  kill -STOP "${p}"; sleep 15; kill -CONT "${p}"
  sleep 60
  echo "log size ${size0} -> $(stat -c %s "${log}") (grew => resumed after SIGSTOP/SIGCONT)"
  echo "[after] main: $(thread_line /proc/"${p}"/task/"${p}")"
  echo "---- py-spy dump --native (single ptrace sample)"
  timeout 120 "${DBG_BIN}/py-spy" dump --native --pid "${p}" 2>&1 | head -45 || true
  echo "########## END B3 GPU/DRIVER DIAG"
}

HANGS=0
HANG_LIST=""
RESULTS=""
NSYS_SESSION=""
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
  local rc=0
  if (( hung )); then
    HANGS=$((HANGS + 1))
    HANG_LIST="${HANG_LIST} ${name}"
    echo "!!!!! HANG DETECTED in ${name} after $(( $(date +%s) - start ))s (no output for ${stall}s), last test: $(grep -aoE "tests/[a-z_]+\.py::[^ ]+" "${log}" | tail -1)"
    local cp
    cp=$(cuda_pid "${pid}")
    if [[ -z "${cp}" ]]; then cp=$(pgrep -n -f "pytest" || true); fi
    echo "watched pid ${pid}, CUDA process ${cp}"; ps -eo pid,ppid,stat,etime,time,cmd --forest | cut -c1-200 | grep -vE "ps -eo|cut -c" || true
    gpu_diag "${cp}" "${log}"
    if kill -0 "${cp}" 2>/dev/null && [[ -n "${NSYS_SESSION}" ]]; then
      echo "---- nsys stop --session=${NSYS_SESSION}"
      timeout 300 "${NSYS}" stop --session="${NSYS_SESSION}" 2>&1 | tail -5 || echo "nsys stop failed/timed out"
      for _ in $(seq 1 24); do kill -0 "${pid}" 2>/dev/null || break; sleep 5; done
    fi
    for p in $(descendants "${cp}"); do kill -9 "${p}" 2>/dev/null || true; done
    # Give nsys time to write the report after the target dies.
    for _ in $(seq 1 120); do kill -0 "${pid}" 2>/dev/null || break; sleep 5; done
    for p in $(descendants "${pid}"); do kill -9 "${p}" 2>/dev/null || true; done
    wait "${pid}" 2>/dev/null
    rc=124
  else
    wait "${pid}"
    rc=$?
  fi
  sleep 2
  kill "${tailpid}" 2>/dev/null || true
  wait "${tailpid}" 2>/dev/null || true
  RESULTS="${RESULTS} ${name}:${rc}"
  echo "<<<<< END ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  return ${rc}
}

nsys_summary() {
  local rep=$1
  [[ -f "${rep}" ]] || { echo "no report ${rep}"; ls -la "$(dirname "${rep}")"; return; }
  ls -la "${rep}"
  local db="${rep%.nsys-rep}.sqlite"
  timeout 900 "${NSYS}" export --type sqlite --force-overwrite true -o "${db}" "${rep}" > /dev/null 2>&1 || echo "export failed"
  timeout 900 "${NSYS}" stats -q --format csv -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,um_sum,um_total_sum,um_cpu_page_faults_sum "${db}" 2>&1 | cut -c1-240 | head -150 || true
  python "${PWD}/ci/b3_nsys_query.py" "${db}" 2>&1 | head -300 || true
}

set +e

CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
CI_DIR="${PWD}/ci"
PYTEST_ARGS=(-v -p no:cacheprovider -o faulthandler_timeout=240 -p b3_shim)
export PYTHONPATH="${CI_DIR}:${PYTHONPATH:-}"
export B3_SHIM=""
STALL=420
MAX_HANGS=3
case "${EXP}" in
  baseline|nsys|drv*) ;;
  clb) export CUDA_LAUNCH_BLOCKING=1 ;;
  pinned) B3_SHIM=pq_pinned ;;
  pqdev) B3_SHIM=pq_device ;;
  rmmasync) B3_SHIM=rmm_async ;;
  rmmpool) B3_SHIM=rmm_pool ;;
  rmmmanaged) B3_SHIM=rmm_managed ;;
  sepproc) ;;
  *) echo "unknown experiment ${EXP}"; exit 1 ;;
esac
env | grep -E "^(CUDA_|B3_)" || true

BUDGET=$(( 115 * 60 ))
LOOP_START=$(date +%s)
NSYS_REF_DONE=0
i=0
pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
while (( $(date +%s) - LOOP_START < BUDGET )); do
  i=$((i + 1))
  rapids-logger "[${EXP}] iteration ${i} (hangs so far: ${HANGS}:${HANG_LIST})"
  case "${EXP}" in
    sepproc)
      run_watched "iter${i}-pq-only" "${STALL}" python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" tests/test_product_quantizer.py
      run_watched "iter${i}-cagra-only" "${STALL}" python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" tests/test_cagra.py
      ;;
    nsys)
      rep="${ART_DIR}/b3-iter${i}"
      NSYS_SESSION="b3iter${i}"
      run_watched "iter${i}-pq-cagra" "${STALL}" "${NSYS}" profile -o "${rep}" --force-overwrite true \
        --session-new="${NSYS_SESSION}" \
        --trace=cuda,nvtx,osrt --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
        --cuda-memory-usage=true --cuda-flush-interval=2000 --sample=none --cpuctxsw=none \
        python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" tests/test_product_quantizer.py tests/test_cagra.py
      rc=$?
      NSYS_SESSION=""
      if (( rc == 124 )); then
        rapids-logger "nsys summary of the hung iteration ${i}"
        nsys_summary "${rep}.nsys-rep"
      elif (( NSYS_REF_DONE == 0 )); then
        rapids-logger "nsys summary of a passing iteration ${i} (reference)"
        nsys_summary "${rep}.nsys-rep"
        NSYS_REF_DONE=1
      else
        rm -f "${rep}.nsys-rep" "${rep}.sqlite"
      fi
      ;;
    *)
      run_watched "iter${i}-pq-cagra" "${STALL}" python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" tests/test_product_quantizer.py tests/test_cagra.py
      ;;
  esac
  if (( HANGS >= MAX_HANGS )); then
    break
  fi
done
popd > /dev/null || exit 1

rapids-logger "B3 SUMMARY [${EXP}]: ${HANGS} hang(s) in ${i} iteration(s):${HANG_LIST} | results:${RESULTS}"
du -sh "${ART_DIR}" 2>/dev/null || true
if (( HANGS > 0 )); then
  exit 1
fi
exit 0
