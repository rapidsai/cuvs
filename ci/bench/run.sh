#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Benchmark three libcuvs builds made identically in this workflow run:
#   base   = upstream main (this tree)
#   pr2801 = main + #2801 (deterministic k-means fit)
#   pr2804 = main + #2804 (balanced k-means donor selection / IVF-PQ grouping)
# The benchmark is compiled once against the public headers (identical in all three); each variant
# gets a copy whose DT_RPATH points at that variant's libcuvs.so (checked at run time). Runs are interleaved: every round runs each variant
# once, rotating the order (ABC, BCA, CAB, ...); each run does 1 warmup + `reps` timed repetitions.
#
# The workload group comes from the matrix's DEPENDENCIES field (RAPIDS_DEPENDENCIES).
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
GROUP=${RAPIDS_DEPENDENCIES:-lite}
VARIANTS=(base pr2801 pr2804)
REPO=$(pwd)
W=/tmp/bench
mkdir -p "${W}"
RESULTS="${W}/results.txt"
: > "${RESULTS}"

declare -A CHANNEL
for v in "${VARIANTS[@]}"; do
  CHANNEL[${v}]=$(rapids-download-from-github \
    "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "${RAPIDS_CUDA_VERSION}")_${v}")
done
CUDA_MM=${RAPIDS_CUDA_VERSION%.*}

nvidia-smi
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | head -80 || true
lscpu | head -20 || true

if [[ "${GROUP}" == "tests" ]]; then
  rapids-logger "Run the C++ tests of the #2804 build"
  rapids-mamba-retry create --yes -n tests --override-channels --strict-channel-priority \
    -c "${CHANNEL[pr2804]}" -c rapidsai-nightly -c conda-forge \
    libcuvs libcuvs-tests "cuda-version=${CUDA_MM}"
  set +u
  conda activate tests
  set -u
  conda list | grep -E "^(libcuvs|libraft|cuda-version) "
  cd "${CONDA_PREFIX}/bin/gtests/libcuvs"
  EXITCODE=0
  ./CLUSTER_TEST --gtest_filter='KmeansBalancedAdjustCenters.*' --gtest_repeat=50 \
    --gtest_brief=1 || EXITCODE=1
  ./CLUSTER_TEST --gtest_filter='*KmeansBalanced*' --gtest_brief=1 || EXITCODE=1
  ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_brief=1 || EXITCODE=1
  ./NEIGHBORS_ANN_IVF_FLAT_TEST --gtest_brief=1 || EXITCODE=1
  exit ${EXITCODE}
fi

PKGS=(libcuvs "cuda-version=${CUDA_MM}" cuda-nvcc cuda-cudart-dev libcurand-dev cxx-compiler patchelf)
rapids-logger "Create the benchmark environment"
rapids-mamba-retry create --yes -n bench --override-channels --strict-channel-priority \
  -c "${CHANNEL[base]}" -c rapidsai-nightly -c conda-forge "${PKGS[@]}"
set +u
conda activate bench
set -u
conda list | grep -E "^(libcuvs|libraft|librmm|cuda-version|cuda-nvcc|cuda-cudart) "

rapids-logger "Extract libcuvs.so of each variant"
for v in "${VARIANTS[@]}"; do
  pkg=$(find "${CHANNEL[${v}]}" -name 'libcuvs-[0-9]*.conda' | head -n 1)
  /opt/conda/bin/python -c "import sys; from conda_package_handling import api; api.extract(sys.argv[1], sys.argv[2])" \
    "${pkg}" "${W}/${v}"
  echo "${v}: ${pkg}"
done
md5sum "${W}"/*/lib/libcuvs.so

rapids-logger "Compile the benchmarks"
compile() {
  local out=$1
  shift
  # shellcheck disable=SC2046
  nvcc -std=c++20 -O3 -arch=native --extended-lambda --expt-relaxed-constexpr -ccbin "${CXX}" -w \
    -I"${CONDA_PREFIX}/include/rapids" -I"${CONDA_PREFIX}/include" \
    $(for d in "${CONDA_PREFIX}"/targets/*/include; do echo "-I$d"; done) \
    "$@" -o "${out}" \
    -L"${CONDA_PREFIX}/lib" $(for d in "${CONDA_PREFIX}"/targets/*/lib; do echo "-L$d"; done) \
    -lcuvs -lrmm -lrapids_logger -lcudart -lcurand
}
compile "${W}/bench" ci/bench/bench.cu
for v in "${VARIANTS[@]}"; do
  cp "${W}/bench" "${W}/bench_${v}"
  patchelf --force-rpath --set-rpath "${W}/${v}/lib:${CONDA_PREFIX}/lib" "${W}/bench_${v}"
  echo "bench_${v} rpath: $(patchelf --print-rpath "${W}/bench_${v}")"
done
# The centroid-sum microbenchmark includes #2801's header directly.
mkdir -p "${W}/tree2801"
cp -r cpp "${W}/tree2801/"
(cd "${W}/tree2801" && git apply --include='cpp/src/*' "${REPO}/ci/bench/patches/pr2801.patch")
compile "${W}/sums_2801" -I"${W}/tree2801/cpp/src" ci/bench/sums_2801.cu

run() {
  local v=$1
  shift
  echo "+ ${v} $*"
  # libcuvs.so comes from the binary's DT_RPATH (searched before LD_LIBRARY_PATH); its own
  # dependencies come from the environment.
  BENCH_EXPECT_LIB="${W}/${v}/lib/libcuvs.so" LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    "${W}/bench_${v}" "${v}" "$@" 2>&1 | tee -a "${RESULTS}" | grep -E "^(RESULT|SKIP|ERROR|LIB)" || true
}
# rounds R args...: R interleaved rounds over all variants, rotating the order.
rounds() {
  local R=$1
  shift
  for ((r = 0; r < R; r++)); do
    for i in 0 1 2; do
      run "${VARIANTS[$(((r + i) % 3))]}" "$@"
    done
  done
}

micro() {
  rapids-logger "Centroid-sum microbenchmark (#2801 kernels vs. atomics)"
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" "${W}/sums_2801" | tee -a "${RESULTS}"
}

R=5
START=$(date +%s)
case "${GROUP}" in
  kmeans)
    micro
    rapids-logger "k-means fit sweep (float, device data, Array init)"
    for nd in "100000 32" "100000 128" "100000 1024" "1000000 32" "1000000 128" "1000000 1024" \
      "10000000 32" "10000000 128" "10000000 1024"; do
      read -r n d <<< "${nd}"
      rounds ${R} kmeans n="${n}" d="${d}" k=10,100,1000,10000
    done
    rapids-logger "k-means fit with sample weights"
    rounds ${R} kmeans n=1000000 d=128 k=100,1000 w=1
    rounds ${R} kmeans n=1000000 d=1024 k=100 w=1
    rounds ${R} kmeans n=10000000 d=32 k=1000 w=1
    rapids-logger "k-means fit, double"
    rounds ${R} kmeans dtype=f64 n=1000000 d=128 k=100,1000
    rounds ${R} kmeans dtype=f64 n=100000 d=1024 k=1000
    rounds ${R} kmeans dtype=f64 n=10000000 d=32 k=100
    rounds ${R} kmeans dtype=f64 n=1000000 d=128 k=1000 w=1
    rapids-logger "k-means fit, batched host data"
    rounds ${R} kmeans n=10000000 d=128 k=1000 batch=1000000
    rounds ${R} kmeans n=1000000 d=128 k=100 batch=250000 w=1
    rapids-logger "k-means fit, k-means++ init"
    rounds ${R} kmeans n=1000000 d=128 k=1000 init=kpp
    rapids-logger "k-means quality vs. a float64 fit"
    rounds 2 kmeans data=blobs n=1000000 d=128 k=100,1000 ref64=1 reps=2 iters=20
    rounds 2 kmeans data=blobs n=100000 d=32 k=10,100 ref64=1 reps=2 iters=20
    rounds 2 kmeans data=blobs n=1000000 d=32 k=1000 w=1 ref64=1 reps=2 iters=20
    rounds 2 kmeans data=uniform n=1000000 d=128 k=1000 ref64=1 reps=2 iters=20
    ;;
  ivf)
    micro
    rapids-logger "Balanced k-means fit / predict"
    rounds ${R} balanced n=1000000 d=128 k=1024,4096 predict=1
    rounds ${R} balanced n=2500000 d=128 k=4096
    rounds ${R} balanced n=10000000 d=128 k=16384 predict=1 reps=1
    rounds ${R} balanced n=1000000 d=768 k=1024
    rounds ${R} balanced n=100000 d=32 k=4096
    rounds ${R} balanced dtype=i8 n=1000000 d=128 k=1024
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=random iters=25
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=size iters=25
    rapids-logger "Index builds (+ recall@10, search QPS)"
    rounds ${R} ivf n=1000000 d=128 index=flat:1024,pq:1024:64,pqc:1024:32,sq:1024,rbq:1024,cagra:64
    rounds ${R} ivf n=5000000 d=128 index=flat:4096,pq:4096:64,sq:4096,rbq:4096
    rounds ${R} ivf n=10000000 d=96 index=flat:16384,pq:16384:48 reps=1
    rounds ${R} ivf n=1000000 d=768 index=flat:1024,pq:1024:96
    ;;
  profile)
    micro
    rapids-logger "Balanced k-means, 64 small fits (IVF-PQ codebook-like)"
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=size iters=25 distinct=1
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=random iters=25 distinct=1
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=size iters=25
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=random iters=25
    rapids-logger "Install Nsight Systems"
    rapids-mamba-retry install --yes -n bench -c conda-forge nsight-systems
    i=0
    for cfg in "n=100000 d=128 k=100" "n=100000 d=32 k=1000" "n=100000 d=1024 k=10" \
      "n=1000000 d=128 k=10"; do
      i=$((i + 1))
      for v in base pr2801; do
        rapids-logger "nsys: ${v} kmeans ${cfg}"
        # shellcheck disable=SC2086
        BENCH_EXPECT_LIB="${W}/${v}/lib/libcuvs.so" LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
          nsys profile -t cuda --stats=false -f true -o "${W}/prof_${v}_${i}" \
          "${W}/bench_${v}" "${v}" kmeans ${cfg} reps=3 warmup=1 | grep -E "^(RESULT|ERROR|LIB)" || true
        nsys stats -r cuda_gpu_kern_sum,cuda_api_sum --format table "${W}/prof_${v}_${i}.nsys-rep" \
          2>/dev/null | cut -c1-220 | head -n 70 || true
      done
    done
    ;;
  profile2)
    rapids-logger "Install Nsight Systems"
    rapids-mamba-retry install --yes -n bench -c conda-forge nsight-systems
    i=0
    for cfg in "donor=size" "donor=random distinct=1"; do
      i=$((i + 1))
      for v in base pr2804 base pr2804; do
        rapids-logger "nsys: ${v} balanced 64 calls ${cfg}"
        # shellcheck disable=SC2086
        BENCH_EXPECT_LIB="${W}/${v}/lib/libcuvs.so" LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
          nsys profile -t cuda --stats=false -f true -o "${W}/prof_${v}_${i}" \
          "${W}/bench_${v}" "${v}" balanced n=65536 d=2 k=256 calls=64 iters=25 ${cfg} \
          reps=1 warmup=0 | grep -E "^(RESULT|ERROR|LIB)" || true
        nsys stats -r cuda_gpu_kern_sum,cuda_api_sum --format table "${W}/prof_${v}_${i}.nsys-rep" \
          2>/dev/null | cut -c1-250 | head -n 70 || true
      done
    done
    ;;
  lite)
    micro
    rounds ${R} kmeans n=1000000 d=128 k=100,1000,10000 reps=1
    rounds ${R} kmeans n=10000000 d=32 k=1000 reps=1
    rounds ${R} kmeans n=1000000 d=1024 k=1000 reps=1
    rounds ${R} kmeans n=1000000 d=128 k=1000 w=1 reps=1
    rounds ${R} kmeans dtype=f64 n=1000000 d=128 k=1000 reps=1
    rounds ${R} kmeans n=10000000 d=128 k=1000 batch=1000000 reps=1
    rounds ${R} balanced n=1000000 d=128 k=1024 reps=1
    rounds ${R} balanced n=65536 d=2 k=256 calls=64 donor=random iters=25 reps=1
    rounds ${R} ivf n=1000000 d=128 index=flat:1024,pq:1024:64,pqc:1024:32,cagra:64 reps=1
    rounds ${R} ivf n=5000000 d=128 index=flat:4096,pq:4096:64 reps=1
    ;;
  *)
    echo "unknown group ${GROUP}"
    exit 1
    ;;
esac
echo "Benchmark time: $(($(date +%s) - START)) s"
nvidia-smi -q -d CLOCK,PERFORMANCE | head -60 || true

rapids-logger "Summary (${GROUP})"
/opt/conda/bin/python ci/bench/summarize.py "${RESULTS}"
