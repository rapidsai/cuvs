#!/bin/bash
# Build script for cuvs-bench-cpu without AVX2 requirements
set -x

# Remove AVX2 flags from compilation
export CFLAGS="${CFLAGS} -mno-avx2 -mno-fma"
export CXXFLAGS="${CXXFLAGS} -mno-avx2 -mno-fma"

# Also remove the debug prefix map flags that might break caching
export CFLAGS=$(echo $CFLAGS | sed -E 's@\-fdebug\-prefix\-map[^ ]*@@g')
export CXXFLAGS=$(echo $CXXFLAGS | sed -E 's@\-fdebug\-prefix\-map[^ ]*@@g')

# Build with explicit no-AVX2 flag
./build.sh bench-ann --cpu-only --no-nvtx \
  --build-metrics=bench_ann_cpu \
  --incl-cache-stats \
  --cmake-args="-DCMAKE_CXX_FLAGS='-mno-avx2 -mno-fma'"

# Install the benchmarks
cmake --install cpp/build --component ann_bench
