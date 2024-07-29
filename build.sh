#!/bin/bash

# Copyright (c) 2020-2024, NVIDIA CORPORATION.

# cuvs build scripts

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# scripts, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libcuvs python rust docs tests bench-ann examples --uninstall  -v -g -n --compile-static-lib --allgpuarch --no-nvtx --show_depr_warn --incl-cache-stats --time -h"
HELP="$0 [<target> ...] [<flag> ...] [--cmake-args=\"<args>\"] [--cache-tool=<tool>] [--limit-tests=<targets>] [--limit-bench-ann=<targets>] [--build-metrics=<filename>]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libcuvs          - build the cuvs C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   python           - build the cuvs Python package
   rust             - build the cuvs Rust bindings
   docs             - build the documentation
   tests            - build the tests
   bench-ann        - build end-to-end ann benchmarks
   examples         - build the examples

 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --uninstall                 - uninstall files for specified targets which were built and installed prior
   --compile-static-lib        - compile static library for all components
   --limit-tests               - semicolon-separated list of test executables to compile (e.g. NEIGHBORS_TEST;CLUSTER_TEST)
   --limit-bench-ann           - semicolon-separated list of ann benchmark executables to compute (e.g. HNSWLIB_ANN_BENCH;RAFT_IVF_PQ_ANN_BENCH)
   --allgpuarch                - build for all supported GPU architectures
   --no-nvtx                   - disable nvtx (profiling markers), but allow enabling it in downstream projects
   --show_depr_warn            - show cmake deprecation warnings
   --build-metrics             - filename for generating build metrics report for libcuvs
   --incl-cache-stats          - include cache statistics in build metrics report
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   --cache-tool=<tool>         - pass the build cache tool (eg: ccache, sccache, distcc) that will be used
                                 to speedup the build process.
   --time                      - Enable nvcc compilation time logging into cpp/build/nvcc_compile_log.csv.
                                 Results can be interpreted with cpp/scripts/analyze_nvcc_log.py
   -h                          - print this text

 default action (no args) is to build libcuvs, tests and cuvs targets
"
LIBCUVS_BUILD_DIR=${LIBCUVS_BUILD_DIR:=${REPODIR}/cpp/build}
SPHINX_BUILD_DIR=${REPODIR}/docs
DOXYGEN_BUILD_DIR=${REPODIR}/cpp/doxygen
PYTHON_BUILD_DIR=${REPODIR}/python/cuvs/_skbuild
RUST_BUILD_DIR=${REPODIR}/rust/target
BUILD_DIRS="${LIBCUVS_BUILD_DIR} ${PYTHON_BUILD_DIR} ${RUST_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
CMAKE_LOG_LEVEL=""
VERBOSE_FLAG=""
BUILD_ALL_GPU_ARCH=0
BUILD_TESTS=ON
BUILD_TYPE=Release
COMPILE_LIBRARY=OFF
INSTALL_TARGET=install
BUILD_REPORT_METRICS=""
BUILD_REPORT_INCL_CACHE_STATS=OFF

TEST_TARGETS="NEIGHBORS_ANN_CAGRA_TEST"
ANN_BENCH_TARGETS="CUVS_ANN_BENCH_ALL"

CACHE_ARGS=""
NVTX=ON
LOG_COMPILE_TIME=OFF
CLEAN=0
UNINSTALL=0
DISABLE_DEPRECATION_WARNINGS=ON
CMAKE_TARGET=""
EXTRA_CMAKE_ARGS=""

# Set defaults for vars that may not have been defined externally
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBCUVS_BUILD_DIR/install}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}
BUILD_ABI=${BUILD_ABI:=ON}

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

function cacheTool {
    # Check for multiple cache options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cache\-tool" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cache-tool options were provided, please provide only one: ${ARGS}"
        exit 1
    fi
    # Check for cache tool option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cache\-tool" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        CACHE_TOOL=$(echo $ARGS | sed -e 's/.*--cache-tool=//' -e 's/ .*//')
        if [[ -n ${CACHE_TOOL} ]]; then
            # Remove the full CACHE_TOOL argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--cache-tool=$CACHE_TOOL/}
            CACHE_ARGS="-DCMAKE_CUDA_COMPILER_LAUNCHER=${CACHE_TOOL} -DCMAKE_C_COMPILER_LAUNCHER=${CACHE_TOOL} -DCMAKE_CXX_COMPILER_LAUNCHER=${CACHE_TOOL}"
        fi
    fi
}

function limitTests {
    # Check for option to limit the set of test binaries to build
    if [[ -n $(echo $ARGS | { grep -E "\-\-limit\-tests" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        LIMIT_TEST_TARGETS=$(echo $ARGS | sed -e 's/.*--limit-tests=//' -e 's/ .*//')
        if [[ -n ${LIMIT_TEST_TARGETS} ]]; then
            # Remove the full LIMIT_TEST_TARGETS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--limit-tests=$LIMIT_TEST_TARGETS/}
            TEST_TARGETS=${LIMIT_TEST_TARGETS}
            echo "Limiting tests to $TEST_TARGETS"
        fi
    fi
}

function limitAnnBench {
    # Check for option to limit the set of test binaries to build
    if [[ -n $(echo $ARGS | { grep -E "\-\-limit\-bench-ann" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        LIMIT_ANN_BENCH_TARGETS=$(echo $ARGS | sed -e 's/.*--limit-bench-ann=//' -e 's/ .*//')
        if [[ -n ${LIMIT_ANN_BENCH_TARGETS} ]]; then
            # Remove the full LIMIT_TEST_TARGETS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--limit-bench-ann=$LIMIT_ANN_BENCH_TARGETS/}
            ANN_BENCH_TARGETS=${LIMIT_ANN_BENCH_TARGETS}
        fi
    fi
}

function buildMetrics {
    # Check for multiple build-metrics options
    if [[ $(echo $ARGS | { grep -Eo "\-\-build\-metrics" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --build-metrics options were provided, please provide only one: ${ARGS}"
        exit 1
    fi
    # Check for build-metrics option
    if [[ -n $(echo $ARGS | { grep -E "\-\-build\-metrics" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        BUILD_REPORT_METRICS=$(echo $ARGS | sed -e 's/.*--build-metrics=//' -e 's/ .*//')
        if [[ -n ${BUILD_REPORT_METRICS} ]]; then
            # Remove the full BUILD_REPORT_METRICS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--build-metrics=$BUILD_REPORT_METRICS/}
        fi
    fi
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    cmakeArgs
    cacheTool
    limitTests
    limitAnnBench
    buildMetrics
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# This should run before build/install
if hasArg --uninstall; then
    UNINSTALL=1

    if hasArg cuvs || hasArg libcuvs || (( ${NUMARGS} == 1 )); then

      echo "Removing libcuvs files..."
      if [ -e ${LIBCUVS_BUILD_DIR}/install_manifest.txt ]; then
          xargs rm -fv < ${LIBCUVS_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
      fi
    fi

    if hasArg cuvs || (( ${NUMARGS} == 1 )); then
      echo "Uninstalling cuvs package..."
      if [ -e ${PYLIBCUVS_BUILD_DIR}/install_manifest.txt ]; then
          xargs rm -fv < ${PYLIBCUVS_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
      fi

      # Try to uninstall via pip if it is installed
      if [ -x "$(command -v pip)" ]; then
        echo "Using pip to uninstall cuvs"
        pip uninstall -y cuvs

      # Otherwise, try to uninstall through conda if that's where things are installed
      elif [ -x "$(command -v conda)" ] && [ "$INSTALL_PREFIX" == "$CONDA_PREFIX" ]; then
        echo "Using conda to uninstall cuvs"
        conda uninstall -y cuvs

      # Otherwise, fail
      else
        echo "Could not uninstall cuvs from pip or conda. cuvs package will need to be manually uninstalled"
      fi
    fi
    exit 0
fi


# Process flags
if hasArg -n; then
    INSTALL_TARGET=""
fi

if hasArg -v; then
    VERBOSE_FLAG="-v"
    CMAKE_LOG_LEVEL="VERBOSE"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi

if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi

if hasArg tests || (( ${NUMARGS} == 0 )); then
    BUILD_TESTS=ON
    CMAKE_TARGET="${CMAKE_TARGET};${TEST_TARGETS}"

    # Force compile library when needed test targets are specified
    if [[ $CMAKE_TARGET == *"CAGRA_C_TEST"* || \
          $CMAKE_TARGET == *"INTEROP_TEST"* || \
          $CMAKE_TARGET == *"NEIGHBORS_ANN_CAGRA_TEST"* ]]; then
      echo "-- Enabling compiled lib for gtests"
      COMPILE_LIBRARY=ON
    fi
fi

if hasArg bench-ann || (( ${NUMARGS} == 0 )); then
    BUILD_ANN_BENCH=ON
    CMAKE_TARGET="${CMAKE_TARGET};${ANN_BENCH_TARGETS}"
fi

if hasArg --no-nvtx; then
    NVTX=OFF
fi
if hasArg --time; then
    echo "-- Logging compile times to cpp/build/nvcc_compile_log.csv"
    LOG_COMPILE_TIME=ON
fi
if hasArg --show_depr_warn; then
    DISABLE_DEPRECATION_WARNINGS=OFF
fi
if hasArg clean; then
    CLEAN=1
fi
if hasArg --incl-cache-stats; then
    BUILD_REPORT_INCL_CACHE_STATS=ON
fi
if [[ ${CMAKE_TARGET} == "" ]]; then
    CMAKE_TARGET="all"
fi


SKBUILD_EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS}"
if [[ "${EXTRA_CMAKE_ARGS}" != *"DFIND_CUVS_CPP"* ]]; then
    SKBUILD_EXTRA_CMAKE_ARGS="${SKBUILD_EXTRA_CMAKE_ARGS};-DFIND_CUVS_CPP=ON"
fi

# If clean given, run it prior to any other steps
if (( ${CLEAN} == 1 )); then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
      if [ -d ${bd} ]; then
          find ${bd} -mindepth 1 -delete
          rmdir ${bd} || true
      fi
    done
fi

################################################################################
# Configure for building all C++ targets
if (( ${NUMARGS} == 0 )) || hasArg libcuvs || hasArg docs || hasArg tests || hasArg bench-prims || hasArg bench-ann; then
    COMPILE_LIBRARY=ON
    CMAKE_TARGET="${CMAKE_TARGET};cuvs"

    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUVS_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUVS_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    # get the current count before the compile starts
    CACHE_TOOL=${CACHE_TOOL:-sccache}
    if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v ${CACHE_TOOL})" ]]; then
        "${CACHE_TOOL}" --zero-stats
    fi

    mkdir -p ${LIBCUVS_BUILD_DIR}
    cd ${LIBCUVS_BUILD_DIR}
    cmake -S ${REPODIR}/cpp -B ${LIBCUVS_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUVS_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_C_LIBRARY=${COMPILE_LIBRARY} \
          -DCUVS_NVTX=${NVTX} \
          -DCUDA_LOG_COMPILE_TIME=${LOG_COMPILE_TIME} \
          -DDISABLE_DEPRECATION_WARNINGS=${DISABLE_DEPRECATION_WARNINGS} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_C_TESTS=${BUILD_TESTS} \
          -DBUILD_ANN_BENCH=${BUILD_ANN_BENCH} \
          -DBUILD_CPU_ONLY=${BUILD_CPU_ONLY} \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          ${CACHE_ARGS} \
          ${EXTRA_CMAKE_ARGS}

  compile_start=$(date +%s)
  if [[ ${CMAKE_TARGET} != "" ]]; then
      echo "-- Compiling targets: ${CMAKE_TARGET}, verbose=${VERBOSE_FLAG}"
      if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build  "${LIBCUVS_BUILD_DIR}" ${VERBOSE_FLAG} -j${PARALLEL_LEVEL} --target ${CMAKE_TARGET} ${INSTALL_TARGET}
      else
        cmake --build  "${LIBCUVS_BUILD_DIR}" ${VERBOSE_FLAG} -j${PARALLEL_LEVEL} --target ${CMAKE_TARGET}
      fi
  fi
  compile_end=$(date +%s)
  compile_total=$(( compile_end - compile_start ))

  if [[ -n "$BUILD_REPORT_METRICS" && -f "${LIBCUVS_BUILD_DIR}/.ninja_log" ]]; then
      if ! rapids-build-metrics-reporter.py 2> /dev/null && [ ! -f rapids-build-metrics-reporter.py ]; then
          echo "Downloading rapids-build-metrics-reporter.py"
          curl -sO https://raw.githubusercontent.com/rapidsai/build-metrics-reporter/v1/rapids-build-metrics-reporter.py
      fi

      echo "Formatting build metrics"
      MSG=""
      # get some sccache/ccache stats after the compile
      if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" ]]; then
          if [[ ${CACHE_TOOL} == "sccache" && -x "$(command -v sccache)" ]]; then
              COMPILE_REQUESTS=$(sccache -s | grep "Compile requests \+ [0-9]\+$" | awk '{ print $NF }')
              CACHE_HITS=$(sccache -s | grep "Cache hits \+ [0-9]\+$" | awk '{ print $NF }')
              HIT_RATE=$(echo - | awk "{printf \"%.2f\n\", $CACHE_HITS / $COMPILE_REQUESTS * 100}")
              MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
          elif [[ ${CACHE_TOOL} == "ccache" && -x "$(command -v ccache)" ]]; then
              CACHE_STATS_LINE=$(ccache -s | grep "Hits: \+ [0-9]\+ / [0-9]\+" | tail -n1)
              if [[ ! -z "$CACHE_STATS_LINE" ]]; then
                  CACHE_HITS=$(echo "$CACHE_STATS_LINE" - | awk '{ print $2 }')
                  COMPILE_REQUESTS=$(echo "$CACHE_STATS_LINE" - | awk '{ print $4 }')
                  HIT_RATE=$(echo - | awk "{printf \"%.2f\n\", $CACHE_HITS / $COMPILE_REQUESTS * 100}")
                  MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
              fi
          fi
      fi
      MSG="${MSG}<br/>parallel setting: $PARALLEL_LEVEL"
      MSG="${MSG}<br/>parallel build time: $compile_total seconds"
      if [[ -f "${LIBCUVS_BUILD_DIR}/libcuvs.so" ]]; then
          LIBCUVS_FS=$(ls -lh ${LIBCUVS_BUILD_DIR}/libcuvs.so | awk '{print $5}')
          MSG="${MSG}<br/>libcuvs.so size: $LIBCUVS_FS"
      fi
      BMR_DIR=${RAPIDS_ARTIFACTS_DIR:-"${LIBCUVS_BUILD_DIR}"}
      echo "The HTML report can be found at [${BMR_DIR}/${BUILD_REPORT_METRICS}.html]. In CI, this report"
      echo "will also be uploaded to the appropriate subdirectory of https://downloads.rapids.ai/ci/cuvs/, and"
      echo "the entire URL can be found in \"conda-cpp-build\" runs under the task \"Upload additional artifacts\""
      mkdir -p ${BMR_DIR}
      MSG_OUTFILE="$(mktemp)"
      echo "$MSG" > "${MSG_OUTFILE}"
      PATH=".:$PATH" python rapids-build-metrics-reporter.py ${LIBCUVS_BUILD_DIR}/.ninja_log --fmt html --msg "${MSG_OUTFILE}" > ${BMR_DIR}/${BUILD_REPORT_METRICS}.html
      cp ${LIBCUVS_BUILD_DIR}/.ninja_log ${BMR_DIR}/ninja.log
  fi
fi

# Build and (optionally) install the cuvs Python package
if (( ${NUMARGS} == 0 )) || hasArg python; then
    SKBUILD_CMAKE_ARGS="${SKBUILD_EXTRA_CMAKE_ARGS}" \
        SKBUILD_BUILD_OPTIONS="-j${PARALLEL_LEVEL}" \
        python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true ${REPODIR}/python/cuvs
fi

# Build the cuvs Rust bindings
if (( ${NUMARGS} == 0 )) || hasArg rust; then
    cd ${REPODIR}/rust
    cargo build --examples --lib
    cargo test
fi

export RAPIDS_VERSION="$(sed -E -e 's/^([0-9]{2})\.([0-9]{2})\.([0-9]{2}).*$/\1.\2.\3/' "${REPODIR}/VERSION")"
export RAPIDS_VERSION_MAJOR_MINOR="$(sed -E -e 's/^([0-9]{2})\.([0-9]{2})\.([0-9]{2}).*$/\1.\2/' "${REPODIR}/VERSION")"

if hasArg docs; then
    set -x
    cd ${DOXYGEN_BUILD_DIR}
    doxygen Doxyfile
    cd ${SPHINX_BUILD_DIR}
    sphinx-build -b html source _html
    cd ${REPODIR}/rust
    cargo doc -p cuvs --no-deps
    rsync -av ${RUST_BUILD_DIR}/doc/ ${SPHINX_BUILD_DIR}/_html/_static/rust
fi

################################################################################
# Initiate build for c++ examples (if needed)

if hasArg examples; then
    pushd ${REPODIR}/examples
    ./build.sh
    popd
fi
