#!/bin/bash
# cuVS Installation & Environment Verification
# Builds tarball, extracts, verifies deps, installs missing ones, runs sanity checks.
# Usage: ./setup_and_test_cuvs_c.sh [--build|--extract|--test|--allgpuarch|--all] [tarball]
# Prereqs: micromamba, Python 3.x, NVIDIA GPU
#
# This script is located in the notebooks/ directory. Paths auto-detect the repo root.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration: paths auto-detect from script location (override via env vars)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUVS_REPO="${CUVS_REPO:-${REPO_ROOT}}"
TARBALL="${CUVS_REPO}/libcuvs_c.tar.gz"
INSTALL_PREFIX="${CUVS_INSTALL_PREFIX:-/usr/local}"
ENV_NAME="${CUVS_CONSUMER_ENV:-cuvs}"
CUVS_YAML="${CUVS_YAML:-${CUVS_REPO}/conda/environments/all_cuda-131_arch-x86_64.yaml}"
LIBCUVS_C="${INSTALL_PREFIX}/lib/libcuvs_c.so"

# -----------------------------------------------------------------------------
# detect_rapids_version: Read MAJOR.MINOR from version_config.h or VERSION file.
# Used to pin librmm/pylibraft when installing missing deps. Output format: 26.4
# -----------------------------------------------------------------------------
detect_rapids_version() {
  local m j
  if [[ -f "${INSTALL_PREFIX}/include/cuvs/version_config.h" ]]; then
    m=$(grep CUVS_VERSION_MAJOR "${INSTALL_PREFIX}/include/cuvs/version_config.h" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    j=$(grep CUVS_VERSION_MINOR "${INSTALL_PREFIX}/include/cuvs/version_config.h" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    [[ -n "$m" && -n "$j" ]] && echo "${m}.${j}" && return
  fi
  if [[ -f "${CUVS_REPO}/VERSION" ]]; then
    local v=$(head -1 "${CUVS_REPO}/VERSION" 2>/dev/null | sed -n 's/^\([0-9][0-9]\)\.\([0-9][0-9]\).*/\1.\2/p')
    [[ -n "$v" ]] && echo "$v" | sed 's/\.0\([1-9]\)/.\1/' && return
  fi
  echo "26.4"
}

# -----------------------------------------------------------------------------
# Argument parsing: --build, --extract, --test, --allgpuarch, --all
# -----------------------------------------------------------------------------
DO_BUILD=0 DO_EXTRACT=0 DO_TEST=0 ALLGPUARCH=0
for arg in "$@"; do
  case "$arg" in
    --build)      DO_BUILD=1 ;;
    --extract)    DO_EXTRACT=1 ;;
    --test)       DO_TEST=1 ;;
    --allgpuarch) ALLGPUARCH=1 ;;
    --all)        DO_BUILD=1; DO_EXTRACT=1; DO_TEST=1 ;;
    -h|--help)
      echo "Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS] [tarball]"
      echo "Options: --build --extract --test --allgpuarch --all"
      exit 0 ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *)  TARBALL="$arg" ;;
  esac
done
[[ $DO_BUILD -eq 0 && $DO_EXTRACT -eq 0 && $DO_TEST -eq 0 ]] && DO_BUILD=1 && DO_EXTRACT=1 && DO_TEST=1

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
fail() { echo "FAILED: $1"; exit 1; }
info() { echo "[INFO] $1"; }

echo "=== cuVS Installation & Verification ==="
echo "Tarball: ${TARBALL} | Install: ${INSTALL_PREFIX} | Env: ${ENV_NAME}"
echo "Phases: build=$DO_BUILD extract=$DO_EXTRACT test=$DO_TEST"
echo ""

# -----------------------------------------------------------------------------
# Step -1: Build libcuvs tarball via build.sh standalone (--allgpuarch for multi-GPU)
# -----------------------------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
  info "Step -1: Building libcuvs tarball..."
  [[ ! -d "${CUVS_REPO}" ]] && fail "cuvs repo not found: ${CUVS_REPO}"
  [[ ! -f "${CUVS_REPO}/build.sh" ]] && fail "build.sh not found in ${CUVS_REPO}"
  set +u
  (cd "${CUVS_REPO}" && ./build.sh standalone $([ "$ALLGPUARCH" -eq 1 ] && echo --allgpuarch))
  set -u
  TARBALL="${CUVS_REPO}/build/libcuvs_c.tar.gz"
  [[ ! -f "${TARBALL}" ]] && fail "Tarball was not created at ${TARBALL}"
  info "Build complete: $(ls -lh "${TARBALL}" 2>/dev/null | awk '{print $5}')"
  echo ""
fi

# -----------------------------------------------------------------------------
# Step 0: Extract tarball to install prefix (required before test)
# -----------------------------------------------------------------------------
if [[ $DO_EXTRACT -eq 1 || $DO_TEST -eq 1 ]]; then
  [[ ! -f "${TARBALL}" ]] && fail "Tarball not found: ${TARBALL}. Run with --build first."
  info "Step 0: Extracting to ${INSTALL_PREFIX}..."
  mkdir -p "${INSTALL_PREFIX}"
  case "${TARBALL}" in
    *.tar.xz|*.txz) tar -xJf "${TARBALL}" -C "${INSTALL_PREFIX}" ;;
    *.tar.gz|*.tgz) tar -xzf "${TARBALL}" -C "${INSTALL_PREFIX}" ;;
    *)              tar -xf "${TARBALL}" -C "${INSTALL_PREFIX}" ;;
  esac
  [[ ! -f "${LIBCUVS_C}" ]] && fail "libcuvs_c.so not found after extract."
  info "Located ${LIBCUVS_C}"
  echo ""
fi

if [[ $DO_TEST -eq 1 ]]; then
RAPIDS_VER=$(detect_rapids_version)
info "Detected RAPIDS version: ${RAPIDS_VER}"

# -----------------------------------------------------------------------------
# Step 1: Ensure conda env exists (create from YAML or minimal deps if missing)
# -----------------------------------------------------------------------------
info "Step 1: Ensuring conda env '${ENV_NAME}'..."
eval "$(micromamba shell hook -s bash)" 2>/dev/null || true
if ! micromamba env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
  if [[ -f "${CUVS_YAML}" ]]; then
    set +u
    micromamba create -n "${ENV_NAME}" -f "${CUVS_YAML}" -y
    set -u
  else
    info "YAML not found, creating minimal env..."
    set +u
    micromamba create -n "${ENV_NAME}" -y -c rapidsai -c conda-forge -c nvidia \
      cuda-toolkit=13.1 "librmm=${RAPIDS_VER}.*" dlpack cmake "gcc_linux-64=14.*" ninja
    set -u
  fi
else
  info "Env '${ENV_NAME}' already exists."
fi
# Some conda/mamba activate.d scripts reference optional vars (e.g., NVCC_PREPEND_FLAGS).
# With `set -u` enabled, those can trigger "unbound variable". Temporarily relax -u.
set +u
micromamba activate "${ENV_NAME}"
set -u
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Verify NVIDIA driver and CUDA runtime (libcuda)
# -----------------------------------------------------------------------------
info "Step 2: Verifying NVIDIA driver & CUDA runtime..."
nvidia-smi &>/dev/null || fail "nvidia-smi failed. Install NVIDIA driver."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || nvidia-smi
if ! ldconfig -p 2>/dev/null | grep -q "libcuda"; then
  find "${CONDA_PREFIX:-/none}" -name "libcuda*.so*" 2>/dev/null | head -1 | grep -q . || fail "libcuda not found."
  info "libcuda via conda path"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 4: Check libcuvs_c.so dependencies via ldd; install librmm/pylibraft if missing
# -----------------------------------------------------------------------------
info "Step 4: Verifying libcuvs_c dependencies..."
MISSING=$(ldd "${LIBCUVS_C}" 2>/dev/null | grep "not found" || true)
if [[ -n "${MISSING}" ]]; then
  echo "${MISSING}"
  info "Installing missing deps (RAPIDS ${RAPIDS_VER})..."
  set +u
  micromamba install -n "${ENV_NAME}" -y -c rapidsai -c conda-forge -c nvidia \
    cuda-toolkit=13.1 "librmm=${RAPIDS_VER}.*" "pylibraft=${RAPIDS_VER}.*" nccl dlpack
  set -u
  export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
  MISSING=$(ldd "${LIBCUVS_C}" 2>/dev/null | grep "not found" || true)
  [[ -n "${MISSING}" ]] && echo "${MISSING}" && fail "Dependencies still missing. Try: micromamba create -n cuvs -f ${CUVS_YAML}"
fi
info "All dependencies resolve OK"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Python sanity check (ResourcesCreate, StreamSync, ResourcesDestroy)
# -----------------------------------------------------------------------------
info "Step 5: Python sanity check..."
CUVS_SANITY_PY="${SCRIPT_DIR}/.cuvs_sanity.py"
cat > "${CUVS_SANITY_PY}" << 'SANITY_EOF'
import ctypes, os, sys
def fail(m): print(f"FAILED: {m}"); sys.exit(1)
d = os.environ.get("CUVS_LIB_DIR", "")
if d: os.environ["LD_LIBRARY_PATH"] = d + ":" + os.environ.get("LD_LIBRARY_PATH", "")
try: ctypes.CDLL("libcuda.so.1")
except OSError: fail("libcuda.so.1 not found")
try: cuvs = ctypes.CDLL("libcuvs_c.so")
except OSError as e: fail(f"Could not load libcuvs_c.so: {e}")
cuvs.cuvsResourcesCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]; cuvs.cuvsResourcesCreate.restype = ctypes.c_int
cuvs.cuvsStreamSync.argtypes = [ctypes.c_void_p]; cuvs.cuvsStreamSync.restype = ctypes.c_int
cuvs.cuvsResourcesDestroy.argtypes = [ctypes.c_void_p]; cuvs.cuvsResourcesDestroy.restype = ctypes.c_int
r = ctypes.c_void_p()
cuvs.cuvsResourcesCreate(ctypes.byref(r)) or fail("cuvsResourcesCreate")
cuvs.cuvsStreamSync(r) or fail("cuvsStreamSync")
cuvs.cuvsResourcesDestroy(r) or fail("cuvsResourcesDestroy")
print("cuVS sanity check PASSED")
SANITY_EOF
export CUVS_LIB_DIR="${INSTALL_PREFIX}/lib"
python3 "${CUVS_SANITY_PY}" || { rm -f "${CUVS_SANITY_PY}"; fail "Python sanity failed."; }
rm -f "${CUVS_SANITY_PY}"
echo ""

# -----------------------------------------------------------------------------
# Step 6: C API test (compile minimal program, link libcuvs_c, run)
# -----------------------------------------------------------------------------
info "Step 6: C API test..."
MINIMAL_TEST_SRC="${SCRIPT_DIR}/.cuvs_minimal_test.c"
cat > "${MINIMAL_TEST_SRC}" << 'MINIMAL_EOF'
#include <cuvs/core/c_api.h>
#include <stdio.h>
int main(void) {
  cuvsResources_t res;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "FAIL: %s\n", cuvsGetLastErrorText()); return 1; }
  if (cuvsResourcesDestroy(res) != CUVS_SUCCESS) { fprintf(stderr, "FAIL: %s\n", cuvsGetLastErrorText()); return 1; }
  printf("PASS: cuvs C API\n"); return 0;
}
MINIMAL_EOF
# Ensure CUDA headers are available for compile
if [[ ! -f "${CONDA_PREFIX:-}/include/cuda_runtime.h" && ! -f "${CONDA_PREFIX:-}/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
  info "cuda_runtime.h not found; installing cuda-toolkit=13.1 for headers..."
  set +u
  micromamba install -n "${ENV_NAME}" -y -c rapidsai -c conda-forge -c nvidia cuda-toolkit=13.1
  set -u
  export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
fi
# Compose CUDA include/lib flags supporting split CUDA layout on conda (targets/x86_64-linux)
CUDA_INC_FLAGS="-I${INSTALL_PREFIX}/include"
for d in "${CONDA_PREFIX:-}/include" "${CONDA_PREFIX:-}/targets/x86_64-linux/include"; do
  [[ -d "${d}" ]] && CUDA_INC_FLAGS+=" -I${d}"
done
CUDA_LIB_FLAGS="-L${INSTALL_PREFIX}/lib -Wl,-rpath,${INSTALL_PREFIX}/lib"
for d in "${CONDA_PREFIX:-}/lib" "${CONDA_PREFIX:-}/targets/x86_64-linux/lib"; do
  [[ -d "${d}" ]] && CUDA_LIB_FLAGS+=" -L${d} -Wl,-rpath,${d}"
done
gcc -o "${SCRIPT_DIR}/.cuvs_minimal_test" "${MINIMAL_TEST_SRC}" ${CUDA_INC_FLAGS} ${CUDA_LIB_FLAGS} \
  -lcuvs_c -lcudart -lm
"${SCRIPT_DIR}/.cuvs_minimal_test" || fail "C API test failed"
rm -f "${MINIMAL_TEST_SRC}" "${SCRIPT_DIR}/.cuvs_minimal_test"
echo ""

# -----------------------------------------------------------------------------
# Step 7: Optional L2 distance example (if source exists in cuvs repo)
# -----------------------------------------------------------------------------
L2_SRC="${CUVS_REPO}/examples/c/src/L2_c_example.c"
COMMON_SRC="${CUVS_REPO}/examples/c/src/common.c"
if [[ -f "${L2_SRC}" ]] && [[ -f "${CUVS_REPO}/examples/c/src/common.h" ]]; then
  info "Step 7: L2 example..."
  CUDA_INC_FLAGS="-I${INSTALL_PREFIX}/include"
  for d in "${CONDA_PREFIX:-}/include" "${CONDA_PREFIX:-}/targets/x86_64-linux/include"; do
    [[ -d "${d}" ]] && CUDA_INC_FLAGS+=" -I${d}"
  done
  CUDA_LIB_FLAGS="-L${INSTALL_PREFIX}/lib -Wl,-rpath,${INSTALL_PREFIX}/lib"
  for d in "${CONDA_PREFIX:-}/lib" "${CONDA_PREFIX:-}/targets/x86_64-linux/lib"; do
    [[ -d "${d}" ]] && CUDA_LIB_FLAGS+=" -L${d} -Wl,-rpath,${d}"
  done
  EXTRA_SRCS=""
  [[ -f "${COMMON_SRC}" ]] && EXTRA_SRCS="${COMMON_SRC}"
  gcc -o "${SCRIPT_DIR}/.cuvs_l2_test" ${EXTRA_SRCS} "${L2_SRC}" ${CUDA_INC_FLAGS} ${CUDA_LIB_FLAGS} -lcuvs_c -lcudart -lm -fgnu89-inline
  "${SCRIPT_DIR}/.cuvs_l2_test" && info "L2 PASSED" || info "L2 failed (optional)"
  rm -f "${SCRIPT_DIR}/.cuvs_l2_test"
fi

fi
# -----------------------------------------------------------------------------
# Summary: print completion and usage hints for downstream projects
# -----------------------------------------------------------------------------
PHASES=""
[[ $DO_BUILD -eq 1 ]] && PHASES+="build "; [[ $DO_EXTRACT -eq 1 ]] && PHASES+="extract "; [[ $DO_TEST -eq 1 ]] && PHASES+="test "
echo "=== cuVS complete: ${PHASES}==="
echo "Use: export CUVS_PREFIX=${INSTALL_PREFIX}"
echo "     export LD_LIBRARY_PATH=\${CUVS_PREFIX}/lib:\${LD_LIBRARY_PATH}"
echo "     Compile: -I\${CUVS_PREFIX}/include -L\${CUVS_PREFIX}/lib -lcuvs_c"
