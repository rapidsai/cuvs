# cuVS C Library Setup & Verification Tutorial

This tutorial walks you through installing and verifying the cuVS (CUDA Vector Search) C library using the setup script. It is organized into separate phases you can run independently or together.

**Files in this folder:** Run `setup_and_test_cuvs_c.sh` and `build_libcuvs_c.sh` from the `notebooks/` directory of your cuVS clone. Paths auto-detect the repo root.

---

## Overview

**Prerequisites:**
- micromamba (or conda)
- Python 3.x
- NVIDIA GPU with compatible driver
- (Optional) Build phase requires the cuVS source repo

---

## Section 1: Configuration & Options

### Default Paths

Paths auto-detect when scripts run from `notebooks/`:

| Variable | Default | Override |
|----------|---------|----------|
| Tarball | `{repo_root}/libcuvs_c.tar.gz` | Pass tarball path as last argument |
| Install prefix | `/usr/local` | `CUVS_INSTALL_PREFIX` |
| Conda env | `cuvs` | `CUVS_CONSUMER_ENV` |
| cuVS repo | `notebooks/..` (repo root) | `CUVS_REPO` |

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--build` | Build the libcuvs tarball from source |
| `--extract` | Extract the tarball to the install prefix |
| `--test` | Run verification tests (env setup, sanity checks, C API test) |
| `--allgpuarch` | Build for all GPU architectures (multi-GPU support) |
| `--all` | Equivalent to `--build --extract --test` |
| `-h`, `--help` | Show usage |

### Basic Usage

```bash
cd notebooks
./setup_and_test_cuvs_c.sh [OPTIONS] [tarball]
```

If no options are given, the script runs build, extract, and test by default.

---

## Section 2: Build Phase (--build)

**Purpose:** Compile cuVS from source and create a tarball.

**When to run:** When you have the cuVS repo and want to produce `libcuvs_c.tar.gz` yourself.

**Steps:**
1. Ensure `build_libcuvs_c.sh` exists in `notebooks/` (same directory)
2. Run the build from the repo root or let paths auto-detect
3. Package output into `libcuvs_c.tar.gz` at repo root

**Example:**
```bash
cd notebooks
./setup_and_test_cuvs_c.sh --build
```

For all GPU architectures:
```bash
./setup_and_test_cuvs_c.sh --build --allgpuarch
```

---

## Section 3: Extract Phase (--extract)

**Purpose:** Unpack the tarball into the install prefix so the library and headers are available.

**When to run:** After you have a tarball (from build or copied from elsewhere, e.g., via `scp` from your laptop).

**Example:**
```bash
cd notebooks
./setup_and_test_cuvs_c.sh --extract
```

With a custom tarball path:
```bash
./setup_and_test_cuvs_c.sh --extract /path/to/libcuvs_c.tar.gz
```

**Installing to /usr/local (requires sudo):**
```bash
sudo ./setup_and_test_cuvs_c.sh --extract
```

---

## Section 4: Test Phase – Environment Setup

**Purpose:** Ensure the conda/micromamba environment exists and is activated for testing.

**Steps:**
1. Detect RAPIDS version (from `version_config.h` or `VERSION` file)
2. Create the conda env `cuvs` if it does not exist
3. Activate the env and set `LD_LIBRARY_PATH`

---

## Section 5: Test Phase – NVIDIA Driver & CUDA

**Purpose:** Verify the system has a working NVIDIA driver and CUDA runtime.

**Checks:** `nvidia-smi`, GPU info, `libcuda` availability.

---

## Section 6: Test Phase – Dependency Resolution

**Purpose:** Ensure all shared library dependencies of `libcuvs_c.so` can be resolved via `ldd`. Missing deps trigger auto-install of cuda-toolkit, librmm, pylibraft, nccl, dlpack.

---

## Section 7: Test Phase – Python Sanity Check

**Purpose:** Verify that the C library can be loaded and basic APIs work from Python.

**Success:** Prints `cuVS sanity check PASSED`.

---

## Section 8: Test Phase – C API Compilation Test

**Purpose:** Compile a minimal C program that uses the cuVS C API and run it.

**Success:** Prints `PASS: cuvs C API`.

---

## Section 9: Test Phase – Optional L2 Example

**Purpose:** Build and run the L2 distance example from `examples/c/src/` (if present).

**Success:** Prints `L2 PASSED`. Failures are reported as optional.

---

## Section 10: Using cuVS in Your Project

After a successful setup:

```bash
export CUVS_PREFIX=/usr/local
export LD_LIBRARY_PATH=${CUVS_PREFIX}/lib:${LD_LIBRARY_PATH}
```

**Compile your C/C++ code:**
```bash
gcc -o myapp myapp.c \
  -I${CUVS_PREFIX}/include \
  -L${CUVS_PREFIX}/lib \
  -lcuvs_c -lcudart -lm \
  -Wl,-rpath,${CUVS_PREFIX}/lib
```

---

## Appendix: Copying a Tarball from Your Laptop

**On your laptop:**
```bash
scp ~/Downloads/libcuvs_c.tar.gz user@<SERVER>:path/to/cuvs/
```

**On the server:**
```bash
cd path/to/cuvs/notebooks
./setup_and_test_cuvs_c.sh --extract --test
```

---

*See [cuVS documentation](https://docs.rapids.ai/api/cuvs/stable/) and [GitHub](https://github.com/rapidsai/cuvs) for more.*
