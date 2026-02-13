Installation
============

The cuVS software development kit provides APIs for C, C++, Python, and Rust languages. This guide outlines how to install the pre-compiled packages, build it from source, and use it in downstream applications.

- `Installing pre-compiled packages`_

  * `C, C++, and Python through Conda`_

  * `Python through Pip`_

- `Build from source`_

  * `Prerequisites`_

  * `Create a build environment`_

  * `C and C++ Libraries`_

    * `Tarball (Build from Source)`_

    * `Building the Googletests`_

  * `Python Library`_

  * `Rust Library`_

  * `Using CMake Directly`_

- `Build Documentation`_

- `Installation & Environment Verification FAQ`_


Installing Pre-compiled Packages
--------------------------------

**Note:** The cuVS pre-compiled packages are available for **Linux** only (x86_64 and aarch64 architectures). Native Windows support is not available at this time. On Windows, use **WSL2** with GPU passthrough. See the `RAPIDS WSL2 guide <https://rapids.ai/start.html#wsl2>`_.

C, C++, and Python through Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install the pre-compiled C, C++, and Python packages is through conda. You can get a minimal conda installation with `miniforge <https://github.com/conda-forge/miniforge>`__.

Use the following commands, depending on your CUDA version, to install cuVS packages (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command and can be enabled using `this guide <https://conda.github.io/conda-libmamba-solver/user-guide/>`_.

C/C++ Package
~~~~~~~~~~~~~

.. code-block:: bash

   # CUDA 13
   conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.1

   # CUDA 12
   conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9

Python Package
~~~~~~~~~~~~~~

.. code-block:: bash

   # CUDA 13
   conda install -c rapidsai -c conda-forge cuvs cuda-version=13.1

   # CUDA 12
   conda install -c rapidsai -c conda-forge cuvs cuda-version=12.9

Python through Pip
^^^^^^^^^^^^^^^^^^

The cuVS Python package can also be `installed through pip <https://docs.rapids.ai/install#pip>`_.

.. code-block:: bash

    # CUDA 13
    pip install cuvs-cu13 --extra-index-url=https://pypi.nvidia.com

    # CUDA 12
    pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

Note: these packages statically link the C and C++ libraries so the `libcuvs` and `libcuvs_c` shared libraries won't be readily available to use in your code.

Tarball
^^^^^^^

The cuVS tarball includes the C and C++ APIs along with core dependencies (RAFT, RMM). It requires a compatible CUDA Toolkit, NVIDIA driver, and system libraries such as NCCL, OpenMP, and GLIBC.

Two pre-built tarballs are available:

1. **Open Source:** Download from `developer.nvidia.com <https://developer.nvidia.com/cuvs>`_.
2. **Enterprise Supported:** Download from `NVIDIA NGC <https://catalog.ngc.nvidia.com/>`_ (requires NVAIE license).

Build from source
-----------------

The core cuVS source code is written in C++ and wrapped through a C API. The C API is wrapped around the C++ APIs and the other supported languages are built around the C API.


Prerequisites
^^^^^^^^^^^^^

- CMake 3.26.4+
- GCC 9.3+ (11.4+ recommended)
- CUDA Toolkit 12.2+
- Ampere architecture or better (compute capability >= 8.0)

Create a build environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda environment scripts are provided for installing the necessary dependencies to build cuVS from source. It is preferred to use `mamba`, as it provides significant speedup over `conda`:

.. code-block:: bash

    conda env create --name cuvs -f conda/environments/all_cuda-131_arch-$(uname -m).yaml
    conda activate cuvs

The recommended way to build and install cuVS from source is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides CMake options for building and installing the headers, tests, benchmarks, and the pre-compiled shared library.


C and C++ libraries
^^^^^^^^^^^^^^^^^^^

The C and C++ shared libraries are built together using the following arguments to `build.sh`:

.. code-block:: bash

    ./build.sh libcuvs

In above example the `libcuvs.so` and `libcuvs_c.so` shared libraries are installed by default into `$INSTALL_PREFIX/lib`. To disable this, pass `-n` flag.

Once installed, the shared libraries, headers (and any dependencies downloaded and installed via `rapids-cmake`) can be uninstalled using `build.sh`:

.. code-block:: bash

    ./build.sh libcuvs --uninstall


Multi-GPU features
^^^^^^^^^^^^^^^^^^

To disable the multi-gpu features run :

.. code-block:: bash

    ./build.sh libcuvs --no-mg


Tarball (Build from Source)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a standalone tarball (``libcuvs_c.tar.gz``) from source, use the ``standalone`` target with ``--allgpuarch`` to build for all supported GPU architectures:

.. code-block:: bash

    ./build.sh standalone --allgpuarch -v

The tarball will be created in the ``cpp/build/`` directory.


Building the Googletests
~~~~~~~~~~~~~~~~~~~~~~~~

Compile the C and C++ Googletests using the `tests` target in `build.sh`.

.. code-block:: bash

    ./build.sh libcuvs tests

The tests will be written to the build directory, which is `cpp/build/` by default, and they will be named `*_TEST`.

It can take some time to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`. Make sure to pass the `-n` flag so the tests are not installed.

.. code-block:: bash

    ./build.sh libcuvs tests -n --limit-tests=NEIGHBORS_TEST;CAGRA_C_TEST

Python library
^^^^^^^^^^^^^^

The Python library should be built and installed using the `build.sh` script:

.. code-block:: bash

    ./build.sh python

The Python packages can also be uninstalled using the `build.sh` script:

.. code-block:: bash

    ./build.sh python --uninstall

Go library
^^^^^^^^^^

After building the C and C++ libraries, the Golang library can be built with the following command:

.. code-block:: bash

    export CUDA_HOME="/usr/local/cuda" # or wherever your CUDA installation is.
    export CGO_CFLAGS="-I${CONDA_PREFIX}/include -I${CUDA_HOME}/include"
    export CGO_LDFLAGS="-L${CONDA_PREFIX}/lib -lcudart -lcuvs -lcuvs_c"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export CC=clang

    ./build.sh go

Rust library
^^^^^^^^^^^^

The Rust bindings can be built with

.. code-block:: bash

    ./build.sh rust

Using CMake directly
^^^^^^^^^^^^^^^^^^^^

When building cuVS from source, the `build.sh` script offers a nice wrapper around the `cmake` commands to ease the burdens of manually configuring the various available cmake options. When more fine-grained control over the CMake configuration is desired, the `cmake` command can be invoked directly as the below example demonstrates.

The `CMAKE_INSTALL_PREFIX` installs cuVS into a specific location. The example below installs cuVS into the current Conda environment:

.. code-block:: bash

    cd cpp
    mkdir build
    cd build
    cmake -D BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
    make -j<parallel_level> install

cuVS has the following configurable cmake flags available:

.. list-table:: CMake Flags

 * - Flag
   - Possible Values
   - Default Value
   - Behavior

 * - BUILD_TESTS
   - ON, OFF
   - ON
   - Compile Googletests

 * - CUDA_ENABLE_KERNELINFO
   - ON, OFF
   - OFF
   - Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer`

 * - CUDA_ENABLE_LINEINFO
   - ON, OFF
   - OFF
   - Enable the `-lineinfo` option for nvcc

 * - CUDA_STATIC_RUNTIME
   - ON, OFF
   - OFF
   - Statically link the CUDA runtime

 * - CUDA_STATIC_MATH_LIBRARIES
   - ON, OFF
   - OFF
   - Statically link the CUDA math libraries

 * - DETECT_CONDA_ENV
   - ON, OFF
   - ON
   - Enable detection of conda environment for dependencies

 * - CUVS_NVTX
   - ON, OFF
   - OFF
   - Enable NVTX markers


Build documentation
^^^^^^^^^^^^^^^^^^^

The documentation requires that the C, C++ and Python libraries have been built and installed. The following will build the docs along with the necessary libraries:

.. code-block:: bash

    ./build.sh libcuvs python docs


Installation & Environment Verification FAQ
--------------------------------------------

This guide helps verify that NVIDIA drivers, CUDA runtime, and cuVS are correctly installed and compatible.

Quick Summary
^^^^^^^^^^^^^

If you can successfully run the Python cuVS sanity check below and see: **cuVS sanity check PASSED**

Then:

- NVIDIA driver is working
- CUDA runtime is accessible
- cuVS is correctly installed and loadable
- Basic GPU execution works

Prerequisites
^^^^^^^^^^^^^

Same as the installation requirements (Linux, NVIDIA GPU) described in the sections above. Additionally, Python 3.x is required to run the sanity check script.

Library Path Setup (Commonly Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If cuVS is installed in a non-standard location, the dynamic loader must be told where to find it:

.. code-block:: bash

    export LD_LIBRARY_PATH=/path/to/cuvs/lib:$LD_LIBRARY_PATH


Step 1: Locate libcuvs_c.so
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running any checks, confirm where cuVS is installed.

Option A: Using ldconfig (preferred)

.. code-block:: bash

    ldconfig -p | grep libcuvs

Example output:

    libcuvs_c.so (libc6,x86-64) => /usr/local/cuvs/lib/libcuvs_c.so

Option B: Using find (if not registered)

.. code-block:: bash

    sudo find / -name "libcuvs_c.so" 2>/dev/null

Once located, ensure the directory containing `libcuvs_c.so` is included in ``LD_LIBRARY_PATH``.


Step 2: Verify NVIDIA Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    nvidia-smi

**Expected result:** GPU is listed, driver version is shown, no error messages.

**Common failure:** ``NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`` — indicates a driver installation or kernel mismatch.


Step 3: Verify CUDA Runtime Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cuVS requires the CUDA runtime provided by the NVIDIA driver.

.. code-block:: bash

    ldconfig -p | grep libcuda

**Expected result:** ``libcuda.so.1 (libc6,x86-64) => /usr/lib/...``

If nothing appears, the NVIDIA driver is not correctly installed.


Step 4: Verify cuVS Shared Library Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once `libcuvs_c.so` has been located, run:

.. code-block:: bash

    ldd /full/path/to/libcuvs_c.so

**Expected result:** No lines containing ``not found``; ``libcudart.so``, ``libstdc++.so`` resolve correctly.

**Optional (advanced):**

.. code-block:: bash

    objdump -p /full/path/to/libcuvs_c.so | grep NEEDED

Shows required shared library versions.


Step 5: cuVS Runtime Sanity Check (Python)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This test dynamically loads cuVS, creates GPU resources, synchronizes a CUDA stream, and cleans up.

Copy-paste script: ``cuvs_sanity.py``

.. code-block:: python

    import ctypes
    import sys

    def fail(msg):
        print(f"FAILED: {msg}")
        sys.exit(1)

    # Check NVIDIA driver
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        fail("libcuda.so.1 not found (NVIDIA driver missing or not loaded)")

    # Load cuVS
    try:
        cuvs = ctypes.CDLL("libcuvs_c.so")
    except OSError as e:
        fail(f"Could not load libcuvs_c.so: {e}")

    # Define function signatures
    try:
        cuvs.cuvsResourcesCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cuvs.cuvsResourcesCreate.restype = ctypes.c_int

        cuvs.cuvsStreamSync.argtypes = [ctypes.c_void_p]
        cuvs.cuvsStreamSync.restype = ctypes.c_int

        cuvs.cuvsResourcesDestroy.argtypes = [ctypes.c_void_p]
        cuvs.cuvsResourcesDestroy.restype = ctypes.c_int
    except AttributeError as e:
        fail(f"Missing cuVS symbol: {e}")

    # Create resources
    res = ctypes.c_void_p()
    if cuvs.cuvsResourcesCreate(ctypes.byref(res)) == 0:
        fail("cuvsResourcesCreate failed")

    # Sync stream
    if cuvs.cuvsStreamSync(res) == 0:
        fail("cuvsStreamSync failed")

    # Destroy resources
    if cuvs.cuvsResourcesDestroy(res) == 0:
        fail("cuvsResourcesDestroy failed")

    print("cuVS sanity check PASSED")


Step 6: Run the Test
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python3 cuvs_sanity.py

**Expected output:** ``cuVS sanity check PASSED``


Common Errors & What They Mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Common Errors
   :header-rows: 1

   * - Error Message
     - Likely Cause
   * - libcuda.so.1 not found
     - NVIDIA driver not installed or not loaded
   * - Could not load libcuvs_c.so
     - LD_LIBRARY_PATH not set or cuVS not installed
   * - Missing cuVS symbol
     - cuVS version mismatch
   * - cuvsResourcesCreate failed
     - CUDA runtime / driver incompatibility
   * - cuvsStreamSync failed
     - GPU execution failure


Optional: Debug Library Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced troubleshooting:

.. code-block:: bash

    LD_DEBUG=libs python3 cuvs_sanity.py

This prints exactly which shared libraries are loaded and from where.


FAQ
^^^

- **Do I need gcc, nvcc, or CUDA headers?** No. This check uses Python's dynamic loader and requires no compilation.

- **Does this test run on the GPU?** Yes. It creates cuVS resources and synchronizes a CUDA stream.

- **Is this safe to run on production systems?** Yes. It performs a minimal, non-destructive GPU operation.

- **Can this be used in containers?** Yes, as long as NVIDIA Container Toolkit is installed and GPU access is enabled (``--gpus ..``).


Support Checklist
^^^^^^^^^^^^^^^^^

When reporting issues, please include:

- Output of ``nvidia-smi``
- Output of ``ldd libcuvs_c.so``
- Output of ``python3 cuvs_sanity.py``
